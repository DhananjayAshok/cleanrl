import numpy as np
import torch
import torch.nn as nn
from typing import List

from .utils import FRAME_STACK


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_gameboy_cnn_chain(stacked=True):
    use_stack = FRAME_STACK if stacked else 1
    return nn.Sequential(
        layer_init(
            nn.Conv2d(use_stack, 32, kernel_size=16, stride=16)
        ),  # (batch_size, 32, 9, 10)
        nn.ReLU(),
        layer_init(
            nn.Conv2d(32, 64, kernel_size=4, stride=2)
        ),  # (batch_size, 64, 3, 4)
        nn.ReLU(),
        layer_init(
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        ),  # (batch_size, 64, 1, 2)
        nn.ReLU(),
        nn.Flatten(),  # (batch_size, 128)
    )


def invert_gameboy_cnn_chain(stacked=True):
    use_stack = FRAME_STACK if stacked else 1
    return nn.Sequential(
        nn.Unflatten(1, (64, 1, 2)),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=(1, 0)),
        nn.ReLU(),
        nn.ConvTranspose2d(32, use_stack, kernel_size=16, stride=16),
    )


class PatchProjection(nn.Module):
    """
    Works with the 144 x 160 pixel observations from gameboy_worlds.
    Divides the image into 16x16 patches, applies a random linear projection to each patch, and concatenates the results.
    """

    def __init__(self, seed, normalized_observations=True):
        super().__init__()
        self.normalized_observations = normalized_observations
        self.seed = seed
        self.make_network()
        self.output_dim = 90 * 4
        self.dtype = self.project[0].weight.dtype

    def make_network(self):
        torch.manual_seed(
            42
        )  # always force so that the same random projection is used even across different scripts
        self.project = nn.Sequential(
            nn.Conv2d(
                1,
                1,
                kernel_size=8,
                stride=8,  # 8x8 patches with no overlap to get 4 snapshots of each of the gameboys 16x16 cells.
            ),
            nn.Flatten(),
        )
        torch.manual_seed(self.seed)

    def forward(self, x):
        vector = self.project(x)
        if self.normalized_observations:
            normalized = nn.functional.normalize(vector, dim=-1)
            return normalized
        return vector

    def embed(self, items: List[np.ndarray]) -> torch.Tensor:
        with torch.no_grad():
            if not isinstance(items, torch.Tensor):
                batch_tensor = torch.tensor(
                    items.reshape(-1, 1, 144, 160),
                )
            else:
                batch_tensor = items.reshape(-1, 1, 144, 160)
            batch_tensor = batch_tensor.to(self.dtype).to(
                next(self.parameters()).device
            )
            embeddings = self(batch_tensor)
            return embeddings

    def reset(self):
        self.make_network()


class CNNEmbedder(nn.Module):
    def __init__(self, seed, hidden_dim=128, normalized_observations=True):
        super().__init__()
        torch.manual_seed(42)
        self.norm1 = nn.BatchNorm2d(1, affine=False)
        self.internal_norm = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm2d(1, affine=False)
        encoder_cnn_chain = get_gameboy_cnn_chain(stacked=False)
        dummy_input = torch.zeros(1, 1, 144, 160)
        with torch.no_grad():
            dummy_output = encoder_cnn_chain(dummy_input)
        chain_dim = dummy_output.shape[1]
        self.encoder = nn.Sequential(
            *get_gameboy_cnn_chain(stacked=False),
            layer_init(nn.Linear(chain_dim, hidden_dim)),
            nn.Sigmoid(),
            self.internal_norm,
        )
        self.decoder = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, chain_dim)),
            *invert_gameboy_cnn_chain(stacked=False),
        )
        self.output_dim = hidden_dim
        self.normalized_observations = normalized_observations
        torch.manual_seed(seed)
        self.reset()

    def do_embed(self, x):
        normed = self.norm1(x)
        raw = self.encoder(normed)
        if self.normalized_observations:
            normalized = nn.functional.normalize(
                raw, dim=-1
            )  # Normalize the output embeddings
            return normalized
        return raw

    def forward(self, x):
        embedding = self.do_embed(x)
        unembed = self.decoder(embedding)
        normed = self.norm2(unembed)
        return normed

    def embed(self, items: List[np.ndarray]) -> torch.Tensor:
        with torch.no_grad():
            batch_tensor = torch.tensor(
                items.reshape(-1, 1, 144, 160),
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )
            embeddings = self.do_embed(batch_tensor)
            return embeddings  # + noise

    def load(self, path):
        loaded_state = torch.load(path + "observation_encoder.pt")
        self.load_state_dict(loaded_state)
        print(f"Loaded CNN embedder from {path}")

    def reset(self):
        self.noise = 0.0001 * torch.randn(
            self.output_dim, device=next(self.parameters()).device
        )  # not used. Its there cause im curious
