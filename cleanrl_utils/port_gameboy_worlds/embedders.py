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


def extract_patches(x, kernel_size=4):
    assert kernel_size in [2, 4, 8, 16]
    max_h, max_w = 144, 160
    n_h = max_h // kernel_size
    n_w = max_w // kernel_size
    n_patches = n_h * n_w
    if len(x.shape) == 2:  # then its H x W
        x = x.reshape(1, 1, x.shape[0], x.shape[1])
    elif len(x.shape) == 3:  # then its C x H x W or H x W x C
        if x.shape[0] == 1:
            x = x.reshape(1, 1, x.shape[1], x.shape[2])
        elif x.shape[2] == 1:
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
    elif len(x.shape) == 4:  # then its N x C x H x W or N x H x W x C
        if x.shape[1] == 1:
            pass  # already in N x C x H x W
        elif x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)  # convert from N x H x W x C to N x C x H x W
    else:
        raise ValueError(f"Unexpected input shape {x.shape}")
    patches = x.unfold(2, kernel_size, kernel_size).unfold(
        3, kernel_size, kernel_size
    )  # (N, C, n_h, n_w, kernel_size, kernel_size)
    # reshape to (N, n_patches, C, kernel_size, kernel_size)
    patches = patches.contiguous().view(
        patches.shape[0], -1, 1, kernel_size, kernel_size
    )
    return patches


def reconstruct_from_patches(patches, kernel_size=4):
    batch_size = patches.shape[0]
    new_tensor = patches.view(
        batch_size, 1, 144 // kernel_size, 160 // kernel_size, kernel_size, kernel_size
    )
    new_tensor = (
        new_tensor.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, 1, 144, 160)
    )
    return new_tensor


class PatchProjection(nn.Module):
    """
    Works with the 144 x 160 pixel observations from gameboy_worlds.
    Divides the image into 16x16 patches, applies a random linear projection to each patch, and concatenates the results.
    """

    def __init__(self, seed, normalized_observations=True, kernel_size=4, patch_dim=4):
        super().__init__()
        self.normalized_observations = normalized_observations
        self.seed = seed
        self.kernel_size = kernel_size
        self.patch_dim = patch_dim
        self.patch_dim = 4
        n_h = 144 // self.kernel_size
        n_w = 160 // self.kernel_size
        n_patches = n_h * n_w
        self.output_dim = n_patches * self.patch_dim
        self.make_network()

    def make_network(self):
        torch.manual_seed(
            42
        )  # always force so that the same random projection is used even across different scripts
        linear = nn.Linear(self.kernel_size**2, self.patch_dim, bias=False)
        # initialize with a values from standard normal
        torch.nn.init.normal_(linear.weight, mean=0.0, std=1.0)
        self.project = nn.Sequential(
            nn.Flatten(),
            linear,
        )
        self.dtype = linear.weight.dtype
        torch.manual_seed(self.seed)

    def forward(self, x):
        patches = extract_patches(
            x, kernel_size=self.kernel_size
        )  # (N, n_patches, 1, kernel_size, kernel_size)
        vector = self.project(patches)  # (N, n_patches, patch_dim)
        vector = vector.view(vector.shape[0], -1)  # (N, n_patches * patch_dim)
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
    def __init__(self, seed, normalized_observations=True, kernel_size=8, patch_dim=4):
        super().__init__()
        torch.manual_seed(42)
        n_h = 144 // kernel_size
        n_w = 160 // kernel_size
        n_patches = n_h * n_w
        hidden_dim = patch_dim * n_patches
        self.kernel_size = kernel_size
        self.patch_dim = patch_dim
        self.n_patches = n_patches
        self.patch_norm = nn.BatchNorm2d(1, affine=False)
        self.internal_norm = nn.LayerNorm(patch_dim)
        self.norm2 = nn.BatchNorm2d(1, affine=False)
        encoder_cnn_chain = nn.Sequential(
            layer_init(nn.Conv2d(1, 2, kernel_size=1, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(2, 4, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(4, 4, kernel_size=2, stride=1)),
            nn.ReLU(),
        )
        self.decoder_cnn_chain = nn.Sequential(
            layer_init(nn.ConvTranspose2d(4, 4, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(4, 2, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(2, 1, kernel_size=1, stride=1)),
        )
        dummy_input = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        with torch.no_grad():
            dummy_output = encoder_cnn_chain(dummy_input)
        self.dummy_output_shape = (
            dummy_output.shape[1],
            dummy_output.shape[2],
            dummy_output.shape[3],
        )
        chain_dim = dummy_output.reshape(1, -1).shape[1]
        self.encoder = nn.Sequential(
            *encoder_cnn_chain,
            nn.Flatten(),
            layer_init(nn.Linear(chain_dim, patch_dim)),
            nn.Sigmoid(),
            self.internal_norm,
        )
        self.decoder_linear = layer_init(nn.Linear(patch_dim, chain_dim))
        self.output_dim = hidden_dim
        self.normalized_observations = normalized_observations
        torch.manual_seed(seed)
        self.reset()

    def embed_patches(self, x):
        # x is (N, n_patches, 1, kernel_size, kernel_size)
        N, n_patches, _, _, _ = x.shape
        x = x.view(-1, 1, self.kernel_size, self.kernel_size)
        normed_x = self.patch_norm(x)
        encoded = self.encoder(normed_x)  # (N * n_patches, patch_dim)
        if self.normalized_observations:
            encoded = nn.functional.normalize(encoded, dim=-1)
        encoded = encoded.view(N, n_patches, -1)  # (N, n_patches, patch_dim)
        normed_x = normed_x.view(N, n_patches, 1, self.kernel_size, self.kernel_size)
        return encoded, normed_x

    def do_embed(self, x):
        patches = extract_patches(
            x, kernel_size=self.kernel_size
        )  # (N, n_patches, 1, kernel_size, kernel_size)
        patch_embeddings, normed_x = self.embed_patches(
            patches
        )  # (N, n_patches, patch_dim)
        return (
            patch_embeddings,
            normed_x,
        )  # (N, n_patches * patch_dim), (N, n_patches, 1, kernel_size, kernel_size)

    def forward(self, x):
        patch_embeddings, normed_x = self.do_embed(
            x
        )  # (N, n_patches, patch_dim), (N, n_patches, 1, kernel_size, kernel_size)
        bridged = self.decoder_linear(patch_embeddings)  # (N, n_patches, chain_dim)
        bridged = bridged.view(-1, *self.dummy_output_shape)  # (N * n_patches, C, H, W)
        reconstructed = self.decoder_cnn_chain(bridged)
        normed = self.norm2(reconstructed)
        reconstructed = reconstructed.view(
            -1, self.n_patches, 1, self.kernel_size, self.kernel_size
        )  # (N, n_patches, 1, kernel_size, kernel_size)
        reconstructed = reconstruct_from_patches(
            reconstructed, kernel_size=self.kernel_size
        )  # (N, 1, 144, 160)
        reconstructed_normed_x = reconstruct_from_patches(
            normed_x, kernel_size=self.kernel_size
        )  # (N, 1, 144, 160)
        return reconstructed, reconstructed_normed_x

    def denormalize_reconstruction(self, reconstruction):
        patches = extract_patches(
            reconstruction, kernel_size=self.kernel_size
        )  # (N, n_patches, 1, kernel_size, kernel_size)
        # undo the patch norm
        patches = patches.view(-1, 1, self.kernel_size, self.kernel_size)
        denormed = (
            patches * (self.patch_norm.running_var.sqrt() + self.patch_norm.eps)
        ) + self.patch_norm.running_mean
        denormed = denormed.view(
            -1, self.n_patches, 1, self.kernel_size, self.kernel_size
        )
        reconstructed = reconstruct_from_patches(denormed, kernel_size=self.kernel_size)
        return reconstructed

    def embed(self, items: List[np.ndarray]) -> torch.Tensor:
        with torch.no_grad():
            batch_tensor = torch.tensor(
                items.reshape(-1, 1, 144, 160),
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )
            # 0 to get only patch_embeddings, not normed_x
            embeddings = self.do_embed(batch_tensor)[0].view(
                batch_tensor.shape[0], -1
            )  # (N, n_patches, patch_dim) -> (N, n_patches * patch_dim)
            return embeddings  # + noise

    def load(self, path):
        loaded_state = torch.load(path + "/observation_encoder.pt")
        self.load_state_dict(loaded_state)
        print(f"Loaded CNN embedder from {path}")

    def reset(self):
        self.noise = 0.0001 * torch.randn(
            self.output_dim, device=next(self.parameters()).device
        )  # not used. Its there cause im curious
