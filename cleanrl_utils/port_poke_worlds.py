from poke_worlds import get_environment
from poke_worlds.emulation import StateParser
import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import torch
import torch.nn as nn
from typing import List
from sklearn.cluster import MiniBatchKMeans, KMeans


class OneOfToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Calculate total actions across all sub-spaces
        # Example: OneOf([Discrete(2), Discrete(3)]) -> total 5
        self.internal_env = env
        self.sub_spaces = env.action_space.spaces
        self.total_actions = sum(s.n for s in self.sub_spaces)
        self.action_space = Discrete(self.total_actions)

    def action(self, action):
        # Map the single integer back to (choice, sub_action)
        offset = 0
        for i, space in enumerate(self.sub_spaces):
            if action < offset + space.n:
                return (i, action - offset)
            offset += space.n
        print("Action mapping error!")
        return (0, 0)  # Fallback

    def get_high_level_action(self, action):
        # Map the single integer back to choice only
        action = self.action(action)
        high_level_action, kwargs = (
            self.internal_env._controller._space_action_to_high_level_action(action)
        )
        return high_level_action, kwargs

    def set_render_mode(self, mode):
        self.internal_env.render_mode = mode


def parse_pokeworlds_id_string(id_string):
    """

    :param id_string: should be in format "poke_worlds-game-environment_variant-controller_variant-max_steps-save_video"
    Example: poke_worlds-pokemon_red-starter_explore-low_level-20-true
    :return: tuple (game, environment_variant, controller_variant, max_steps, save_video)
    """
    #
    parts = id_string.split("-")
    if len(parts) != 6 or parts[0] != "poke_worlds":
        raise ValueError(
            f"Invalid ID string format. Expected 'poke_worlds-game-environment_variant-controller_variant-max_steps-save_video'. Got {id_string}"
        )
    _, game, environment_variant, controller_variant, max_steps_str, save_video_str = (
        parts
    )
    if not max_steps_str.isdigit():
        raise ValueError(
            f"Invalid max_steps value. Expected an integer. Got {max_steps_str}"
        )
    max_steps = int(max_steps_str)
    save_video = save_video_str.lower() == "true"
    return game, environment_variant, controller_variant, max_steps, save_video


def get_poke_worlds_environment(id_string, render_mode=None):
    game, environment_variant, controller_variant, max_steps, save_video = (
        parse_pokeworlds_id_string(id_string)
    )
    env = get_environment(
        game=game,
        controller_variant=controller_variant,
        environment_variant=environment_variant,
        max_steps=max_steps,
        headless=True,
        save_video=save_video,
    )
    env = OneOfToDiscreteWrapper(env)
    if render_mode is not None:
        env.set_render_mode(render_mode)
    return env


def poke_worlds_make_env(env_id, seed, idx, capture_video, run_name, gamma=0.99):
    def thunk():
        if capture_video and idx == 0:
            env = get_poke_worlds_environment(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = get_poke_worlds_environment(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)

        if seed is not None:
            env.action_space.seed(seed)
        return env

    return thunk


class PatchProjection:
    """
    Works with the 144 x 160 pixel observations from poke_worlds.
    Divides the image into 16x16 patches, applies a random linear projection to each patch, and concatenates the results.
    """

    cell_reduction_dimension = 8  # hidden dimension is cell_reduction_dimension * 90

    def __init__(self):
        start = 16 * 16
        end = self.cell_reduction_dimension
        my_local_rng = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        my_local_rng.manual_seed(42)
        breakpoint()  # Change this to a CNN-based projector.
        step1 = nn.Linear(
            start,
            end,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        nn.init.kaiming_normal_(step1.weight, generator=my_local_rng)
        self.random_projection = nn.Sequential(
            step1,
        )
        self.output_dim = self.cell_reduction_dimension * 90

    def _embed_single(self, item: np.ndarray) -> torch.Tensor:
        assert (
            item.shape.prod() == 144 * 160 * 1
        ), f"Expected input shape to have 144*160*1 elements, got {item.shape}. Make sure to provide the raw pixel observation without any wrappers that change the shape."
        item = np.array(item)
        grid_cells = StateParser.capture_grid_cells(item, y_offset=0)
        # Always 90 grid cells that are 16x16 (because 160 * 144)
        cell_embeddings = []
        cell_keys = sorted(grid_cells.keys())
        for key in cell_keys:
            cell_image = grid_cells[key]
            cell_image_resized = np.resize(cell_image, (16, 16))
            cell_image_flat = cell_image_resized.flatten()
            cell_image_tensor = torch.tensor(
                cell_image_flat,
                dtype=torch.bfloat16,
                device=self.random_projection[0].weight.device,
            )
            with torch.no_grad():
                cell_embedding = self.random_projection(cell_image_tensor)
                # normalize
                cell_embedding = nn.functional.normalize(cell_embedding, dim=-1)
            cell_embeddings.append(cell_embedding)
        # cell_embeddings_tensor = torch.stack(cell_embeddings, dim=0)
        # image_embedding = torch.mean(cell_embeddings_tensor, dim=0) # shape (cell_reduction_dimension,)
        image_embedding = torch.cat(
            cell_embeddings, dim=0
        )  # shape (cell_reduction_dimension*90,)
        return image_embedding

    def embed(self, items: List[np.ndarray]) -> torch.Tensor:
        embeddings = []
        for item in items:
            embedding = self._embed_single(item)
            embeddings.append(embedding)
        return torch.stack(embeddings, dim=0)

    def train(self, **kwargs):
        pass


class CNNEmbedder(nn.Module):
    def __init__(self, hidden_dim=720):
        self.cnn = nn.Sequential(
            nn.Conv2d(
                4, 32, kernel_size=8, stride=4
            ),  # Assuming input shape (4, 84, 84) # TODO: Check the input shape. This is likely wrong
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        raw = self.cnn(x)
        normalized = nn.functional.normalize(
            raw, dim=-1
        )  # Normalize the output embeddings
        return normalized

    def embed(self, items: List[np.ndarray]) -> torch.Tensor:
        batch_tensor = torch.tensor(
            items, dtype=torch.float32, device=next(self.parameters()).device
        )  # TODO: Check this.
        embeddings = self(batch_tensor)
        return embeddings

    def train(self, **kwargs):
        raise NotImplementedError


class WorldModel(nn.Module):
    def __init__(self, embedder, hidden_dim=512, normalized_observations=True):
        super().__init__()
        self.embedder = embedder
        observation_dim = embedder.output_dim
        self.model = nn.Sequential(
            nn.Linear(observation_dim + 1, hidden_dim),  # +1 for action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim),
        )
        self.normalized_observations = normalized_observations

    def forward(self, raw_obs, action):
        with torch.no_grad():
            obs = self.embedder.embed(raw_obs)
        x = torch.cat([obs, action], dim=-1)
        next_obs_pred = self.model(x)
        if self.normalized_observations:
            next_obs_pred = nn.functional.normalize(next_obs_pred, dim=-1)
        return next_obs_pred

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        with torch.no_grad():
            next_obs_embed = self.embedder.embed(next_obs)
            predicted_next_obs_embed = self.forward(obs, actions)
        # reward is the error in the embedding space
        reward = torch.norm(predicted_next_obs_embed - next_obs_embed, dim=-1).item()
        return reward


def get_passed_frames(infos) -> np.ndarray:
    # infos['core']['passed_frames'].shape == (1, n_frames, 144, 160, 1)
    frames = infos["core"]["passed_frames"]
    return frames.squeeze(0).reshape(-1, 144, 160)


class EmbedBuffer:
    def __init__(self, embedder, max_size=10_000):
        self.max_size = max_size
        self.buffer = None
        self.embedder = embedder

    def add(self, items: np.ndarray, embeddings=None):
        if self.buffer is None:
            self.buffer = self.embedder.embed(items)
        else:
            if embeddings is not None:
                new_embedding = embeddings
            else:
                new_embedding = self.embedder.embed(items)
            self.buffer = torch.cat([self.buffer, new_embedding], dim=0)
            if self.buffer.shape[0] > self.max_size:
                self.rationalize_buffer()

    def rationalize_buffer(self):
        # cluster down to half the size and keep the cluster centers only
        target_size = self.max_size // 2
        kmeans = KMeans(n_clusters=target_size, random_state=42)
        kmeans.fit(self.buffer.cpu().numpy())
        self.buffer = torch.tensor(
            kmeans.cluster_centers_, dtype=self.buffer.dtype, device=self.buffer.device
        )

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        passed_frames = get_passed_frames(infos)
        score = self.compare(passed_frames)
        self.add(passed_frames)
        if self.buffer is None:
            self.add(passed_frames)
            return 0.0
        else:
            breakpoint()  # TODO: Check shapes etc
            item_embeddings = self.embedder.embed(passed_frames)
            # assume they are normalized, so cosine similarity is just dot product
            cosine_similarities = torch.matmul(self.buffer, item_embeddings.T).squeeze()
            max_similarity = torch.max(cosine_similarities).item()
            self.add(passed_frames, embeddings=item_embeddings)
            return 1 - max_similarity


class ClusterOnlyBuffer(Buffer):
    def __init__(self, n_clusters=100):
        self.clusters = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)

    def add(self, items: np.ndarray):
        self.clusters.partial_fit(items)

    def compare(self, items: np.ndarray) -> int:
        score = self.clusters.score(items)
        # if one cluster dominates, then we are close to a known state, otherwise we are in a novel state.
        breakpoint()  # TODO: figure out how to use the score to determine novelty.
        return score

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        passed_frames = get_passed_frames(infos)
        score = self.compare(passed_frames)
        self.add(passed_frames)
        return score


def get_curiosity_module(args):
    if args.observation_embedder == "random_patch":
        embedder = PatchProjection()
    elif args.observation_embedder == "cnn":
        embedder = CNNEmbedder()
    if "buffer" in args.curiosity_module:
        if args.curiosity_module == "embedbuffer":
            module = EmbedBuffer(embedder)
        elif args.curiosity_module == "clusterbuffer":
            module = ClusterOnlyBuffer()
    elif args.curiosity_module == "world_model":
        module = WorldModel(embedder=embedder)
    return module
