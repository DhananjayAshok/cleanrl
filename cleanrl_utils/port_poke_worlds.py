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

    :param id_string: should be in format "poke_worlds-game-environment_variant-init_state-controller_variant-max_steps-save_video"
    Example: poke_worlds-pokemon_red-starter_explore-starter-low_level-20-true
    :return: tuple (game, environment_variant, init_state, controller_variant, max_steps, save_video)
    """
    #
    parts = id_string.split("-")
    if len(parts) != 7 or parts[0] != "poke_worlds":
        raise ValueError(
            f"Invalid ID string format. Expected 'poke_worlds-game-environment_variant-init_state-controller_variant-max_steps-save_video'. Got {id_string}"
        )
    (
        _,
        game,
        environment_variant,
        init_state,
        controller_variant,
        max_steps_str,
        save_video_str,
    ) = parts
    if not max_steps_str.isdigit():
        raise ValueError(
            f"Invalid max_steps value. Expected an integer. Got {max_steps_str}"
        )
    max_steps = int(max_steps_str)
    save_video = save_video_str.lower() == "true"
    return (
        game,
        environment_variant,
        init_state,
        controller_variant,
        max_steps,
        save_video,
    )


def get_poke_worlds_environment(id_string, render_mode=None):
    game, environment_variant, init_state, controller_variant, max_steps, save_video = (
        parse_pokeworlds_id_string(id_string)
    )
    env = get_environment(
        game=game,
        controller_variant=controller_variant,
        init_state=init_state,
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


class PatchProjection(nn.Module):
    """
    Works with the 144 x 160 pixel observations from poke_worlds.
    Divides the image into 16x16 patches, applies a random linear projection to each patch, and concatenates the results.
    """

    cell_reduction_dimension = 8  # hidden dimension is cell_reduction_dimension * 90

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(
                1,
                1,
                kernel_size=16,
                stride=16,  # 16x16 patches with no overlap to get each of the gameboys 16x16 cells.
            ),  # (batch_size, 32, 9, 10)
            nn.Flatten(),
        )
        self.dtype = self.project[0].weight.dtype
        self.device = self.project[0].weight.device

    def forward(self, x):
        vector = self.project(x)
        normalized = nn.functional.normalize(vector, dim=-1)
        return normalized

    def embed(self, items: List[np.ndarray]) -> torch.Tensor:
        if not isinstance(items, torch.Tensor):
            batch_tensor = torch.tensor(
                items.reshape(-1, 1, 144, 160),
            )
        batch_tensor = batch_tensor.to(self.dtype).to(self.device)
        embeddings = self(batch_tensor)
        return embeddings

    def train(self, **kwargs):
        pass


class CNNEmbedder(nn.Module):
    def __init__(self, hidden_dim=720):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=16,
                stride=16,  # 16x16 patches with no overlap to get each of the gameboys 16x16 cells.
            ),  # (batch_size, 32, 9, 10)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=2),  # (batch_size, 64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),  # (batch_size, 64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
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
            items.reshape(-1, 1, 144, 160),
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
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

    def reset(self):
        pass  # I don't think world model needs to reset anything, but we include this method for API consistency with the other curiosity modules.


def get_passed_frames(infos) -> np.ndarray:
    # infos['core']['passed_frames'].shape == (1, n_frames, 144, 160, 1)
    frames = infos["core"]["passed_frames"]
    return frames.squeeze(0).reshape(-1, 144, 160)


class EmbedBuffer:
    def __init__(self, embedder, max_size=10_000):
        self.max_size = max_size
        self.embedder = embedder
        self.buffer = None
        self.reset()

    def reset(self):
        del self.buffer
        self.buffer = None

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
        with torch.no_grad():
            if self.buffer is None:
                self.add(passed_frames)
                return 0.0
            else:
                item_embeddings = self.embedder.embed(passed_frames)
                # assume they are normalized, so cosine similarity is just dot product
                cosine_similarities = torch.matmul(
                    self.buffer, item_embeddings.T
                ).T  # shape (n_frames, buffer_size)
                # get max per frame, then average across frames
                score = (
                    (1 - torch.max(cosine_similarities, dim=-1).values).mean().item()
                )
                self.add(passed_frames, embeddings=item_embeddings)
                return score


class ClusterOnlyBuffer:
    def __init__(self, embedder, n_clusters=100):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.reset()

    def reset(self):
        self.clusters = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42)
        self.has_fit = False
        self.initial_buffer = None

    def add(self, items: np.ndarray):
        if self.has_fit:
            self.clusters.partial_fit(items)
        else:
            if self.initial_buffer is None:
                self.initial_buffer = items
            else:
                self.initial_buffer = np.concatenate(
                    [self.initial_buffer, items], axis=0
                )
                if len(self.initial_buffer) >= self.clusters.n_clusters:
                    self.clusters.fit(self.initial_buffer)
                    self.has_fit = True
                    self.initial_buffer = None

    def compare(self, items: np.ndarray) -> int:
        score = self.clusters.score(items)
        return -score

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        with torch.no_grad():
            passed_frames = get_passed_frames(infos)
            embedding = self.embedder.embed(passed_frames).cpu().numpy()
            if self.has_fit:
                score = self.compare(embedding)
            else:
                score = 0.0
            self.add(embedding)
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
            module = ClusterOnlyBuffer(embedder=embedder)
    elif args.curiosity_module == "world_model":
        module = WorldModel(embedder=embedder)
    return module
