import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans, KMeans

from .utils import FRAME_STACK
from .env_factory import get_pokeworlds_n_actions
from .embedders import PatchProjection, CNNEmbedder


def get_passed_frames(infos) -> np.ndarray:
    # infos['core']['passed_frames'].shape == (1, n_frames, 144, 160, 1)
    frames = infos["core"]["passed_frames"]
    if len(frames.shape) == 1:  # then a reset has happened. must use current frame
        frames = infos["core"]["current_frame"]
    return frames.squeeze(0).reshape(-1, 144, 160)


def chunk_ocr_frame(frame, n_chunks=8):
    """
    Returns n_chunk non-overlapping chunks of the input frame, where each chunk is of varying shape based on the region
    Horizontal chunking: splits frame into n_chunks equal-width vertical strips along the width axis
    frame: np.ndarray of shape (height, width)
    returns: list of n_chunks arrays each of shape (height, width//n_chunks)
    """
    _, width = frame.shape
    chunk_width = width // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_width
        end = (i + 1) * chunk_width
        chunks.append(frame[:, start:end])
    return chunks


class OCRBuffer:
    """
    Stores the frames as chunks (no embedding, just raw frames)
    OCR in the game env state info is returned as a separate frame of fixed size by key
    e.g. 'dialogue': np array of specific shape
            'menu': np array of a different shape, but constant for all menu frames
    will store a separate buffer for each of these and chunk them to facilitate chunk wise comparison
    A new buffer is created automatically when a region key is seen for the first time.
    """

    def __init__(self, n_chunks=8, save_path=None, load_path=None):
        self.n_chunks = n_chunks
        self.save_path = save_path
        self.load_path = load_path
        self.buffers = (
            {}
        )  # dict[str, torch.Tensor]  region_key -> (buffer_size, chunk_size)
        self.reset()

    def _frames_to_chunks(self, frames: np.ndarray) -> torch.Tensor:
        """Convert frames (n_frames, height, width) to flat chunk tensor (n_frames*n_chunks, chunk_size)."""
        all_chunks = []
        for frame in frames:
            for chunk in chunk_ocr_frame(frame, self.n_chunks):
                all_chunks.append(chunk.astype(np.float32) / 255.0)
        return torch.tensor(np.stack(all_chunks), dtype=torch.float32)

    def get_unseen_elements(self, chunks: torch.Tensor, buffer: torch.Tensor):
        """
        Same logic as EmbedBuffer: return only chunks not already in the buffer.
        chunks: (n, chunk_size), buffer: (buffer_size, chunk_size)
        Returns unseen chunks tensor or None if all are duplicates.
        """
        if buffer is None:
            return chunks
        diffs = chunks.unsqueeze(1) - buffer.unsqueeze(
            0
        )  # (n, buffer_size, chunk_size)
        new_chunks = []
        for i in range(chunks.shape[0]):
            max_dim_diff = (
                diffs[i].reshape(len(buffer), -1).abs().max(-1).values
            )  # (buffer_size,)
            if max_dim_diff.min().item() >= 0.001:
                new_chunks.append(chunks[i])
        if len(new_chunks) == 0:
            return None
        return torch.stack(new_chunks)

    def iterative_save(self):
        if self.save_path is None:
            return
        os.makedirs(self.save_path, exist_ok=True)
        for key, buffer in self.buffers.items():
            if not buffer:
                continue
            file_path = os.path.join(self.save_path, f"ocr_buffer_{key}.pt")
            save_size = buffer.shape[0]
            if os.path.exists(file_path):
                existing_buffer = torch.load(file_path)
                existing_unseen = self.get_unseen_elements(existing_buffer, buffer)
                if existing_unseen is not None:
                    merged_buffer = torch.cat([existing_unseen, buffer], dim=0)
                else:
                    merged_buffer = buffer
                save_size = merged_buffer.shape[0]
                torch.save(merged_buffer.cpu(), file_path)
            else:
                torch.save(buffer.cpu(), file_path)
            print(
                f"Saved OCR buffer for '{key}' with {save_size} entries to {file_path}"
            )

    def load(self):
        if self.load_path is None:
            return
        if not os.path.exists(self.load_path):
            raise ValueError(f"No OCR buffer directory found at {self.load_path}")
        for filename in os.listdir(self.load_path):
            if filename.startswith("ocr_buffer_") and filename.endswith(".pt"):
                key = filename[len("ocr_buffer_") : -len(".pt")]
                self.buffers[key] = torch.load(os.path.join(self.load_path, filename))

    def reset(self):
        self.buffers = {}
        self.load()

    def add(self, items: dict[str, torch.Tensor]):
        for key, chunks in items.items():
            if key not in self.buffers:
                self.buffers[key] = chunks
            elif self.buffers[key] is None:
                self.buffers[key] = chunks
            else:
                self.buffers[key] = torch.cat([self.buffers[key], chunks], dim=0)

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        if "ocr_regions" not in infos["ocr"]:
            return 0.0  # no ocr screens up
        regions = [
            key for key in infos["ocr"]["ocr_regions"].keys() if not key.startswith("_")
        ]

        region_scores = []
        items_to_add = {}

        for region in regions:
            ocr_frames = infos["ocr"]["ocr_regions"][region][0]
            ocr_frames = ocr_frames.reshape(
                ocr_frames.shape[0], ocr_frames.shape[1], ocr_frames.shape[2]
            )  # n_frames, height, width
            chunks = self._frames_to_chunks(ocr_frames)
            buffer = self.buffers.get(region, None)

            if buffer is None:
                score = 1.0  # first time seeing this region — fully novel
                items_to_add[region] = chunks
            else:
                # if chunk is already in buffer, score 0
                original_chunk_count = chunks.shape[0]
                chunks = self.get_unseen_elements(chunks, buffer)
                if chunks is None:
                    score = 0.0
                else:
                    score = chunks.shape[0] / original_chunk_count
                    items_to_add[region] = chunks
            region_scores.append(score)

        self.add(items_to_add)

        if len(region_scores) == 0:
            return 0.0
        return max(region_scores)


class CombinationBuffer:
    def __init__(
        self,
        observation_embedder,
        n_chunks=8,
        ocr_alpha=0.5,
        similarity_metric="cosine",
        load_path=None,
        save_path=None,
    ):
        self.observation_buffer = EmbedBuffer(
            observation_embedder, similarity_metric, load_path, save_path
        )
        self.ocr_buffer = OCRBuffer(n_chunks)
        self.ocr_alpha = ocr_alpha

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        obs_reward = self.observation_buffer.get_reward(obs, actions, next_obs, infos)
        ocr_reward = self.ocr_buffer.get_reward(obs, actions, next_obs, infos)
        # simple combination strategy: weighted average of the two rewards
        return self.ocr_alpha * ocr_reward + (1 - self.ocr_alpha) * obs_reward

    def reset(self):
        self.observation_buffer.reset()
        self.ocr_buffer.reset()

    def iterative_save(self):
        self.observation_buffer.iterative_save()
        self.ocr_buffer.iterative_save()


class EmbedBuffer:
    def __init__(
        self,
        embedder,
        similarity_metric="cosine",
        load_path=None,
        save_path=None,
        max_size=10_000,
    ):
        self.max_size = max_size
        self.embedder = embedder
        self.save_path = save_path
        self.load_path = load_path
        if self.save_path is not None and self.save_path == self.load_path:
            print(
                f"Warning: save_path and load_path are the same. This means the buffer will be overwritten on reset and grow over time. This should only be used with a random agent to accumilate base observation data."
            )
        similarity_options = ["cosine", "distance", "hinge"]
        if similarity_metric not in similarity_options:
            raise ValueError(
                f"Invalid similarity metric {similarity_metric}. Must be one of {similarity_options}"
            )
        self.similarity_metric = similarity_metric
        self.buffer = None
        self.reset()

    def get_unseen_elements(self, embeddings, buffer=None):
        if buffer is None:
            buffer = self.buffer
        if buffer is None:
            return embeddings
        # embedding shape: (n_frames, embedding_dim)
        # buffer shape: (buffer_size, embedding_dim)
        diffs = embeddings.unsqueeze(1) - buffer.unsqueeze(0)
        new_embeddings = []
        for i in range(embeddings.shape[0]):
            max_dimension_diff = (
                diffs[i].abs().max(-1).values
            )  # max absolute difference across dimensions for each buffer element
            has_element_too_close = (
                max_dimension_diff.min().item() < 0.001
            )  # if any buffer element is too close in any dimension, we consider it already in the buffer
            if not has_element_too_close:
                new_embeddings.append(embeddings[i])
        if len(new_embeddings) == 0:
            return None
        return torch.stack(new_embeddings)

    def iterative_save(self):
        if self.save_path is not None and self.buffer is not None:
            os.makedirs(self.save_path, exist_ok=True)
            save_size = self.buffer.shape[0]
            if os.path.exists(self.save_path + "/embed_buffer.pt"):
                existing_buffer = torch.load(self.save_path + "/embed_buffer.pt").to(
                    next(self.embedder.parameters()).device
                )
                existing_buffer = self.get_unseen_elements(existing_buffer)
                if existing_buffer is not None:
                    merged_buffer = torch.cat([existing_buffer, self.buffer], dim=0)
                else:
                    print(
                        f"All current buffer entries are already in the existing buffer. Not merging."
                    )
                    merged_buffer = self.buffer
                save_size = merged_buffer.shape[0]
                torch.save(merged_buffer.cpu(), self.save_path + "/embed_buffer.pt")
            else:
                torch.save(self.buffer.cpu(), self.save_path + "/embed_buffer.pt")
            print(
                f"Saved embed buffer with {save_size} entries to {self.save_path}/embed_buffer.pt"
            )

    def load(self):
        if self.load_path is not None:
            if not os.path.exists(self.load_path + "/embed_buffer.pt"):
                raise ValueError(f"No embed buffer found at {self.load_path}")
            self.buffer = torch.load(self.load_path + "/embed_buffer.pt").to(
                next(self.embedder.parameters()).device
            )

    def reset(self):
        del self.buffer
        self.buffer = None
        self.first_add = False
        self.load()

    def add(self, items: np.ndarray, embeddings=None):
        if self.buffer is None:
            self.buffer = self.embedder.embed(items)
            self.first_add = True
        else:
            self.first_add = False
            if embeddings is not None:
                new_embedding = embeddings
            else:
                new_embedding = self.embedder.embed(items)
            # check if new_embeddings is already in the buffer. and if it is, skip adding:
            new_embedding = self.get_unseen_elements(new_embedding)
            if new_embedding is None:
                return
            self.buffer = torch.cat([self.buffer, new_embedding], dim=0)
            if self.buffer.shape[0] > self.max_size:
                self.rationalize_buffer()

    def rationalize_buffer(self):
        print(
            f"Rationalizing buffer with current size {self.buffer.shape[0]} and max size {self.max_size}..."
        )
        # cluster down to half the size and keep the cluster centers only
        target_size = self.max_size // 2
        kmeans = KMeans(n_clusters=target_size, random_state=42)
        kmeans.fit(self.buffer.cpu().numpy())
        self.buffer = torch.tensor(
            kmeans.cluster_centers_,
            dtype=self.buffer.dtype,
            device=self.buffer.device,
        )

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        passed_frames = next_obs[0][-1].reshape(1, 144, 160)  # get_passed_frames(infos)
        with torch.no_grad():
            if self.buffer is None or self.first_add:
                self.add(passed_frames)
                return 0.0
            else:
                item_embeddings = self.embedder.embed(passed_frames)
                if self.similarity_metric == "cosine":
                    # assume they are normalized, so cosine similarity is just dot product
                    cosine_similarities = torch.matmul(
                        self.buffer, item_embeddings.T
                    ).T  # shape (n_frames, buffer_size)
                    # get max per frame, then average across frames
                    score = (
                        (1 - torch.max(cosine_similarities, dim=-1).values)
                        .mean()
                        .item()
                    )
                elif self.similarity_metric == "distance":
                    # compute pairwise distances and take min per frame, then average across frames
                    distances = torch.cdist(
                        item_embeddings, self.buffer
                    )  # shape (n_frames, buffer_size)
                    score = torch.min(distances, dim=-1).values.mean().item()
                elif self.similarity_metric == "hinge":
                    # essentially find the percentage of dimensions where item_embedding - self.buffer_element < margin, max over buffer elements, then average across frames
                    margin = 0.01
                    diffs = (
                        item_embeddings.unsqueeze(1) - self.buffer.unsqueeze(0)
                    ).abs()
                    hinge = (diffs < margin).float()
                    scores = hinge.mean(
                        dim=-1
                    )  # percentage of dimensions that are close
                    max_scores = torch.max(
                        scores, dim=-1
                    ).values  # max over buffer elements
                    score = (1 - max_scores).mean().item()  # average across frames
                self.add(passed_frames, embeddings=item_embeddings)
                return score


class ClusterOnlyBuffer:
    def __init__(self, embedder, load_path=None, save_path=None, n_clusters=100):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.save_path = save_path
        self.load_path = load_path
        if self.save_path is not None and self.save_path == self.load_path:
            print(
                f"Warning: save_path and load_path are the same. This means the buffer will be overwritten on reset and grow over time. This should only be used with a random agent to accumilate base observation data."
            )
        self.reset()

    def iterative_save(self):
        if self.save_path is not None:
            print(
                "ClusterOnlyBuffer does not support iterative saving. Clusters are only saved on reset to avoid excessive file I/O. Call save() method on reset instead."
            )
        return

    def load(self):
        if self.load_path is not None:
            if not os.path.exists(self.load_path + "/cluster_buffer.pkl"):
                raise ValueError(f"No cluster buffer found at {self.load_path}")
            with open(self.load_path + "/cluster_buffer.pkl", "rb") as f:
                self.clusters = pickle.load(f)
                self.has_fit = True
                self.initial_buffer = None

    def reset(self):
        self.clusters = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42)
        self.has_fit = False
        self.initial_buffer = None
        self.load()

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


class WorldModel(nn.Module):
    def __init__(
        self,
        embedder,
        hidden_dim=512,
        normalized_observations=True,
        load_path=None,
        save_path=None,
        env_id=None,
    ):
        super().__init__()
        self.embedder = embedder
        self.model = None
        self.hidden_dim = hidden_dim
        self.normalized_observations = normalized_observations
        self.save_path = save_path
        self.load_path = load_path
        if env_id is not None:
            action_dim = get_pokeworlds_n_actions(env_id)
            self.create_model(action_dim)

    def create_model(self, action_dim=None):
        observation_dim = self.embedder.output_dim * FRAME_STACK
        if action_dim is None:
            action_dim = get_pokeworlds_n_actions()  # attempt to get from static map
        self.action_dim = action_dim
        hidden_dim = self.hidden_dim
        self.model = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedder.output_dim),
        )

    def forward(self, x):
        """x should be concatenated embedding and action tensor of shape (batch_size, observation_dim + 1)"""
        next_obs_pred = self.model(x)
        if self.normalized_observations:
            next_obs_pred = nn.functional.normalize(next_obs_pred, dim=-1)
        return next_obs_pred

    def predict(self, raw_obs, action):
        if self.model is None:
            self.create_model()
        with torch.no_grad():
            obs = self.embedder.embed(raw_obs).reshape(-1)  # flatten the frame stack
            action_vector = torch.zeros(
                self.action_dim, dtype=obs.dtype, device=obs.device
            )
            action_vector[action] = 1.0  # one-hot encode the action
            x = torch.cat([obs, action_vector], dim=-1)
            output = self.forward(x)
        return output

    def get_reward(self, obs, actions, next_obs, infos) -> float:
        with torch.no_grad():
            next_obs = next_obs[
                0, -1
            ]  # get the last frame of the frame stack. THIS COMMITS TO ONLY ONE ENV
            next_obs_embed = self.embedder.embed(next_obs)[0]
            predicted_next_obs_embed = self.predict(raw_obs=obs, action=actions)
            # reward is the error in the embedding space
            if self.normalized_observations:
                # reward is 1 - cosine_similarity. Since the embeddings are normalized, cosine similarity is just the dot product.
                reward = 1 - torch.dot(predicted_next_obs_embed, next_obs_embed).item()
            else:
                # MSE between the vectors
                reward = torch.mean(
                    (predicted_next_obs_embed - next_obs_embed) ** 2
                ).item()
        return reward

    def reset(self):
        if self.load_path is not None and self.model is None:
            self.load()

    def iterative_save(self):
        pass

    def load(self):
        self.create_model(
            action_dim=get_pokeworlds_n_actions()
        )  # this is safe because it is only called after the STATIC_MAP is initialized by creating an environment, which happens in the training loop before the world model is used.
        loaded_state = torch.load(self.load_path)
        self.model.load_state_dict(loaded_state)
        print(f"Loaded world model from {self.load_path}")


def get_curiosity_module(args):
    if args.observation_embedder == "random_patch":
        embedder = PatchProjection(
            seed=args.seed, normalized_observations=args.similarity_metric == "cosine"
        ).eval()

    elif args.observation_embedder == "cnn":
        embedder = CNNEmbedder(
            seed=args.seed, normalized_observations=args.similarity_metric == "cosine"
        ).eval()
        if args.embedder_load_path is not None:
            embedder.load(args.embedder_load_path)
    if "buffer" in args.curiosity_module:
        if args.curiosity_module == "embedbuffer":
            module = EmbedBuffer(
                embedder,
                similarity_metric=args.similarity_metric,
                save_path=args.buffer_save_path,
                load_path=args.buffer_load_path,
            )
        elif args.curiosity_module == "clusterbuffer":
            module = ClusterOnlyBuffer(
                embedder=embedder,
                save_path=args.buffer_save_path,
                load_path=args.buffer_load_path,
            )
        elif args.curiosity_module == "combinationbuffer":
            module = CombinationBuffer(
                observation_embedder=embedder,
                similarity_metric=args.similarity_metric,
                save_path=args.buffer_save_path,
                load_path=args.buffer_load_path,
            )
        else:
            raise ValueError(f"Invalid curiosity module type: {args.curiosity_module}")
    elif args.curiosity_module == "world_model":
        module = WorldModel(embedder=embedder)
    else:
        raise ValueError(f"Invalid curiosity module type: {args.curiosity_module}")
    return module
