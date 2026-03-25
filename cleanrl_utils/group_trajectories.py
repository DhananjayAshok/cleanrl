import os
import numpy as np
import torch
import pickle
from cleanrl_utils.port_gameboy_worlds import (
    PatchProjection,
    CNNEmbedder,
)
from tqdm import tqdm
from dataclasses import dataclass
import tyro

threshold = (
    0.01  # threshold for considering two frames as similar in the embedding space
)


@dataclass
class Args:
    # Curiosity module specific arguments
    observation_embedder: str = "random_patch"
    """the type of observation embedder to use for the curiosity module."""
    embedder_load_path: str | None = None
    """path to load the observation embedder's weights from. Only applicable if the observation embedder supports loading."""
    skip_embedding: bool = False
    """if true, will skip the embedding step and directly compare raw observations for similarity. """
    replay_buffer_folder: str | None = None
    """ Will get all high reward trajectories inside this folder and subfolders """
    save_path: str | None = None
    """path to save the grouped high reward trajectories to."""
    seed: int = 0
    """ """


class EmptyEmbedder:
    def embed(self, x):
        return x


def is_equal(frame1, frame2):
    return (frame1 - frame2).sum() <= threshold // 2


def get_embedder(args):
    if args.observation_embedder == "empty":
        observation_embedder = EmptyEmbedder()
    elif args.observation_embedder == "random_patch":
        observation_embedder = PatchProjection(
            seed=args.seed, normalized_observations=True
        ).to("cuda")
    elif args.observation_embedder == "cnn":
        observation_embedder = CNNEmbedder(
            seed=args.seed, normalized_observations=True
        ).to("cuda")
        if args.embedder_load_path is not None:
            observation_embedder.load(args.embedder_load_path)
    else:
        raise ValueError(
            f"Invalid observation embedder type: {args.observation_embedder}"
        )
    return observation_embedder


def is_similar_enough(embedding1, embedding2):
    # normalize then check cosine similarity
    embedding1 = embedding1 / (embedding1.norm() + 1e-8)
    embedding2 = embedding2 / (embedding2.norm() + 1e-8)
    cosine_similarity = torch.dot(embedding1, embedding2)
    return cosine_similarity >= 1 - threshold


def frames_similar(embedder, frame1, frame2):
    if not isinstance(embedder, EmptyEmbedder):
        frame1 = torch.Tensor(frame1).to("cuda")
        frame2 = torch.Tensor(frame2).to("cuda")
    embedding1 = embedder.embed(frame1)
    embedding2 = embedder.embed(frame2)
    return is_similar_enough(embedding1, embedding2)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.skip_embedding:
        args.observation_embedder = "empty"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    embedder = get_embedder(args)
    assert (
        args.save_path is not None
    ), "Must provide a save path to save the grouped high reward trajectories."

    high_reward_trajectories = []
    for root, dirs, files in os.walk(args.replay_buffer_folder):
        if "high_reward_trajectories.pkl" in files:
            with open(os.path.join(root, "high_reward_trajectories.pkl"), "rb") as f:
                high_reward_trajectories.extend(pickle.load(f))
    if len(high_reward_trajectories) == 0:
        raise ValueError(
            f"No high reward trajectories found in {args.replay_buffer_folder}. Please make sure to run the save_outliers function first to save high reward trajectories."
        )

    # group high reward trajectories by similarity of their final frames
    groups = []
    # each group is a dict {indexes: list of trajectory indices in self.high_reward_trajectories, final_frames: list of final frames for those trajectories}
    for i, (traj_observations, traj_actions, traj_rewards) in tqdm(
        enumerate(high_reward_trajectories), desc="Grouping high reward trajectories"
    ):
        final_frame = traj_observations[-1]
        found_group = False
        for group in groups:
            all_final_frames = group["final_frames"]
            for group_final_frame in all_final_frames:
                if frames_similar(embedder, final_frame, group_final_frame):
                    group["indexes"].append(i)
                    if not is_equal(final_frame, group_final_frame):
                        group["final_frames"].append(final_frame)
                    found_group = True
                    break
            if found_group:
                break
        if not found_group:
            groups.append(
                {
                    "indexes": [i],
                    "final_frames": [final_frame],
                }
            )
    print(f"Created {len(groups)} groups of high reward trajectories.")

    final_groups = []
    for i, group in enumerate(groups):
        group_indexes = group["indexes"]
        all_trajectories = []
        for index in group_indexes:
            traj_observations, traj_actions, traj_rewards = high_reward_trajectories[
                index
            ]
            all_trajectories.append((traj_observations, traj_actions, traj_rewards))
        final_groups.append(all_trajectories)
    os.makedirs(args.save_path, exist_ok=True)
    with open(
        os.path.join(args.save_path, "grouped_high_reward_trajectories.pkl"), "wb"
    ) as f:
        pickle.dump(final_groups, f)
    print(
        f"Saved grouped high reward trajectories to {args.save_path}/grouped_high_reward_trajectories.pkl"
    )
