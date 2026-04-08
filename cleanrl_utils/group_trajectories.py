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
    0.05  # threshold for considering two frames as similar in the embedding space
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
    z_min: float = 3.0
    """ the minimum z value for a trajectory to be considered an outlier and saved. """
    z_kind: str = "global"
    """ the kind of z value to use for filtering trajectories. Can be "global" or "local". Global z values are computed across all trajectories in a replay buffer, while local z values are computed within each episode. """
    protected_lookback: int = 3
    """ the number of frames at the end of the trajectory to protect from cycle snipping. """


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
    cosine_similarity = torch.dot(embedding1[0], embedding2[0])
    return cosine_similarity >= 1 - threshold


def frames_similar(embedder, frame1, frame2):
    if not isinstance(embedder, EmptyEmbedder):
        frame1 = torch.Tensor(frame1).to("cuda")
        frame2 = torch.Tensor(frame2).to("cuda")
    embedding1 = embedder.embed(frame1)
    embedding2 = embedder.embed(frame2)
    return is_similar_enough(embedding1, embedding2)


def get_first_cycle(unprotected_observation_frames):
    """Check if there are any non consecutive repeated frames in the unprotected portion of the trajectory."""
    seen_frames = []
    for i, frame in enumerate(unprotected_observation_frames):
        for seen_i, seen_frame in seen_frames:
            if np.array_equal(frame, seen_frame) and abs(i - seen_i) > 1:
                return (seen_i, i)
        seen_frames.append((i, frame))
    return None


def snip_cycles(trajectory, protected_lookback=3):
    """
    Start at the first obs frame until the protected_lookback from the end, and look for cycles of repeated frames. If a cycle is found, snip out the frames from the first occurrence of the cycle until the last occurrence of the cycle, and concatenate the remaining frames together.
    """
    assert protected_lookback > 1, "protected_lookback must be greater than 1"
    observations, actions, high_level_actions, rewards = trajectory
    n = len(observations)
    if n <= protected_lookback:
        return trajectory  # not enough frames to snip

    protected = n - protected_lookback
    protected_observation_frames = observations[protected:]
    protected_actions = actions[protected:]
    protected_high_level_actions = high_level_actions[protected:]
    protected_rewards = rewards[protected:]
    working_observation_frames = observations[:protected]
    working_actions = actions[:protected]
    working_high_level_actions = high_level_actions[:protected]
    working_rewards = rewards[:protected]

    cycle = get_first_cycle(working_observation_frames)
    if cycle is None:
        return trajectory  # no cycles found, return original trajectory
    while cycle is not None:
        start, end = cycle
        # Snip out the cycle frames from start to end (inclusive)
        working_observation_frames = np.concatenate(
            (working_observation_frames[:start], working_observation_frames[end + 1 :]),
            axis=0,
        )
        working_actions = np.concatenate(
            (working_actions[:start], working_actions[end + 1 :]), axis=0
        )
        working_high_level_actions = np.concatenate(
            (working_high_level_actions[:start], working_high_level_actions[end + 1 :]),
            axis=0,
        )
        working_rewards = np.concatenate(
            (working_rewards[:start], working_rewards[end + 1 :]), axis=0
        )
        cycle = get_first_cycle(working_observation_frames)
    observations = np.concatenate(
        (working_observation_frames, protected_observation_frames), axis=0
    )
    actions = np.concatenate((working_actions, protected_actions), axis=0)
    high_level_actions = np.concatenate(
        (working_high_level_actions, protected_high_level_actions), axis=0
    )
    rewards = np.concatenate((working_rewards, protected_rewards), axis=0)
    return trajectory


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.z_kind in ["global", "local"], "z_kind must be either global or local"
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
        if f"{args.z_kind}_high_reward_trajectories.pkl" in files:
            with open(
                os.path.join(root, f"{args.z_kind}_high_reward_trajectories.pkl"), "rb"
            ) as f:
                high_reward_trajes = pickle.load(f)
            z_values = np.load(
                os.path.join(root, f"{args.z_kind}_high_reward_z_values.npy")
            )
            selected_trajectories = []
            if args.z_min is None:
                selected_trajectories = high_reward_trajes
            else:
                for i in range(len(z_values)):
                    if z_values[i] >= args.z_min:
                        selected_trajectories.append(high_reward_trajes[i])
            if len(selected_trajectories) == 0:
                print(
                    f"Found no high reward trajectories in {root} with z values above {args.z_min}."
                )
            else:
                high_reward_trajectories.extend(selected_trajectories)
    if len(high_reward_trajectories) == 0:
        raise ValueError(
            f"No high reward trajectories found in {args.replay_buffer_folder}. Please make sure to run the save_outliers function first to save high reward trajectories."
        )

    # group high reward trajectories by similarity of their final frames
    groups = []
    # each group is a dict {indexes: list of trajectory indices in self.high_reward_trajectories, final_frames: list of final frames for those trajectories}
    # breakpoint()
    for i, (
        traj_observations,
        traj_actions,
        traj_high_level_actions,
        traj_rewards,
    ) in tqdm(
        enumerate(high_reward_trajectories),
        desc="Grouping high reward trajectories",
        total=len(high_reward_trajectories),
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
            traj_observations, traj_actions, traj_high_level_actions, traj_rewards = (
                high_reward_trajectories[index]
            )
            trajectory = (
                traj_observations,
                traj_actions,
                traj_high_level_actions,
                traj_rewards,
            )
            trajectory = snip_cycles(
                trajectory, protected_lookback=args.protected_lookback
            )
            all_trajectories.append(trajectory)
        final_groups.append(all_trajectories)
    os.makedirs(args.save_path, exist_ok=True)
    with open(
        os.path.join(
            args.save_path, f"grouped_{args.z_kind}_high_reward_trajectories.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(final_groups, f)
    print(
        f"Saved grouped {args.z_kind} high reward trajectories to {args.save_path}/grouped_{args.z_kind}_high_reward_trajectories.pkl"
    )
