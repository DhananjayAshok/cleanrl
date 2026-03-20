import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from .utils import FRAME_STACK
from .env_factory import OneOfToDiscreteWrapper


def stacked_frame_to_single(observation):
    # observation shape is (4, 144, 160)
    # We want to convert it to a (144 x 4, 160) image where each of the 4 frames is stacked vertically. This is just for visualization purposes.
    all_obs = observation.reshape(FRAME_STACK, 144, 160)
    show_obs = np.zeros((144 * FRAME_STACK, 160), dtype=np.uint8)
    for i in range(FRAME_STACK):
        show_obs[i * 144 : (i + 1) * 144] = all_obs[i]
    return show_obs


def plot_observation(
    observation, save_name, save_folder="../frame_saves/", title="Observation Frames"
):
    save_path = f"{save_folder}/{save_name}.png"
    os.makedirs(save_folder, exist_ok=True)
    obs_single = stacked_frame_to_single(observation)
    plt.figure(figsize=(5, 20))
    plt.imshow(obs_single, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_transition(
    observation, new_observation, action, reward, global_step, step, save_path
):
    obs_single = stacked_frame_to_single(observation)
    new_obs_single = stacked_frame_to_single(new_observation)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(obs_single, cmap="gray")
    action_class, action_kwargs = OneOfToDiscreteWrapper.get_high_level_action_static(
        action.reshape(-1)[0]
    )
    action = action_kwargs
    axes[0].set_title(
        f"\nGlobal Step {global_step}\nEnvironment Step {step.reshape(-1)[0]}\nObservation\nAction:\n{action}"
    )
    axes[1].imshow(new_obs_single, cmap="gray")
    axes[1].set_title(f"New Observation\nReward: {reward.reshape(-1)[0]}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def infer_global_step(index, n_pos_loops, buffer_size):
    if n_pos_loops == -1:
        return -1
    return n_pos_loops * buffer_size + index


def save_transition_visualizations(
    observations,
    actions,
    rewards,
    steps,
    save_folder,
    n_pos_loops,
    top_sample_indices,
    bottom_sample_indices,
    n_plots=3,
):
    if n_plots <= 0:
        return
    save_path = f"{save_folder}/transition_visualizations/"
    os.makedirs(save_path, exist_ok=True)
    buffer_size = len(rewards)
    for i in range(n_plots):
        observation, new_observation, action, reward, step = (
            observations[top_sample_indices[i]],
            observations[top_sample_indices[i] + 1],
            actions[top_sample_indices[i]],
            rewards[top_sample_indices[i]],
            steps[top_sample_indices[i]],
        )
        visualize_transition(
            observation,
            new_observation,
            action,
            reward,
            infer_global_step(top_sample_indices[i], n_pos_loops, buffer_size),
            step,
            save_path + f"top_transition_{i}.png",
        )
        """
        observation, new_observation, action, reward, step = (
            observations[bottom_sample_indices[i]],
            observations[bottom_sample_indices[i] + 1],
            actions[bottom_sample_indices[i]],
            rewards[bottom_sample_indices[i]],
            steps[bottom_sample_indices[i]],
        )
        visualize_transition(
            observation,
            new_observation,
            action,
            reward,
            infer_global_step(bottom_sample_indices[i], n_pos_loops, buffer_size),
            step,
            save_path + f"bottom_transition_{i}.png",
        )
        """
    print(f"Saved transition visualizations for top {n_plots} rewards to {save_path}")


def save_outlier_trajectories(
    observations,
    actions,
    rewards,
    steps,
    load_path,
    high_reward_indices,
    max_trajectory_length=30,
):
    trajectories = []
    for high_reward_index in high_reward_indices:
        traj_observations = [None for i in range(max_trajectory_length + 1)]
        traj_actions = [None for i in range(max_trajectory_length)]
        traj_high_level_actions = [None for i in range(max_trajectory_length)]
        traj_rewards = [None for i in range(max_trajectory_length)]
        current_index = high_reward_index
        traj_observations[-1] = observations[current_index][
            0, -1
        ]  # assume single env and get last frame in the stack
        for i in range(max_trajectory_length):
            traj_observations[-2 - i] = observations[current_index - 1][0, -1]
            traj_actions[-1 - i] = actions[current_index - 1]
            traj_high_level_actions[-1 - i] = (
                OneOfToDiscreteWrapper.get_high_level_action_static(
                    actions[current_index - 1].reshape(-1)[0]
                )
            )
            traj_rewards[-1 - i] = rewards[current_index - 1]
            if steps[current_index] == 0:
                break
            current_index -= 1
        traj_observations = [
            traj_obs for traj_obs in traj_observations if traj_obs is not None
        ]
        traj_actions = [traj_act for traj_act in traj_actions if traj_act is not None]
        traj_rewards = [traj_rew for traj_rew in traj_rewards if traj_rew is not None]
        traj_high_level_actions = [
            traj_high_act
            for traj_high_act in traj_high_level_actions
            if traj_high_act is not None
        ]
        trajectories.append(
            (traj_observations, traj_actions, traj_high_level_actions, traj_rewards)
        )
    pickle.dump(
        trajectories,
        open(load_path + "high_reward_trajectories.pkl", "wb"),
    )
    print(
        f"Saved {len(trajectories)} high reward trajectories to {load_path + 'high_reward_trajectories.pkl'}"
    )


def save_outliers(
    observations,
    actions,
    rewards,
    steps,
    save_folder,
    n_pos_loops,
    frac_samples=0.05,
    outlier_threshold=2.5,
):
    print("Analyzing rewards for outliers and visualization...")
    load_path = f"{save_folder}/"
    new_episode_indices = np.where(steps == 0)[0]
    last_step_indices = new_episode_indices - 1
    # replace -1 values with the last index of the buffer for the first episode
    minus_one_indices = np.where(last_step_indices == -1)[0]
    if len(minus_one_indices) > 0:
        last_step_indices[minus_one_indices] = len(steps) - 1
    np.save(load_path + "last_step_indices.npy", last_step_indices)
    reward_mean = np.nanmean(rewards)
    reward_std = np.nanstd(rewards)
    rewards[new_episode_indices] = reward_mean
    rewards[last_step_indices] = reward_mean
    reward_mean = rewards.mean()
    reward_std = rewards.std()
    normalized_rewards = (rewards - reward_mean) / (reward_std + 1e-8)
    # identify the indices of the top and bottom n_samples rewards
    sorted_indices = np.argsort(normalized_rewards, axis=0)
    n_samples = int(len(rewards) * frac_samples)
    top_sample_indices = sorted_indices[-n_samples:]
    top_sample_indices = top_sample_indices[::-1]
    bottom_sample_indices = sorted_indices[:n_samples]
    high_reward_indices = np.where(normalized_rewards > outlier_threshold)[0]
    if len(high_reward_indices) > 0:
        np.save(load_path + "high_reward_indices.npy", high_reward_indices)
        print(
            f"Saved {len(high_reward_indices)} reward indices to {load_path + 'high_reward_indices.npy'}"
        )
        save_outlier_trajectories(
            observations,
            actions,
            rewards,
            steps,
            load_path,
            high_reward_indices,
        )
    else:
        print("No high reward outliers found.")

    save_transition_visualizations(
        observations,
        actions,
        rewards,
        steps,
        save_folder,
        n_pos_loops,
        top_sample_indices,
        bottom_sample_indices,
    )
