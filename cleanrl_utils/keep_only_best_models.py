import shutil
import os
import tyro
from tqdm import tqdm
import os
from dataclasses import dataclass


@dataclass
class Args:
    model_dir: str
    """ Path to a directory /path/to/model_dir/<exp_name>/<model_rank>/(model.pt and eval_reward.txt) """
    video_session_dir: str
    """ Path to a directory /path/to/video_session_dir/<exp_name>/<instance_id>/(video files). Every run name in the model_dir must be in this directory """
    best_k: int = 10
    """ Number of best models to keep. The models will be sorted by the average episodic return in the eval_reward.txt file. """
    clear_loser_replay_buffer: bool = True
    """ If True, the replay buffers of the models that are not in the best k will be deleted to save disk space. """
    clear_loser_high_reward_trajectories: bool = True
    """ If False, do not clear the high reward trajectories when deleting buffer """
    clear_winner_replay_buffer: bool = True
    """ If True, the replay buffers of all models will be deleted to save disk space. Overrides clear_loser_replay_buffer. """
    replay_buffer_save_folder: str = None
    """ Path to a directory /path/to/replay_buffer_save_folder/<exp_name>/(replay buffer info like observations.npy). Must match the run names of model_dir exactly """
    verbose: bool = True
    """ If True, print the names of the winning models and their rewards. """


def clear_replay_buffer(rb_exp_path, clear_high_reward_trajectories):
    # rb_exp_path may already have been deleted, in which case skip
    if not os.path.exists(rb_exp_path):
        return
    if clear_high_reward_trajectories:
        shutil.rmtree(rb_exp_path)
    else:
        for fname in [
            "observations.npy",
            "actions.npy",
            "rewards.npy",
            "steps.npy",
        ]:
            fpath = os.path.join(rb_exp_path, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)


def experiment_has_video(video_dirs, exp_name):
    for video_dir in video_dirs:
        if exp_name in video_dir:
            return True
    return False


if __name__ == "__main__":
    args = tyro.cli(Args)

    if not os.path.exists(args.model_dir):
        raise ValueError(f"model_dir does not exist: {args.model_dir}")

    experiment_dirs = os.listdir(args.model_dir)
    total_models = 0
    model_dirs = {}
    video_dirs = os.listdir(args.video_session_dir)
    for exp_name in experiment_dirs:
        if not experiment_has_video(video_dirs, exp_name):
            raise ValueError(
                f"video_session_dir {args.video_session_dir} is missing experiment dir: {exp_name} (has: {os.listdir(args.video_session_dir)}). Every run name in the model_dir must be in this directory"
            )
        # Delete the exp_video directory for all models, only the winning models will regenerate their video directories after the losers are deleted.
        exp_video_dir = os.path.join(args.video_session_dir, exp_name)
        shutil.rmtree(exp_video_dir)
        model_dirs[exp_name] = []
        for model_dir in os.listdir(os.path.join(args.model_dir, exp_name)):
            true_model_dir = os.path.join(args.model_dir, exp_name, model_dir)
            if not os.path.exists(true_model_dir + "/model.pt"):
                raise ValueError(f"Couldn't find model.pt in {true_model_dir}")
            if not os.path.exists(true_model_dir + "/eval_reward.txt"):
                raise ValueError(f"Couldn't find eval_reward.txt in {true_model_dir}")
            model_dirs[exp_name].append(model_dir)
            total_models += 1

    if total_models < args.best_k:
        print(
            f"Warning: Number of experiment dirs in model_dir is less than best_k: {total_models} < {args.best_k}"
        )
        exit(0)

    if args.clear_loser_high_reward_trajectories and not args.clear_loser_replay_buffer:
        raise ValueError(
            "clear_loser_high_reward_trajectories requires clear_loser_replay_buffer to be True"
        )

    if args.clear_loser_replay_buffer or args.clear_winner_replay_buffer:
        if args.replay_buffer_save_folder is None:
            raise ValueError(
                "replay_buffer_save_folder must be provided when clear_loser_replay_buffer is True or clear_winner_replay_buffer is True"
            )
        rb_dirs = set(os.listdir(args.replay_buffer_save_folder))
        for exp_name in experiment_dirs:
            if exp_name not in rb_dirs:
                raise ValueError(
                    f"replay_buffer_path {args.replay_buffer_save_folder} is missing experiment dir: {exp_name} (has: {os.listdir(args.replay_buffer_save_folder)})"
                )
            obs_path = os.path.join(
                args.replay_buffer_save_folder, exp_name, "observations.npy"
            )
            if not os.path.isfile(obs_path):
                raise ValueError(
                    f"Missing observations.npy in {os.path.join(args.replay_buffer_save_folder, exp_name)} (has: {os.listdir(os.path.join(args.replay_buffer_save_folder, exp_name))})"
                )

    rewards = {}
    for exp_name in tqdm(model_dirs, desc="Reading rewards"):
        for model_dir in model_dirs[exp_name]:
            true_model_dir = os.path.join(args.model_dir, exp_name, model_dir)
            rewards_path = os.path.join(true_model_dir, "eval_reward.txt")
            with open(rewards_path, "r") as f:
                values = [float(line.strip()) for line in f if line.strip()]
            rewards[(exp_name, model_dir)] = (
                sum(values) / len(values) if values else float("-inf")
            )

    sorted_dirs = sorted(rewards, key=lambda x: rewards[x], reverse=True)
    best_dirs = set(sorted_dirs[: args.best_k])
    loser_dirs = [d for d in sorted_dirs if d not in best_dirs]
    if (
        args.clear_winner_replay_buffer
    ):  # just clear all replay buffers, no need to check if it's a winner or loser
        for exp_name, model_dir in tqdm(
            sorted_dirs, desc="Clearing all replay buffers"
        ):
            rb_exp_path = os.path.join(args.replay_buffer_save_folder, exp_name)
            clear_replay_buffer(rb_exp_path, False)

    winner_exp_names = {winner_exp for winner_exp, _ in best_dirs}
    for exp_name, model_dir in tqdm(loser_dirs, desc="Deleting losers"):
        loser_model_path = os.path.join(args.model_dir, exp_name, model_dir)
        shutil.rmtree(loser_model_path)
        rb_exp_path = os.path.join(args.replay_buffer_save_folder, exp_name)
        is_actually_loser = (
            exp_name not in winner_exp_names
        )  # might have a different checkpoint that won.
        if args.clear_loser_replay_buffer and is_actually_loser:
            clear_replay_buffer(rb_exp_path, args.clear_loser_high_reward_trajectories)

    if args.verbose:
        print("Kept the following best models:")
        for winner_name in best_dirs:
            print(f"  - {winner_name} | Reward: {rewards[winner_name]}")
