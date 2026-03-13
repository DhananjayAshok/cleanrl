import shutil
import os
import tyro
from tqdm import tqdm
import os
from dataclasses import dataclass


@dataclass
class Args:
    model_dir: str
    """ Path to a directory /path/to/model_dir/<exp_name>/<model_rank>/(model.pt and eval_rewards.txt) """
    best_k: int = 10
    """ Number of best models to keep. The models will be sorted by the average episodic return in the eval_reward.txt file. """
    clear_loser_replay_buffer: bool = False
    """ If True, the replay buffers of the models that are not in the best k will be deleted to save disk space. """
    clear_loser_high_reward_trajectories: bool = False
    """ If False, clear_loser_replay buffer will leave the high reward trajectories """
    replay_buffer_path: str = None
    """ Path to a directory /path/to/replay_buffer_path/<exp_name>/(replay buffer info like observations.npy). Must match the run names of model_dir exactly """
    verbose: bool = True
    """ If True, print the names of the winning models and their rewards. """


if __name__ == "__main__":
    args = tyro.cli(Args)

    if not os.path.exists(args.model_dir):
        raise ValueError(f"model_dir does not exist: {args.model_dir}")

    experiment_dirs = os.listdir(args.model_dir)
    total_models = 0
    model_dirs = {}
    for exp_name in experiment_dirs:
        for model_dir in os.listdir(os.path.join(args.model_dir, exp_name)):
            true_model_dir = os.path.join(args.model_dir, exp_name, model_dir)
            if not os.path.exists(true_model_dir + "/model.pt"):
                raise ValueError(f"Couldn't find model.pt in {true_model_dir}")
            if not os.path.exists(true_model_dir + "/eval_rewards.txt"):
                raise ValueError(f"Couldn't find eval_rewards.txt in {true_model_dir}")
            model_dirs[exp_name].append(model_dir)
            total_models += 1

    if len(total_models) < args.best_k:
        print(
            f"Warning: Number of experiment dirs in model_dir is less than best_k: {total_models} < {args.best_k}"
        )
        exit(0)

    if args.clear_loser_high_reward_trajectories and not args.clear_loser_replay_buffer:
        raise ValueError(
            "clear_loser_high_reward_trajectories requires clear_loser_replay_buffer to be True"
        )

    if args.clear_loser_replay_buffer:
        if args.replay_buffer_path is None:
            raise ValueError(
                "replay_buffer_path must be provided when clear_loser_replay_buffer is True"
            )
        rb_dirs = set(os.listdir(args.replay_buffer_path))
        for exp_name in experiment_dirs:
            if exp_name not in rb_dirs:
                raise ValueError(
                    f"replay_buffer_path is missing experiment dir: {exp_name}"
                )
            obs_path = os.path.join(
                args.replay_buffer_path, exp_name, "observations.npy"
            )
            if not os.path.isfile(obs_path):
                raise ValueError(
                    f"Missing observations.npy in {os.path.join(args.replay_buffer_path, exp_name)}"
                )

    rewards = {}
    for exp_name in tqdm(model_dirs, desc="Reading rewards"):
        for model_dir in model_dirs[exp_name]:
            true_model_dir = os.path.join(args.model_dir, exp_name, model_dir)
            rewards_path = os.path.join(true_model_dir, "eval_rewards.txt")
            with open(rewards_path, "r") as f:
                values = [float(line.strip()) for line in f if line.strip()]
            rewards[(exp_name, model_dir)] = (
                sum(values) / len(values) if values else float("-inf")
            )

    sorted_dirs = sorted(rewards, key=lambda x: rewards[x], reverse=True)
    best_dirs = set(sorted_dirs[: args.best_k])
    loser_dirs = [d for d in sorted_dirs if d not in best_dirs]

    for exp_name, model_dir in tqdm(loser_dirs, desc="Deleting losers"):
        loser_model_path = os.path.join(args.model_dir, exp_name, model_dir)
        shutil.rmtree(loser_model_path)

        if args.clear_loser_replay_buffer:
            rb_exp_path = os.path.join(args.replay_buffer_path, exp_name)
            if args.clear_loser_high_reward_trajectories:
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

    if args.verbose:
        print("Kept the following best models:")
        for winner_name in best_dirs:
            print(f"  - {winner_name} | Reward: {rewards[winner_name]}")
