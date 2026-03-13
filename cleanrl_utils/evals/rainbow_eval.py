import random
from typing import Callable
from argparse import Namespace
import gymnasium as gym
import numpy as np
import torch
import os


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    args=None,
):
    models = os.listdir(model_path)
    for model_name in models:
        print(f"evaluating model {model_name}...")
        full_model_path = os.path.join(model_path, model_name, "model.pt")
        envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, 0, 0, capture_video, run_name)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        model_data = torch.load(full_model_path, map_location="cpu")
        m_args = Namespace(**model_data["args"])
        model = Model(
            envs, n_atoms=m_args.n_atoms, v_min=m_args.v_min, v_max=m_args.v_max
        ).to(device)
        model.load_state_dict(model_data["model_weights"])
        model.eval()

        obs, _ = envs.reset()
        episodic_returns = []
        curiosity_rewards = []
        all_curiosity_rewards = []
        while len(episodic_returns) < eval_episodes:
            q_dist = model(torch.Tensor(obs).to(device))
            q_values = torch.sum(q_dist * model.support, dim=2)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            curiosity_reward = args.curiosity_module.get_reward(
                obs, actions, next_obs, infos
            )
            rewards[0] = rewards[0] + curiosity_reward
            curiosity_rewards.append(curiosity_reward)
            if "final_info" in infos:
                if isinstance(infos["final_info"], dict):
                    infos["final_info"] = [infos["final_info"]]
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    print(
                        f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}, curiosity_reward={sum(curiosity_rewards)}"
                    )
                    episodic_returns += [info["episode"]["r"]]
                all_curiosity_rewards.append(float(sum(curiosity_rewards)))
                args.curiosity_module.iterative_save()
                args.curiosity_module.reset()
                curiosity_rewards = []
            obs = next_obs
        with open(os.path.join(model_path, model_name, "eval_reward.txt"), "w") as f:
            for curiosity_reward in all_curiosity_rewards:
                f.write(f"{curiosity_reward}\n")

    return


if __name__ == "__main__":
    raise NotImplementedError(
        "Run cleanrl_utils/enjoy.py instead to evaluate trained models."
    )
