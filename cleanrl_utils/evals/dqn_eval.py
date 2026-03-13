import random
from typing import Callable

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
        model = Model(envs).to(device)
        model.load_state_dict(torch.load(full_model_path, map_location=device))
        model.eval()

        obs, _ = envs.reset()
        args.curiosity_module.reset()  # reset the curiosity module's buffer at the start of evaluation
        episodic_returns = []
        curiosity_rewards = []
        all_curiosity_rewards = []
        while len(episodic_returns) < eval_episodes:
            q_values = model(torch.Tensor(obs).to(device))
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
    return None


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.dqn import QNetwork, make_env

    model_path = hf_hub_download(
        repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="q_network.pth"
    )
    evaluate(
        model_path,
        make_env,
        "CartPole-v1",
        eval_episodes=10,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False,
    )
