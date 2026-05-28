import os
import numpy as np
from cleanrl_utils.buffers import ReplayBuffer
from cleanrl_utils.port_gameboy_worlds.env_factory import parse_pokeworlds_id_string

from .visualization import save_outliers


class PokemonReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        # self.screens = np.zeros(
        #    (self.buffer_size, self.n_envs, 144, 160),
        #    dtype=np.uint8,
        # )
        # screens not needed because its always the last element of the observations
        self.steps = -np.ones((self.buffer_size, self.n_envs), dtype=np.uint16)
        self.step_counts = np.zeros((self.n_envs,), dtype=np.uint16)
        self.n_pos_loops = -1

    def reset(self):
        # self.screens = np.zeros(
        #    (self.buffer_size, self.n_envs, 144, 160),
        #    dtype=np.uint8,
        # )
        self.steps = -np.ones((self.buffer_size, self.n_envs), dtype=np.uint16)
        self.step_counts = np.zeros((self.n_envs,), dtype=np.uint16)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos,
    ):
        if self.pos == 0:
            self.n_pos_loops += 1
        done = "final_info" in infos
        # self.screens[self.pos, 0] = get_passed_frames(infos)[-1].reshape(144, 160)
        self.steps[self.pos, :] = self.step_counts.copy()

        self.step_counts += 1
        self.step_counts = self.step_counts * (1 - done)  # reset step count on done
        super().add(obs, next_obs, action, reward, done, infos)

    def save(self, save_folder, run_name, env_id):
        if save_folder is not None:
            (
                game,
                environment_variant,
                init_state,
                controller_variant,
                max_steps,
                save_video,
            ) = parse_pokeworlds_id_string(env_id)
            print("Saving replay buffer...")
            save_path = f"{save_folder}/{run_name}/"
            os.makedirs(save_path, exist_ok=True)
            # save an init_state.txt file with the name of the init state:
            with open(save_path + "init_state.txt", "w") as f:
                f.write(init_state)
            save_size = None
            if self.full:
                np.save(save_path + "/observations.npy", self.observations)
                np.save(save_path + "/actions.npy", self.actions)
                np.save(save_path + "/rewards.npy", self.rewards)
                # np.save(save_path + "/screens.npy", self.screens)
                np.save(save_path + "/steps.npy", self.steps)
                save_size = self.buffer_size
                save_outliers(
                    self.observations,
                    self.actions,
                    self.rewards,
                    self.steps,
                    save_path,
                    self.n_pos_loops,
                    init_state,
                )
            else:
                np.save(save_path + "/observations.npy", self.observations[: self.pos])
                np.save(save_path + "/actions.npy", self.actions[: self.pos])
                np.save(save_path + "/rewards.npy", self.rewards[: self.pos])
                # np.save(save_path + "/screens.npy", self.screens[: self.pos])
                np.save(save_path + "/steps.npy", self.steps[: self.pos])
                save_size = self.pos
                save_outliers(
                    self.observations[: self.pos],
                    self.actions[: self.pos],
                    self.rewards[: self.pos],
                    self.steps[: self.pos],
                    save_path,
                    self.n_pos_loops,
                    init_state,
                )
            print(f"Saved replay buffer with {save_size} entries to {save_path}")
