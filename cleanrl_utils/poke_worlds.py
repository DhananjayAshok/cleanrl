from poke_worlds import get_environment
import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np


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

    :param id_string: should be in format "poke_worlds:game:environment_variant:controller_variant:max_steps"
    Example: poke_worlds:pokemon_red:starter_explore:low_level:20
    :return: tuple (game, environment_variant, controller_variant, max_steps)
    """
    #
    parts = id_string.split(":")
    if len(parts) != 5 or parts[0] != "poke_worlds":
        raise ValueError(
            f"Invalid ID string format. Expected 'poke_worlds:game:environment_variant:controller_variant:max_steps'. Got {id_string}"
        )
    _, game, environment_variant, controller_variant, max_steps_str = parts
    if not max_steps_str.isdigit():
        raise ValueError(
            f"Invalid max_steps value. Expected an integer. Got {max_steps_str}"
        )
    max_steps = int(max_steps_str)
    return game, environment_variant, controller_variant, max_steps


def get_poke_worlds_environment(id_string, render_mode=None):
    game, environment_variant, controller_variant, max_steps = (
        parse_pokeworlds_id_string(id_string)
    )
    env = get_environment(
        game=game,
        controller_variant=controller_variant,
        environment_variant=environment_variant,
        max_steps=max_steps,
        headless=True,
        save_video=False,
    )
    env = OneOfToDiscreteWrapper(env)
    if render_mode is not None:
        env.set_render_mode(render_mode)
    return env


def poke_worlds_make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = get_poke_worlds_environment(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = get_poke_worlds_environment(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)

        if seed is not None:
            env.action_space.seed(seed)
        return env

    return thunk
