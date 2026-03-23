from gameboy_worlds import get_environment
import gymnasium as gym
from gymnasium.spaces import Discrete

from .utils import FRAME_STACK


class OneOfToDiscreteWrapper(gym.ActionWrapper):
    STATIC_MAP = {}
    """ Set on init to allow static access of a dict mapping actions to HighLevelActions """

    def __init__(self, env):
        super().__init__(env)
        # Calculate total actions across all sub-spaces
        # Example: OneOf([Discrete(2), Discrete(3)]) -> total 5
        self.internal_env = env
        self.sub_spaces = env.action_space.spaces
        self.total_actions = sum(s.n for s in self.sub_spaces)
        self.action_space = Discrete(self.total_actions)
        for action in range(self.action_space.n):
            high_level_action, kwargs = self.get_high_level_action(action)
            OneOfToDiscreteWrapper.STATIC_MAP[action] = (high_level_action, kwargs)

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

    @staticmethod
    def get_high_level_action_static(action):
        if len(OneOfToDiscreteWrapper.STATIC_MAP) == 0:
            raise ValueError("STATIC_MAP not initialized yet!")
        return OneOfToDiscreteWrapper.STATIC_MAP[action]


def parse_pokeworlds_id_string(id_string):
    """

    :param id_string: should be in format "gameboy_worlds-game-environment_variant-init_state-controller_variant-max_steps-save_video"
    Example: gameboy_worlds-pokemon_red-starter_explore-none-low_level-20-true
    :return: tuple (game, environment_variant, init_state, controller_variant, max_steps, save_video)
    """
    #
    parts = id_string.split("-")
    if len(parts) != 7 or parts[0] != "gameboy_worlds":
        raise ValueError(
            f"Invalid ID string format. Expected 'gameboy_worlds-game-environment_variant-init_state-controller_variant-max_steps-save_video'. Got {id_string}"
        )
    (
        _,
        game,
        environment_variant,
        init_state,
        controller_variant,
        max_steps_str,
        save_video_str,
    ) = parts
    if not max_steps_str.isdigit():
        raise ValueError(
            f"Invalid max_steps value. Expected an integer. Got {max_steps_str}"
        )
    max_steps = int(max_steps_str)
    save_video = save_video_str.lower() == "true"
    if init_state.lower() == "none":
        init_state = None
    return (
        game,
        environment_variant,
        init_state,
        controller_variant,
        max_steps,
        save_video,
    )


def get_gameboy_worlds_environment(id_string, run_name, render_mode=None):
    game, environment_variant, init_state, controller_variant, max_steps, save_video = (
        parse_pokeworlds_id_string(id_string)
    )

    env = get_environment(
        game=game,
        controller_variant=controller_variant,
        init_state=init_state,
        environment_variant=environment_variant,
        max_steps=max_steps,
        headless=True,
        save_video=save_video,
        session_name=run_name,
    )
    env = OneOfToDiscreteWrapper(env)
    if render_mode is not None:
        env.set_render_mode(render_mode)
    return env


def get_pokeworlds_n_actions(id_string=None):
    if len(OneOfToDiscreteWrapper.STATIC_MAP) == 0:
        if id_string is not None:
            _ = get_gameboy_worlds_environment(id_string, run_name=None)
        else:
            raise ValueError(
                f"STATIC_MAP not initialized yet! Please provide an id_string to initialize the environment and action mapping."
            )
    return len(OneOfToDiscreteWrapper.STATIC_MAP)


def gameboy_worlds_make_env(env_id, seed, idx, capture_video, run_name, gamma=0.99):
    # if capture_video == 1:
    #    capture_video = True
    # if isinstance(capture_video, int):
    #    capture_every = max(1, capture_video)
    #    capture_video = True
    # else:
    #    capture_every = None
    capture_video = False  # disable, we capture video through the environment itself

    def thunk():
        if capture_video and idx == 0:
            env = get_gameboy_worlds_environment(
                env_id, run_name, render_mode="rgb_array"
            )
            if capture_every is not None:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    episode_trigger=lambda episode_id: episode_id % capture_every == 0,
                )
            else:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = get_gameboy_worlds_environment(env_id, run_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(
            env, (144, 160)
        )  # Don't ask me why, but this is needed.
        env = gym.wrappers.FrameStackObservation(env, FRAME_STACK)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)

        if seed is not None:
            env.action_space.seed(seed)
        return env

    return thunk
