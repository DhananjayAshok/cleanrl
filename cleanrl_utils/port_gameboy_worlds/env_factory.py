import json
import os

from gameboy_worlds import get_environment
import gymnasium as gym
from gymnasium.spaces import Discrete

from .utils import FRAME_STACK

ACTION_SPACE_FILENAME = "action_space.json"


class OneOfToDiscreteWrapper(gym.ActionWrapper):
    STATIC_MAP = {}
    """ Set on init to allow static access of a dict mapping actions to HighLevelActions """

    STATIC_SPEC = None
    """ Set on init: the serializable description of the discrete action space. """

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
        OneOfToDiscreteWrapper.STATIC_SPEC = self.describe_action_space()

    def describe_action_space(self):
        entries = []
        for action in range(self.action_space.n):
            sub_index, sub_action = self.action(action)
            action_class, _ = OneOfToDiscreteWrapper.STATIC_MAP[action]
            entries.append(
                {
                    "index": action,
                    "action_class": action_class.__name__,
                    "sub_index": int(sub_index),
                    "sub_action": int(sub_action),
                }
            )
        return {
            "n_actions": int(self.action_space.n),
            "sub_space_sizes": [int(s.n) for s in self.sub_spaces],
            "actions": entries,
        }

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


def get_action_space_spec(id_string=None):
    if OneOfToDiscreteWrapper.STATIC_SPEC is None:
        if id_string is None:
            raise ValueError(
                "STATIC_SPEC not initialized yet! Please provide an id_string to initialize the environment and action mapping."
            )
        _ = get_gameboy_worlds_environment(id_string, run_name=None)
    spec = dict(OneOfToDiscreteWrapper.STATIC_SPEC)
    if id_string is not None:
        game, environment_variant, _, controller_variant, _, _ = (
            parse_pokeworlds_id_string(id_string)
        )
        spec["game"] = game
        spec["environment_variant"] = environment_variant
        spec["controller_variant"] = controller_variant
    return spec


def save_action_space(save_path, id_string=None):
    spec = get_action_space_spec(id_string)
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, ACTION_SPACE_FILENAME)
    with open(file_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Saved action space with {spec['n_actions']} actions to {file_path}")
    return spec


def load_action_space(load_path):
    file_path = os.path.join(load_path, ACTION_SPACE_FILENAME)
    if not os.path.exists(file_path):
        raise ValueError(f"No action space found at {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def verify_action_space(spec, id_string=None):
    live = get_action_space_spec(id_string)
    for key in ("game", "controller_variant"):
        if key in spec and key in live and spec[key] != live[key]:
            raise ValueError(
                f"Action space mismatch on {key}: saved model was trained on {spec[key]!r} "
                f"but the live environment is {live[key]!r}."
            )
    saved = [(e["index"], e["action_class"], e["sub_index"], e["sub_action"]) for e in spec["actions"]]
    current = [(e["index"], e["action_class"], e["sub_index"], e["sub_action"]) for e in live["actions"]]
    if saved != current:
        raise ValueError(
            "Action space mismatch: the saved action indices no longer describe the live "
            f"environment's action space. Saved {len(saved)} actions, live has {len(current)}. "
            "A world model trained on the saved indices cannot be used with this environment."
        )
    return live


def resolve_action_class(spec_entry, env):
    name = spec_entry["action_class"]
    for action_class in env.get_action_strings(return_all=True):
        if action_class.__name__ == name:
            return action_class
    raise ValueError(
        f"Action class {name!r} from the saved action space is not offered by this environment."
    )


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
