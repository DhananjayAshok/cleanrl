def dqn():
    import cleanrl.dqn
    import cleanrl_utils.evals.dqn_eval

    return (
        cleanrl.dqn.QNetwork,
        cleanrl.dqn.make_env,
        cleanrl_utils.evals.dqn_eval.evaluate,
    )


def dqn_atari():
    import cleanrl.dqn_atari
    import cleanrl_utils.evals.dqn_eval

    return (
        cleanrl.dqn_atari.QNetwork,
        cleanrl.dqn_atari.make_env,
        cleanrl_utils.evals.dqn_eval.evaluate,
    )


def dqn_jax():
    import cleanrl.dqn_jax
    import cleanrl_utils.evals.dqn_jax_eval

    return (
        cleanrl.dqn_jax.QNetwork,
        cleanrl.dqn_jax.make_env,
        cleanrl_utils.evals.dqn_jax_eval.evaluate,
    )


def dqn_atari_jax():
    import cleanrl.dqn_atari_jax
    import cleanrl_utils.evals.dqn_jax_eval

    return (
        cleanrl.dqn_atari_jax.QNetwork,
        cleanrl.dqn_atari_jax.make_env,
        cleanrl_utils.evals.dqn_jax_eval.evaluate,
    )


def c51():
    import cleanrl.c51
    import cleanrl_utils.evals.c51_eval

    return (
        cleanrl.c51.QNetwork,
        cleanrl.c51.make_env,
        cleanrl_utils.evals.c51_eval.evaluate,
    )


def c51_atari():
    import cleanrl.c51_atari
    import cleanrl_utils.evals.c51_eval

    return (
        cleanrl.c51_atari.QNetwork,
        cleanrl.c51_atari.make_env,
        cleanrl_utils.evals.c51_eval.evaluate,
    )


def c51_jax():
    import cleanrl.c51_jax
    import cleanrl_utils.evals.c51_jax_eval

    return (
        cleanrl.c51_jax.QNetwork,
        cleanrl.c51_jax.make_env,
        cleanrl_utils.evals.c51_jax_eval.evaluate,
    )


def c51_atari_jax():
    import cleanrl.c51_atari_jax
    import cleanrl_utils.evals.c51_jax_eval

    return (
        cleanrl.c51_atari_jax.QNetwork,
        cleanrl.c51_atari_jax.make_env,
        cleanrl_utils.evals.c51_jax_eval.evaluate,
    )


def ppo_atari():
    import cleanrl.ppo_atari
    import cleanrl_utils.evals.ppo_eval

    return (
        cleanrl.ppo_atari.Agent,
        cleanrl.ppo_atari.make_env,
        cleanrl_utils.evals.ppo_eval.evaluate,
    )


def ppo_atari_envpool_xla_jax_scan():
    import cleanrl.ppo_atari_envpool_xla_jax_scan
    import cleanrl_utils.evals.ppo_envpool_jax_eval

    return (
        (
            cleanrl.ppo_atari_envpool_xla_jax_scan.Network,
            cleanrl.ppo_atari_envpool_xla_jax_scan.Actor,
            cleanrl.ppo_atari_envpool_xla_jax_scan.Critic,
        ),
        cleanrl.ppo_atari_envpool_xla_jax_scan.make_env,
        cleanrl_utils.evals.ppo_envpool_jax_eval.evaluate,
    )


def sac_atari():
    import cleanrl.sac_atari
    import cleanrl_utils.evals.sac_eval

    return (
        cleanrl.sac_atari.Actor,
        cleanrl.sac_atari.make_env,
        cleanrl_utils.evals.sac_eval.evaluate,
    )


def rainbow_atari():
    import cleanrl.rainbow_atari
    import cleanrl_utils.evals.rainbow_eval

    return (
        cleanrl.rainbow_atari.NoisyDuelingDistributionalNetwork,
        cleanrl.rainbow_atari.make_env,
        cleanrl_utils.evals.rainbow_eval.evaluate,
    )


def ppo_curiosity():
    import cleanrl.ppo_curiosity
    import cleanrl_utils.evals.ppo_eval

    return (
        cleanrl.ppo_curiosity.Agent,
        cleanrl.ppo_curiosity.make_env,
        cleanrl_utils.evals.ppo_eval.evaluate,
    )


def dqn_curiosity():
    import cleanrl.dqn_curiosity
    import cleanrl_utils.evals.dqn_eval

    return (
        cleanrl.dqn_curiosity.QNetwork,
        cleanrl.dqn_curiosity.make_env,
        cleanrl_utils.evals.dqn_eval.evaluate,
    )


def c51_curiosity():
    import cleanrl.c51_curiosity
    import cleanrl_utils.evals.c51_eval

    return (
        cleanrl.c51_curiosity.QNetwork,
        cleanrl.c51_curiosity.make_env,
        cleanrl_utils.evals.c51_eval.evaluate,
    )


def rainbow_curiosity():
    import cleanrl.rainbow_curiosity
    import cleanrl_utils.evals.rainbow_eval

    return (
        cleanrl.rainbow_curiosity.NoisyDuelingDistributionalNetwork,
        cleanrl.rainbow_curiosity.make_env,
        cleanrl_utils.evals.rainbow_eval.evaluate,
    )


def sac_curiosity():
    import cleanrl.sac_curiosity
    import cleanrl_utils.evals.sac_eval

    return (
        cleanrl.sac_curiosity.Actor,
        cleanrl.sac_curiosity.make_env,
        cleanrl_utils.evals.sac_eval.evaluate,
    )


def random_curiosity():
    import cleanrl.random_curiosity
    import cleanrl_utils.evals.random_eval

    return (
        None,
        cleanrl.random_curiosity.make_env,
        cleanrl_utils.evals.random_eval.evaluate,
    )


def human_curiosity():
    import cleanrl.human_curiosity
    import cleanrl_utils.evals.human_eval

    return (
        cleanrl.human_curiosity.input_sequence,
        cleanrl.human_curiosity.make_env,
        cleanrl_utils.evals.human_eval.evaluate,
    )


MODELS = {
    "dqn": dqn,
    "dqn_atari": dqn_atari,
    "dqn_jax": dqn_jax,
    "dqn_atari_jax": dqn_atari_jax,
    "dqn_curiosity": dqn_curiosity,
    "c51": c51,
    "c51_atari": c51_atari,
    "c51_jax": c51_jax,
    "c51_atari_jax": c51_atari_jax,
    "c51_curiosity": c51_curiosity,
    "ppo_atari_envpool_xla_jax_scan": ppo_atari_envpool_xla_jax_scan,
    "ppo_atari": ppo_atari,
    "ppo_curiosity": ppo_curiosity,
    "sac_atari": sac_atari,
    "sac_curiosity": sac_curiosity,
    "rainbow_atari": rainbow_atari,
    "rainbow_curiosity": rainbow_curiosity,
    "random_curiosity": random_curiosity,
    "human_curiosity": human_curiosity,
}
