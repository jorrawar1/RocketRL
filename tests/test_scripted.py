"""The scripted expert must actually land — it is the label source for
behaviour cloning, so a regression here silently poisons any dataset."""

import numpy as np
import pytest

from rocketenv import RocketEnv, generate_terrain
from rocketenv.config import Config
from rocketenv.scripted import scripted_action

CFG = Config()


def fly(env, seed, options=None, noise_std=0.0, rng=None):
    env.reset(seed=seed, options=options)
    for _ in range(env.cfg.max_steps):
        a = scripted_action(env.state, env.cfg, env.terrain, noise_std, rng)
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            return info.get("outcome", "TIMEOUT")
    return "TIMEOUT"


def test_lands_on_flat():
    env = RocketEnv()
    outcomes = [fly(env, s) for s in range(20)]
    assert all(o == "TOUCHDOWN" for o in outcomes), outcomes


def test_lands_on_generated_terrain():
    landed = 0
    for map_idx in range(1, 9):
        terr, pad_x = generate_terrain(np.random.default_rng(map_idx), CFG)
        env = RocketEnv(terrain=terr)
        for seed in range(3):
            if fly(env, seed, options={"pad_x": pad_x}) == "TOUCHDOWN":
                landed += 1
    assert landed >= 22, f"only {landed}/24 landings on generated maps"


def test_survives_action_noise():
    # Demonstration collection perturbs actions; the expert must still mostly
    # land, or the dataset fills up with discarded episodes.
    env = RocketEnv()
    rng = np.random.default_rng(0)
    outcomes = [fly(env, s, noise_std=0.05, rng=rng) for s in range(20)]
    landed = sum(o == "TOUCHDOWN" for o in outcomes)
    assert landed >= 16, f"only {landed}/20 landed with action noise"


def test_action_is_in_space():
    env = RocketEnv()
    env.reset(seed=0)
    for _ in range(200):
        a = scripted_action(env.state, env.cfg, env.terrain)
        assert env.action_space.contains(a.astype(np.float32)), a
        env.step(a)


def test_is_deterministic_without_noise():
    env = RocketEnv()
    env.reset(seed=3)
    a1 = scripted_action(env.state, env.cfg, env.terrain)
    a2 = scripted_action(env.state, env.cfg, env.terrain)
    assert np.array_equal(a1, a2)


def test_adapts_to_low_gravity():
    # Hand-tuned gains still cope with a moon-gravity override (they would not
    # cope with everything - that is the point of the RL comparison).
    env = RocketEnv()
    assert fly(env, 0, options={"g": 1.62}) == "TOUCHDOWN"
