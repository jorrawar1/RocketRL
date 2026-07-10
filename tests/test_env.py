"""Env-level tests: Gymnasium conformance, seeding, rays, overrides, episode end."""

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketenv import RocketEnv
from rocketenv.config import Config
from rocketenv.physics import FUEL, THETA, VY, X, Y


def test_gymnasium_check_env():
    check_env(RocketEnv(), skip_render_check=True)


def test_spaces():
    env = RocketEnv()
    assert env.observation_space.shape == (13,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (2,)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.shape == (13,)
    assert "state" in info


def test_reset_seeding_reproducible():
    e1, e2 = RocketEnv(), RocketEnv()
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    assert np.array_equal(o1, o2)
    o3, _ = e1.reset(seed=124)
    assert not np.array_equal(o1, o3)


def test_full_episode_determinism():
    rng = np.random.default_rng(7)
    actions = rng.uniform([0, -1], [1, 1], size=(1200, 2)).astype(np.float32)

    def rollout():
        env = RocketEnv()
        obs, _ = env.reset(seed=42)
        trace = [obs]
        rewards = []
        for a in actions:
            obs, r, term, trunc, _ = env.step(a)
            trace.append(obs)
            rewards.append(r)
            if term or trunc:
                break
        return np.stack(trace), np.array(rewards)

    t1, r1 = rollout()
    t2, r2 = rollout()
    assert np.array_equal(t1, t2)
    assert np.array_equal(r1, r2)


def test_rays_flat_terrain_geometry():
    env = RocketEnv()
    env.reset(seed=0)
    # Place the rocket at a known altitude and check closed-form distances.
    env.state[X], env.state[Y] = 50.0, 30.0
    d = env.ray_distances()
    assert d.shape == (5,)
    # Middle ray points straight down (-90 deg): distance == altitude.
    assert d[2] == pytest.approx(30.0)
    # -150 and -30 deg rays: altitude / sin(30 deg) = 60 == max range.
    assert d[0] == pytest.approx(60.0)
    assert d[4] == pytest.approx(60.0)
    # -120 and -60 deg rays: altitude / sin(60 deg).
    assert d[1] == pytest.approx(30.0 / math.sin(math.radians(60)))
    assert d[3] == pytest.approx(d[1])


def test_params_override_at_reset():
    env = RocketEnv()
    env.reset(seed=0, options={"g": 1.62, "wind_x": 2.0})
    assert env.cfg.g == 1.62
    assert env.cfg.wind_x == 2.0
    # Overrides do not leak into the next episode.
    env.reset(seed=0)
    assert env.cfg.g == Config().g

    with pytest.raises(KeyError):
        env.reset(seed=0, options={"not_a_param": 1})


def test_timeout_is_truncation_not_termination():
    # Free fall from 5 km in a tall world: 20 s of falling covers ~2 km, so the
    # episode can only end by timeout — which must be truncation.
    env = RocketEnv(config=Config(world_h=10_000.0, spawn_y_lo=5000.0,
                                  spawn_y_hi=5000.0, spawn_x_half=0.0,
                                  spawn_v_max=0.0, spawn_theta_max=0.0,
                                  spawn_omega_max=0.0))
    env.reset(seed=0)
    terminated = truncated = False
    for _ in range(Config().max_steps):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
        if terminated or truncated:
            break
    assert truncated and not terminated


def test_free_fall_terminates_with_contact():
    env = RocketEnv()
    env.reset(seed=1)
    for _ in range(10_000):
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0]))
        if term:
            break
    assert term
    assert info["outcome"] in ("LOSS OF VEHICLE", "TOUCHDOWN")
    # Free fall from ~85 m hits hard -> must be a crash with a big penalty.
    assert info["outcome"] == "LOSS OF VEHICLE"
    assert r < -50


def test_env_clips_actions():
    env = RocketEnv()
    env.reset(seed=0)
    s_wild = env.step(np.array([99.0, -99.0]))[0]
    env.reset(seed=0)
    s_clip = env.step(np.array([1.0, -1.0]))[0]
    assert np.array_equal(s_wild, s_clip)


def test_config_json_dump(tmp_path):
    import json

    p = tmp_path / "constants.json"
    Config().dump_json(str(p))
    d = json.loads(p.read_text())
    cfg = Config()
    assert d["dt"] == pytest.approx(1 / 60)
    assert d["T_max"] == pytest.approx(cfg.twr * cfg.m * cfg.g)
    assert d["I"] == pytest.approx(cfg.m * cfg.H**2 / 12)
