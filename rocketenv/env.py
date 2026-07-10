"""RocketEnv — Gymnasium API over the pure physics/terrain/reward modules.

Observation (float32, egocentric only — the agent never sees the raw level):
  [ dx, dy,            pad - rocket, / world size
    vx, vy,            / v_ref
    sin(theta), cos(theta),
    omega,             / omega_ref
    fuel_frac,
    ray_0 .. ray_{N-1} ]  raycast distances / ray_max_range

Action (float32, clipped by the env): [throttle in [0,1], gimbal in [-1,1]].

Episode end:
  * terminated — first body-endpoint contact with terrain, or out of bounds.
  * truncated  — max_steps timeout (never terminated: PPO bootstraps V(s')
    on truncation but not termination).

Per-episode parameter overrides: reset(seed=..., options={"g": 1.62, ...}) —
any Config field.  This is the domain-randomization hook for Phase 3.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import reward as rw
from .config import DEFAULT_CONFIG, Config
from .physics import (
    FUEL,
    OMEGA,
    STATE_DIM,
    THETA,
    VX,
    VY,
    X,
    Y,
    body_endpoints,
    step_dynamics,
)
from .terrain import FlatTerrain, Terrain


class RocketEnv(gym.Env):
    metadata = {"render_modes": []}  # all rendering lives in play.py

    def __init__(self, config: Config | None = None, terrain: Terrain | None = None):
        self.base_config = config or DEFAULT_CONFIG
        self.terrain = terrain or FlatTerrain()
        self.cfg = self.base_config  # active per-episode config

        obs_dim = 8 + self.base_config.n_rays
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.state = np.zeros(STATE_DIM, dtype=np.float64)
        self.steps = 0

    # ------------------------------------------------------------------ api
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options, if given, is a dict of Config field overrides for this
        episode (e.g. {"g": 1.62, "wind_x": 2.0})."""
        super().reset(seed=seed)
        self.cfg = self.base_config.with_overrides(options)
        cfg, rng = self.cfg, self.np_random

        cx = cfg.world_w / 2.0
        self.state = np.array(
            [
                rng.uniform(cx - cfg.spawn_x_half, cx + cfg.spawn_x_half),
                rng.uniform(cfg.spawn_y_lo, cfg.spawn_y_hi),
                rng.uniform(-cfg.spawn_v_max, cfg.spawn_v_max),
                rng.uniform(-cfg.spawn_v_max, cfg.spawn_v_max),
                rng.uniform(-cfg.spawn_theta_max, cfg.spawn_theta_max),
                rng.uniform(-cfg.spawn_omega_max, cfg.spawn_omega_max),
                cfg.fuel_0,
            ],
            dtype=np.float64,
        )
        self.steps = 0
        return self._observe(), self._info()

    def step(self, action):
        cfg = self.cfg
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            [0.0, -1.0],
            [1.0, 1.0],
        )
        prev_state = self.state
        self.state = step_dynamics(prev_state, action, cfg)
        self.steps += 1

        pad_y = self.terrain.height_at(cfg.pad_x)
        reward = rw.step_reward(prev_state, self.state, cfg, pad_y)
        terminated = False
        truncated = False
        outcome = None

        if self._out_of_bounds():
            terminated = True
            r_term, outcome = rw.out_of_bounds_reward(cfg)
            reward += r_term
        elif self._contact():
            terminated = True
            r_term, outcome = rw.contact_reward(self.state, cfg)
            reward += r_term
        elif self.steps >= cfg.max_steps:
            truncated = True

        return self._observe(), reward, terminated, truncated, self._info(outcome)

    # -------------------------------------------------------------- internals
    def _contact(self) -> bool:
        base, tip = body_endpoints(self.state, self.cfg)
        return (
            base[1] <= self.terrain.height_at(base[0])
            or tip[1] <= self.terrain.height_at(tip[0])
        )

    def _out_of_bounds(self) -> bool:
        x, y = self.state[X], self.state[Y]
        return x < 0.0 or x > self.cfg.world_w or y > self.cfg.world_h

    def ray_distances(self) -> np.ndarray:
        """N raycast distances (m), fixed world-frame downward fan."""
        cfg = self.cfg
        angles = np.linspace(cfg.ray_angle_lo, cfg.ray_angle_hi, cfg.n_rays)
        ox, oy = self.state[X], self.state[Y]
        return np.array(
            [
                self.terrain.ray_distance(
                    ox, oy, math.cos(a), math.sin(a), cfg.ray_max_range
                )
                for a in angles
            ],
            dtype=np.float64,
        )

    def _observe(self) -> np.ndarray:
        cfg, s = self.cfg, self.state
        pad_y = self.terrain.height_at(cfg.pad_x)
        obs = np.empty(8 + cfg.n_rays, dtype=np.float64)
        obs[0] = (cfg.pad_x - s[X]) / cfg.world_w
        obs[1] = (pad_y - s[Y]) / cfg.world_h
        obs[2] = s[VX] / cfg.v_ref
        obs[3] = s[VY] / cfg.v_ref
        obs[4] = math.sin(s[THETA])
        obs[5] = math.cos(s[THETA])
        obs[6] = s[OMEGA] / cfg.omega_ref
        obs[7] = s[FUEL] / cfg.fuel_0
        obs[8:] = self.ray_distances() / cfg.ray_max_range
        return obs.astype(np.float32)

    def _info(self, outcome: str | None = None) -> dict:
        info = {"state": self.state.copy(), "steps": self.steps}
        if outcome is not None:
            info["outcome"] = outcome
        return info
