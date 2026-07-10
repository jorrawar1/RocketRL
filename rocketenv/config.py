"""All physics/episode/reward constants in one place — the single source of truth.

Every quantity a later phase varies (gravity, drag, wind, thrust multiplier,
pad, spawn envelope, ...) lives here and can be overridden per-episode via
``RocketEnv.reset(options={...})``.  ``dump_json`` exports the exact numbers so
the future TypeScript port imports the same constants and the two physics
implementations can't silently drift.

Coordinate & sign conventions (load-bearing — see also physics.py):
  * World is y-up.  Ground at y = 0, gravity pulls toward -y.
    pygame renders y-down; the flip happens ONLY at render time (play.py).
  * theta = tilt from vertical, radians, positive counter-clockwise.
    theta = 0 means the rocket points straight up.
  * Nose (body-up) direction in world frame: n = (-sin(theta), cos(theta)).
    Sanity check: theta = 0 -> n = (0, 1).
  * y is the CoM altitude.  The rocket is a segment from base = p - (H/2)*n
    to tip = p + (H/2)*n; the episode ends on first endpoint-terrain contact.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- integration -------------------------------------------------------
    dt: float = 1.0 / 60.0          # s, fixed timestep, never variable

    # --- world -------------------------------------------------------------
    g: float = 9.81                 # m/s^2 (Moon 1.62, Mars 3.72 later)
    world_w: float = 100.0          # m
    world_h: float = 100.0          # m

    # --- rocket ------------------------------------------------------------
    m: float = 1.0                  # kg, constant in v1 (fuel is massless)
    twr: float = 1.8                # thrust-to-weight at full throttle
    H: float = 4.0                  # m, rocket height (inertia / moment arm)
    phi_max: float = math.radians(15.0)  # rad, max gimbal deflection
    fuel_0: float = 100.0           # starting fuel
    burn_rate: float = 10.0         # fuel/s at full throttle

    # --- reserved randomization axes (no-op defaults; Phase 3) --------------
    drag_coeff: float = 0.0         # quadratic drag: F = -c * |v| * v
    wind_x: float = 0.0             # N, constant horizontal force
    wind_gust_x: float = 0.0        # N, reserved gust slot (always 0 in v1)
    thrust_multiplier: float = 1.0  # reserved for engine failure

    # --- pad ----------------------------------------------------------------
    pad_x: float = 50.0             # m, pad center
    pad_half_w: float = 5.0         # m

    # --- spawn envelope (uniform ranges, seeded RNG) -------------------------
    spawn_y_lo: float = 80.0
    spawn_y_hi: float = 90.0
    spawn_x_half: float = 20.0      # x ~ U[world_w/2 - this, world_w/2 + this]
    spawn_v_max: float = 2.0        # |vx|, |vy| <= this
    spawn_theta_max: float = 0.1    # rad
    spawn_omega_max: float = 0.1    # rad/s

    # --- episode -------------------------------------------------------------
    max_steps: int = 1200           # 20 s @ 60 Hz -> truncation (never termination)

    # --- observation ----------------------------------------------------------
    n_rays: int = 5
    ray_angle_lo: float = math.radians(-150.0)  # world-frame, from +x axis
    ray_angle_hi: float = math.radians(-30.0)
    ray_max_range: float = 60.0     # m
    v_ref: float = 20.0             # m/s, velocity normalization
    omega_ref: float = 5.0          # rad/s

    # --- reward (see reward.py; tune freely) -----------------------------------
    step_penalty: float = 0.01
    shaping_k: float = 0.3          # potential Phi(s) = -k * dist_to_pad
    # Shaping discount in gamma*Phi(s') - Phi(s).  Default 1.0 = plain
    # difference, which telescopes to k*(d_start - d_end) and cannot be farmed.
    # gamma < 1 adds a per-step bonus ~ (1-gamma)*k*dist that REWARDS staying
    # far from the pad — only safe if it exactly matches the agent's discount.
    gamma: float = 1.0
    att_c_theta: float = 0.03       # attitude penalty coefficients
    att_c_omega: float = 0.03
    att_alt: float = 25.0           # m, penalty fully on below (att_alt - att_fade)
    att_fade: float = 10.0          # m, linear fade band up to att_alt
    land_vy_max: float = 2.0        # m/s, soft-landing thresholds
    land_vx_max: float = 1.5
    land_theta_max: float = math.radians(10.0)
    land_omega_max: float = 0.5
    reward_land: float = 100.0
    reward_fuel_bonus: float = 20.0  # * fuel_frac on soft landing
    reward_crash: float = -100.0
    crash_speed_ref: float = 10.0   # m/s, impact speed considered a full crash
    partial_credit: float = 50.0    # max partial credit for a gentle near-miss
    partial_dist_ref: float = 20.0  # m, off-pad distance over which credit fades

    # --- derived (do not set directly) ----------------------------------------
    @property
    def T_max(self) -> float:
        """Max thrust, N (before thrust_multiplier)."""
        return self.twr * self.m * self.g

    @property
    def I(self) -> float:
        """Moment of inertia, rod approximation: (1/12) m H^2."""
        return self.m * self.H * self.H / 12.0

    @property
    def L(self) -> float:
        """CoM-to-engine moment arm: H/2."""
        return self.H / 2.0

    def with_overrides(self, params: dict | None) -> "Config":
        """Return a copy with ``params`` applied (per-episode overrides)."""
        if not params:
            return self
        unknown = set(params) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise KeyError(f"unknown config params: {sorted(unknown)}")
        return dataclasses.replace(self, **params)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["T_max"] = self.T_max
        d["I"] = self.I
        d["L"] = self.L
        return d

    def dump_json(self, path: str) -> None:
        """Export every constant (incl. derived) for the TS port."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


DEFAULT_CONFIG = Config()
