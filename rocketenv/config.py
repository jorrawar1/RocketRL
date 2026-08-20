"""All physics/episode/reward constants in one place — the single source of truth.

Shared dynamics constants live here. ``dump_json`` exports the exact values so
the Python simulator and browser implementation cannot silently drift.

Coordinate & sign conventions (load-bearing — see also physics.py):
  * World is y-up.  Ground at y = 0, gravity pulls toward -y.
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
    # TWR 1.8 felt too weak to arrest a free fall (human feedback 2026-07-09);
    # 2.3 recovers a 2 s fall in ~1.5 s. Difficulty held via burn_rate instead.
    twr: float = 2.3                # thrust-to-weight at full throttle
    H: float = 4.0                  # m, rocket height (inertia / moment arm)
    # 10 deg (was 15) tames twitchy attitude control at the higher TWR.
    phi_max: float = math.radians(10.0)  # rad, max gimbal deflection
    fuel_0: float = 100.0           # starting fuel
    # 10 fuel/s = ~23 s at hover throttle: enough slack for a human descent
    # profile (12 left assisted pilots landing on fumes or short of them).
    burn_rate: float = 10.0         # fuel/s at full throttle

    # --- reserved randomization axes (no-op defaults; Phase 3) --------------
    drag_coeff: float = 0.0         # quadratic drag: F = -c * |v| * v
    wind_x: float = 0.0             # N, constant horizontal force
    wind_gust_x: float = 0.0        # N, reserved gust slot (always 0 in v1)
    thrust_multiplier: float = 1.0  # reserved for engine failure

    # --- pad ----------------------------------------------------------------
    pad_x: float = 50.0             # m, pad center (overridden per map)
    pad_half_w: float = 5.0         # m

    # --- procedural terrain (terrain.generate_terrain) -----------------------
    terrain_amp: float = 22.0       # m, max hill height
    terrain_res: float = 4.0        # m, polyline vertex spacing

    # --- spawn envelope (uniform ranges, seeded RNG) -------------------------
    spawn_y_lo: float = 80.0
    spawn_y_hi: float = 90.0
    spawn_x_half: float = 20.0      # x ~ U[world_w/2 - this, world_w/2 + this]
    spawn_v_max: float = 2.0        # |vx|, |vy| <= this
    spawn_theta_max: float = 0.1    # rad
    spawn_omega_max: float = 0.1    # rad/s

    # --- episode -------------------------------------------------------------
    # 30 s @ 60 Hz -> truncation (never termination). Sized so that a sound
    # traverse-and-land flight on a far-pad terrain map fits with margin;
    # fuel (~23 s of hover) is the binding constraint, not the clock.
    max_steps: int = 1800

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
    # Landing model: legs absorb a hard vertical hit, but the rocket must
    # also not TIP OVER — an energy check (reward.sticks_upright) compares
    # lateral speed + spin + tilt against the barrier of rotating the CoM
    # over a leg.  Rewards decisive suicide-burn landings; punishes arriving
    # sideways or spinning.
    leg_half_w: float = 1.5         # m, leg base half-width (tip-over pivot)
    land_vy_max: float = 5.0        # m/s, max vertical impact the legs absorb
    land_vx_max: float = 3.5        # m/s, max lateral shear the legs survive
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
