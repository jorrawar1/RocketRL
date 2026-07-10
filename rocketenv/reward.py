"""Reward function, isolated for constant iteration. Pure math — no pygame,
no gymnasium.  All coefficients live in Config.

Structure:
  * per-step: time penalty + potential-based shaping + attitude penalty.
  * terminal: soft landing bonus / graded partial credit / crash penalty.

Shaping uses the strictly potential-based form gamma*Phi(s') - Phi(s) with
Phi(s) = -k * distance(rocket, pad center at ground level), so it cannot be
farmed by cycling.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config
from .physics import FUEL, OMEGA, THETA, VX, VY, X, Y

# Terminal outcome tags (the harness renders these; keep them stable)
TOUCHDOWN = "TOUCHDOWN"
TIPPED = "TIP-OVER"
CRASH = "LOSS OF VEHICLE"
OUT_OF_BOUNDS = "OUT OF BOUNDS"


def potential(state: np.ndarray, cfg: Config, pad_y: float) -> float:
    """Phi(s) = -k * distance to pad center (at ground level)."""
    d = math.hypot(state[X] - cfg.pad_x, state[Y] - pad_y)
    return -cfg.shaping_k * d


def step_reward(
    prev_state: np.ndarray, state: np.ndarray, cfg: Config, pad_y: float
) -> float:
    r = -cfg.step_penalty
    r += cfg.gamma * potential(state, cfg, pad_y) - potential(prev_state, cfg, pad_y)
    # Attitude penalty, fading in linearly below att_alt (no hard gate — a
    # discontinuity there invites hovering at the boundary).
    w = min(max((cfg.att_alt - state[Y]) / cfg.att_fade, 0.0), 1.0)
    r -= w * (cfg.att_c_theta * abs(state[THETA]) + cfg.att_c_omega * abs(state[OMEGA]))
    return r


def sticks_upright(state: np.ndarray, cfg: Config) -> bool:
    """Tip-over check at the instant of contact.

    Model: the vertical impact is absorbed by the legs (checked separately);
    what can still kill the landing is rotational energy about the downhill
    leg edge.  Treat the touchdown as an inelastic pivot at a leg (half-base
    b from the centerline): angular momentum about the pivot combines body
    spin and horizontal CoM velocity.  The rocket tips if that kinetic
    energy exceeds the potential barrier of lifting the CoM over the pivot.
    Worst-case signs (spin and drift conspiring) are assumed.
    """
    b, arm = cfg.leg_half_w, cfg.L
    theta = abs(state[THETA])
    if theta >= math.atan2(b, arm):  # already past the balance point
        return False
    r_pivot = math.hypot(arm, b)
    com_h = arm * math.cos(theta) + b * math.sin(theta)
    barrier = cfg.m * cfg.g * (r_pivot - com_h)
    i_pivot = cfg.I + cfg.m * r_pivot * r_pivot
    ang_mom = cfg.I * abs(state[OMEGA]) + cfg.m * arm * abs(state[VX])
    tip_energy = ang_mom * ang_mom / (2.0 * i_pivot)
    return tip_energy < barrier


def contact_reward(state: np.ndarray, cfg: Config) -> tuple[float, str]:
    """Terminal reward on first terrain contact. Returns (reward, outcome)."""
    on_pad = abs(state[X] - cfg.pad_x) < cfg.pad_half_w
    survivable = (
        abs(state[VY]) <= cfg.land_vy_max and abs(state[VX]) <= cfg.land_vx_max
    )
    if on_pad and survivable and sticks_upright(state, cfg):
        fuel_frac = state[FUEL] / cfg.fuel_0
        return cfg.reward_land + cfg.reward_fuel_bonus * fuel_frac, TOUCHDOWN
    # Graded near-miss: gentleness x pad proximity earns partial credit;
    # impact speed alone drives the crash penalty. Smooth in both factors.
    impact = math.hypot(state[VX], state[VY])
    gentle = min(max(1.0 - impact / cfg.crash_speed_ref, 0.0), 1.0)
    near = min(max(1.0 - abs(state[X] - cfg.pad_x) / cfg.partial_dist_ref, 0.0), 1.0)
    r = cfg.partial_credit * gentle * near + cfg.reward_crash * (1.0 - gentle)
    outcome = TIPPED if (on_pad and survivable) else CRASH
    return r, outcome


def out_of_bounds_reward(cfg: Config) -> tuple[float, str]:
    return cfg.reward_crash, OUT_OF_BOUNDS
