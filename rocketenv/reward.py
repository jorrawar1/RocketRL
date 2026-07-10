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


def is_soft_landing(state: np.ndarray, cfg: Config) -> bool:
    return (
        abs(state[X] - cfg.pad_x) < cfg.pad_half_w
        and abs(state[VY]) < cfg.land_vy_max
        and abs(state[VX]) < cfg.land_vx_max
        and abs(state[THETA]) < cfg.land_theta_max
        and abs(state[OMEGA]) < cfg.land_omega_max
    )


def contact_reward(state: np.ndarray, cfg: Config) -> tuple[float, str]:
    """Terminal reward on first terrain contact. Returns (reward, outcome)."""
    if is_soft_landing(state, cfg):
        fuel_frac = state[FUEL] / cfg.fuel_0
        return cfg.reward_land + cfg.reward_fuel_bonus * fuel_frac, TOUCHDOWN
    # Graded near-miss: gentleness x pad proximity earns partial credit;
    # impact speed alone drives the crash penalty. Smooth in both factors.
    impact = math.hypot(state[VX], state[VY])
    gentle = min(max(1.0 - impact / cfg.crash_speed_ref, 0.0), 1.0)
    near = min(max(1.0 - abs(state[X] - cfg.pad_x) / cfg.partial_dist_ref, 0.0), 1.0)
    r = cfg.partial_credit * gentle * near + cfg.reward_crash * (1.0 - gentle)
    return r, CRASH


def out_of_bounds_reward(cfg: Config) -> tuple[float, str]:
    return cfg.reward_crash, OUT_OF_BOUNDS
