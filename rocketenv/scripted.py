"""Scripted PD guidance — the classical baseline.

Two roles in this project:
  * the "expert" whose actions a behaviour-cloning dataset is built from, and
  * the baseline the RL policy has to beat.

This is hand-tuned classical control, no learning.  Note that it *cheats*
relative to an RL agent: it reads the raw state (exact position, velocity) and
queries the terrain directly, whereas a policy only sees the 13-float
egocentric observation.  That asymmetry is the point — it is why a cloned
network cannot simply inherit this controller's competence, and why the
controller cannot be retuned automatically when gravity/wind/thrust change.

Flies a two-phase profile: traverse at altitude until above the pad, then
descend on a velocity schedule that scales with clearance.

Pure math — no pygame, no torch, no gymnasium.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config
from .physics import OMEGA, THETA, VX, VY, X, Y
from .terrain import Terrain

CRUISE_ALT = 55.0      # m, altitude held while closing horizontal distance
ARRIVAL_DX = 4.0       # m, |dx| below which the descent phase begins
FLARE_ALT = 8.0        # m, clearance below which commanded tilt is faded out


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def scripted_action(
    state: np.ndarray, cfg: Config, terrain: Terrain, noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return the expert's [throttle, gimbal] for this state.

    noise_std > 0 perturbs the returned action (requires ``rng``).  Used when
    collecting demonstrations: executing slightly noisy actions widens the
    visited state distribution, while the *clean* action stays the label.
    """
    x, y = state[X], state[Y]
    vx, vy = state[VX], state[VY]
    theta, omega = state[THETA], state[OMEGA]

    pad_y = terrain.height_at(cfg.pad_x)
    dx = cfg.pad_x - x
    clearance = y - pad_y

    # --- outer loop: desired velocities -----------------------------------
    if abs(dx) > ARRIVAL_DX:                      # traverse toward the pad
        vx_des = _clip(0.35 * dx, -5.0, 5.0)
        vy_des = _clip((CRUISE_ALT - y) * 0.4, -3.0, 3.0)
    else:                                         # descend onto it
        vx_des = _clip(0.3 * dx, -2.0, 2.0)
        vy_des = -min(6.0, 0.28 * clearance + 0.8)

    # --- horizontal: velocity error -> commanded tilt ----------------------
    ax_des = _clip(0.8 * (vx_des - vx), -3.0, 3.0)
    # thrust_x ~ -T sin(theta): accelerating toward +x needs theta < 0
    theta_des = _clip(-ax_des / cfg.g * 0.9, -0.35, 0.35)
    if clearance < FLARE_ALT:                     # straighten up for touchdown
        theta_des *= max(clearance, 0.0) / FLARE_ALT

    # --- attitude: PD on tilt error (positive gimbal -> negative torque) ----
    gimbal = _clip(-(6.0 * (theta_des - theta) - 3.0 * omega), -1.0, 1.0)

    # --- vertical: hover feed-forward + velocity error ----------------------
    tilt_loss = max(math.cos(theta), 0.3)         # tilted thrust lifts less
    hover = 1.0 / (cfg.twr * cfg.thrust_multiplier * tilt_loss)
    throttle = _clip(hover + 0.35 * (vy_des - vy), 0.0, 1.0)

    action = np.array([throttle, gimbal], dtype=np.float64)
    if noise_std > 0.0:
        if rng is None:
            raise ValueError("noise_std > 0 requires an rng")
        action = action + rng.normal(0.0, noise_std, size=2)
        action = np.clip(action, [0.0, -1.0], [1.0, 1.0])
    return action
