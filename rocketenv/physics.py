"""Pure 2D rigid-body dynamics. No globals, no RNG, no mutation of inputs.

This module must stay importable without pygame or gymnasium — it is reused
verbatim by the training code and mirrored by the TypeScript port.

State layout (float64): s = [x, y, vx, vy, theta, omega, fuel]

Sign conventions (verified by the tests and by flying):
  * y-up world; gravity acts along -y.
  * theta: tilt from vertical, positive counter-clockwise; theta = 0 is upright.
  * Nose direction: n = (-sin theta, cos theta).
  * Gimbal angle phi rotates the thrust direction relative to the body axis;
    thrust direction = (-sin(theta + phi), cos(theta + phi)).
  * Torque from gimbaled thrust applied at the engine (r = -L*n from CoM):
    tau = r x F = -L * T * sin(phi).  Positive gimbal command -> negative
    torque -> theta decreases (rocket tips clockwise / to the right).

Integration: semi-implicit Euler — velocities first, then positions.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config

# State vector indices
X, Y, VX, VY, THETA, OMEGA, FUEL = range(7)
STATE_DIM = 7


def nose_direction(theta: float) -> tuple[float, float]:
    """Body-up (nose) unit vector in world frame."""
    return (-math.sin(theta), math.cos(theta))


def body_endpoints(state: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(base, tip) world positions of the rocket segment."""
    nx, ny = nose_direction(state[THETA])
    half = cfg.L  # H/2
    pos = state[[X, Y]]
    offset = np.array([nx * half, ny * half])
    return pos - offset, pos + offset


def step_dynamics(state: np.ndarray, action: np.ndarray, cfg: Config) -> np.ndarray:
    """Advance one fixed timestep. Pure: returns a new array, inputs untouched."""
    x, y, vx, vy, theta, omega, fuel = state

    throttle = min(max(float(action[0]), 0.0), 1.0)
    gimbal_cmd = min(max(float(action[1]), -1.0), 1.0)
    phi = gimbal_cmd * cfg.phi_max

    thrust = throttle * cfg.T_max * cfg.thrust_multiplier if fuel > 0.0 else 0.0

    # Forces
    fx = thrust * -math.sin(theta + phi) + cfg.wind_x + cfg.wind_gust_x
    fy = thrust * math.cos(theta + phi) - cfg.m * cfg.g
    if cfg.drag_coeff > 0.0:  # quadratic drag, no-op by default
        speed = math.hypot(vx, vy)
        fx -= cfg.drag_coeff * speed * vx
        fy -= cfg.drag_coeff * speed * vy

    tau = -cfg.L * thrust * math.sin(phi)

    # Semi-implicit Euler: velocity first, then position
    vx += (fx / cfg.m) * cfg.dt
    vy += (fy / cfg.m) * cfg.dt
    x += vx * cfg.dt
    y += vy * cfg.dt
    omega += (tau / cfg.I) * cfg.dt
    theta += omega * cfg.dt

    fuel = max(0.0, fuel - cfg.burn_rate * throttle * cfg.dt)

    return np.array([x, y, vx, vy, theta, omega, fuel], dtype=np.float64)
