"""Payload-aware deterministic oracle used to validate mission feasibility."""

from __future__ import annotations

import math

import numpy as np

from ..physics import OMEGA, THETA, VX, VY, X, Y
from .controllers import ContinuousController
from .mission_types import MissionPhase
from .vehicle import dry_body_center


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def scripted_sample_return_action(env) -> np.ndarray:
    """Oracle controller with exact terrain and payload mass properties.

    This is intentionally a feasibility/debug baseline, not a learned result.
    Its payload trim is derived from the engine lever arm; all navigation gains
    remain small hand-written PD terms like the original scripted pilot.
    """
    if env.phase in (MissionPhase.SAMPLING, MissionPhase.SUCCESS, MissionPhase.FAILURE):
        return np.zeros(2, dtype=np.float64)

    cfg = env.cfg
    state = env.state
    body_center = dry_body_center(state, env.vehicle)
    x, y = map(float, body_center)
    vx, vy = float(state[VX]), float(state[VY])
    theta, omega = float(state[THETA]), float(state[OMEGA])
    dx = float(env.target_x - x)
    pad_y = float(env.terrain.height_at(env.target_x))
    clearance = y - pad_y

    highest_ground = float(np.max(env.terrain.ys))
    cruise_altitude = min(cfg.world_h - 25.0, highest_ground + 24.0)

    # Approach speed falls continuously with remaining distance, which gives
    # the narrow crater pad enough braking room without a planner/state graph.
    if abs(dx) > 5.0:
        speed_limit = min(6.0, max(1.2, 0.55 * math.sqrt(abs(dx))))
        vx_des = math.copysign(speed_limit, dx)
        vy_des = _clip(0.42 * (cruise_altitude - y), -3.5, 4.5)
    else:
        vx_des = _clip(0.34 * dx, -1.6, 1.6)
        vy_des = -min(4.5, 0.22 * max(clearance, 0.0) + 0.65)

    ax_des = _clip(0.85 * (vx_des - vx), -3.2, 3.2)
    thrust_angle_des = _clip(-0.92 * ax_des / cfg.g, -0.32, 0.32)

    com_x, com_y = map(float, env.vehicle.com_offset_body)
    trim_phi = -math.atan2(com_x, cfg.L + com_y)
    trim_cmd = trim_phi / cfg.phi_max
    theta_des = thrust_angle_des - trim_phi
    if clearance < 8.0:
        # Keep the physical trim attitude but fade navigation tilt before touch.
        theta_des = -trim_phi + thrust_angle_des * max(clearance, 0.0) / 8.0

    attitude_correction = -5.4 * (theta_des - theta) + 2.8 * omega
    gimbal = _clip(trim_cmd + attitude_correction, -1.0, 1.0)

    total_mass_ratio = env.vehicle.total_mass / env.vehicle.dry_mass
    lift_cosine = max(math.cos(theta + gimbal * cfg.phi_max), 0.30)
    hover = total_mass_ratio / (
        cfg.twr * cfg.thrust_multiplier * lift_cosine
    )
    throttle = _clip(hover + 0.34 * (vy_des - vy), 0.0, 1.0)
    return np.array([throttle, gimbal], dtype=np.float64)


class PayloadAwareScriptedController:
    """Protocol adapter for consumers that operate on observations."""

    def __init__(self, env):
        self.env = env

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return scripted_sample_return_action(self.env)


__all__ = [
    "ContinuousController",
    "PayloadAwareScriptedController",
    "scripted_sample_return_action",
]
