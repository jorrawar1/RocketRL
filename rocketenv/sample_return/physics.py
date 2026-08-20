"""Pure payload-aware rigid-body dynamics for the sample-return vehicle."""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..physics import FUEL, OMEGA, THETA, VX, VY, X, Y
from .vehicle import VehicleModel


def thrust_direction(theta: float, phi: float) -> np.ndarray:
    """World-frame thrust unit vector for attitude ``theta`` and gimbal ``phi``."""
    return np.array(
        [-math.sin(theta + phi), math.cos(theta + phi)], dtype=np.float64
    )


def thrust_torque(vehicle: VehicleModel, thrust: float, phi: float) -> float:
    """Torque about the combined COM from thrust applied at the engine."""
    engine_x, engine_y = vehicle.engine_offset_from_com_body
    force_x = -thrust * math.sin(phi)
    force_y = thrust * math.cos(phi)
    return float(engine_x * force_y - engine_y * force_x)


def step_dynamics(
    state: np.ndarray,
    action: np.ndarray,
    cfg: Config,
    vehicle: VehicleModel,
) -> np.ndarray:
    """Advance one fixed timestep without mutating any input.

    Engine thrust is rated from dry mass, while gravity and translational
    acceleration use total mass.  Fuel remains massless, matching the original
    simulator.
    """
    x, y, vx, vy, theta, omega, fuel = state

    throttle = min(max(float(action[0]), 0.0), 1.0)
    gimbal_cmd = min(max(float(action[1]), -1.0), 1.0)
    phi = gimbal_cmd * cfg.phi_max

    dry_max_thrust = cfg.twr * vehicle.dry_mass * cfg.g
    thrust = (
        throttle * dry_max_thrust * cfg.thrust_multiplier if fuel > 0.0 else 0.0
    )

    total_mass = vehicle.total_mass
    direction = thrust_direction(theta, phi)
    fx = thrust * direction[0] + cfg.wind_x + cfg.wind_gust_x
    fy = thrust * direction[1] - total_mass * cfg.g
    if cfg.drag_coeff > 0.0:
        speed = math.hypot(vx, vy)
        fx -= cfg.drag_coeff * speed * vx
        fy -= cfg.drag_coeff * speed * vy

    tau = thrust_torque(vehicle, thrust, phi)

    # Semi-implicit Euler, kept in the same order as the legacy dynamics.
    vx += (fx / total_mass) * cfg.dt
    vy += (fy / total_mass) * cfg.dt
    x += vx * cfg.dt
    y += vy * cfg.dt
    omega += (tau / vehicle.total_inertia) * cfg.dt
    theta += omega * cfg.dt

    fuel = max(0.0, fuel - cfg.burn_rate * throttle * cfg.dt)

    return np.array([x, y, vx, vy, theta, omega, fuel], dtype=np.float64)


__all__ = ["step_dynamics", "thrust_direction", "thrust_torque"]
