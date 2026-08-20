"""Small, explicit reward scaffold and payload-aware touchdown checks."""

from __future__ import annotations

import math

import numpy as np

from ..physics import FUEL, OMEGA, THETA, VX, VY, X, Y
from .vehicle import VehicleModel, dry_body_center


def potential(state: np.ndarray, target_x: float, target_y: float, k: float) -> float:
    return -k * math.hypot(float(state[X]) - target_x, float(state[Y]) - target_y)


def flight_reward(
    previous: np.ndarray,
    state: np.ndarray,
    *,
    target_x: float,
    target_y: float,
    cfg,
    fuel_penalty: float,
) -> float:
    """Potential difference plus the same readable costs as the base task."""
    reward = -cfg.step_penalty
    reward += cfg.gamma * potential(
        state, target_x, target_y, cfg.shaping_k
    ) - potential(previous, target_x, target_y, cfg.shaping_k)
    reward -= fuel_penalty * max(0.0, float(previous[FUEL] - state[FUEL]))
    return float(reward)


def sticks_upright(state: np.ndarray, cfg, vehicle: VehicleModel) -> bool:
    """Existing leg-pivot model evaluated with the active mass properties."""
    tip_energy, barrier = _tip_energy_and_barrier(state, cfg, vehicle)
    return barrier > 0.0 and tip_energy < barrier


def _tip_energy_and_barrier(
    state: np.ndarray, cfg, vehicle: VehicleModel
) -> tuple[float, float]:
    """Return touchdown tip energy and the active vehicle's stability barrier."""
    half_base = cfg.leg_half_w
    # Evaluate both leg pivots in body coordinates relative to the actual
    # combined COM.  The lower potential barrier is the conservative side for
    # an off-center pod and reduces exactly to the legacy symmetric geometry.
    com_x, com_y = map(float, vehicle.com_offset_body)
    pivots = (
        np.array([-half_base - com_x, -vehicle.body_height / 2.0 - com_y]),
        np.array([half_base - com_x, -vehicle.body_height / 2.0 - com_y]),
    )
    theta = float(state[THETA])
    pivot = min(
        pivots,
        key=lambda point: vehicle.total_mass
        * cfg.g
        * (np.linalg.norm(point) + body_vertical(point, theta)),
    )
    pivot_radius = float(np.linalg.norm(pivot))
    com_height = -body_vertical(pivot, theta)
    barrier = vehicle.total_mass * cfg.g * (pivot_radius - com_height)
    if barrier <= 0.0:
        return 0.0, barrier
    pivot_inertia = vehicle.total_inertia + vehicle.total_mass * pivot_radius**2
    # Horizontal COM velocity contributes angular momentum with the pivot's
    # vertical lever arm in world coordinates.
    world_pivot_y = body_vertical(pivot, theta)
    angular_momentum = (
        vehicle.total_inertia * abs(float(state[OMEGA]))
        + vehicle.total_mass * abs(world_pivot_y) * abs(float(state[VX]))
    )
    tip_energy = angular_momentum**2 / (2.0 * pivot_inertia)
    return tip_energy, barrier


def body_vertical(vector_body: np.ndarray, theta: float) -> float:
    """World-y component of a body-frame vector without allocating a matrix."""
    x_body, y_body = map(float, vector_body)
    return math.sin(theta) * x_body + math.cos(theta) * y_body


def landing_result(
    state: np.ndarray,
    *,
    target_x: float,
    cfg,
    vehicle: VehicleModel,
) -> tuple[bool, str]:
    """Classify a target-pad contact as safe, tipped, or crashed."""
    body_center_x = float(dry_body_center(state, vehicle)[0])
    on_pad = abs(body_center_x - target_x) < cfg.pad_half_w
    survivable = (
        abs(float(state[VY])) <= cfg.land_vy_max
        and abs(float(state[VX])) <= cfg.land_vx_max
    )
    if on_pad and survivable and sticks_upright(state, cfg, vehicle):
        return True, "TOUCHDOWN"
    if on_pad and survivable:
        return False, "TIPPED"
    return False, "CRASHED"


def failed_landing_reward(
    state: np.ndarray,
    *,
    target_x: float,
    cfg,
    vehicle: VehicleModel,
    failure_penalty: float,
) -> float:
    """Grade a failed contact by proximity, impact speed, and tip stability.

    A near-safe miss can earn up to ``partial_credit`` without ever approaching
    the reward for a successful touchdown.  Severe multi-axis failures remain
    close to the unchanged failure penalty, while each individual defect stays
    informative.  Only the terminal state matters, so this does not prescribe
    a flight path.
    """
    body_center_x = float(dry_body_center(state, vehicle)[0])
    distance_past_pad = max(
        abs(body_center_x - target_x) - cfg.pad_half_w, 0.0
    )
    position_quality = cfg.partial_dist_ref / (
        cfg.partial_dist_ref + distance_past_pad
    )

    vx_excess = max(abs(float(state[VX])) - cfg.land_vx_max, 0.0)
    vy_excess = max(abs(float(state[VY])) - cfg.land_vy_max, 0.0)
    vx_half_excess = max(cfg.crash_speed_ref - cfg.land_vx_max, 1e-8)
    vy_half_excess = max(cfg.crash_speed_ref - cfg.land_vy_max, 1e-8)
    vx_quality = vx_half_excess / (vx_half_excess + vx_excess)
    vy_quality = vy_half_excess / (vy_half_excess + vy_excess)

    tip_energy, barrier = _tip_energy_and_barrier(state, cfg, vehicle)
    if barrier <= 0.0:
        stability_quality = 0.0
    else:
        tip_excess = max(tip_energy / barrier - 1.0, 0.0)
        stability_quality = 3.0 / (3.0 + tip_excess)

    # Vertical impact gates the partial credit so a stable free-fall crash
    # cannot be rated highly.  The nonzero baseline keeps braking informative
    # while an early policy is still learning lateral and attitude control.
    control_quality = (
        0.80 + 0.10 * vx_quality + 0.10 * stability_quality
    )
    landing_quality = position_quality * vy_quality * control_quality
    return float(failure_penalty + cfg.partial_credit * landing_quality)


__all__ = [
    "failed_landing_reward",
    "flight_reward",
    "landing_result",
    "potential",
    "sticks_upright",
]
