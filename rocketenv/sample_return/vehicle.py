"""Rigid vehicle properties and payload-aware world geometry.

The flight state stores the combined center of mass.  Render and contact
geometry still belong to the dry rocket body, so helpers in this module first
recover the dry-body center before positioning the rocket, engine, or payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..physics import THETA, X, Y, nose_direction
from .mission_types import PayloadSpec


@dataclass(frozen=True)
class VehicleModel:
    """Mass properties of the dry rocket and an optional rigid payload."""

    dry_mass: float
    dry_inertia: float
    body_height: float
    engine_body_x: float
    engine_body_y: float
    payload: PayloadSpec | None = None

    @classmethod
    def from_config(
        cls, cfg: Config, payload: PayloadSpec | None = None
    ) -> "VehicleModel":
        """Build the legacy rod-shaped vehicle described by ``cfg``."""
        return cls(
            dry_mass=cfg.m,
            dry_inertia=cfg.I,
            body_height=cfg.H,
            engine_body_x=0.0,
            engine_body_y=-cfg.L,
            payload=payload,
        )

    @property
    def total_mass(self) -> float:
        payload_mass = self.payload.mass if self.payload is not None else 0.0
        return self.dry_mass + payload_mass

    @property
    def com_offset_body(self) -> np.ndarray:
        """Combined COM relative to the dry-body center, in body coordinates."""
        if self.payload is None or self.payload.mass == 0.0:
            return np.zeros(2, dtype=np.float64)
        payload_offset = np.array(
            [self.payload.offset_body_x, self.payload.offset_body_y],
            dtype=np.float64,
        )
        return (self.payload.mass / self.total_mass) * payload_offset

    @property
    def total_inertia(self) -> float:
        """Vehicle inertia about the combined center of mass."""
        if self.payload is None or self.payload.mass == 0.0:
            return self.dry_inertia

        payload = self.payload
        com = self.com_offset_body
        payload_offset = np.array(
            [payload.offset_body_x, payload.offset_body_y], dtype=np.float64
        )
        dry_parallel_axis = self.dry_mass * float(np.dot(com, com))
        payload_local = (
            payload.mass * (payload.width**2 + payload.height**2) / 12.0
        )
        payload_from_com = payload_offset - com
        payload_parallel_axis = payload.mass * float(
            np.dot(payload_from_com, payload_from_com)
        )
        return (
            self.dry_inertia
            + dry_parallel_axis
            + payload_local
            + payload_parallel_axis
        )

    @property
    def engine_offset_from_com_body(self) -> np.ndarray:
        engine_from_body = np.array(
            [self.engine_body_x, self.engine_body_y], dtype=np.float64
        )
        return engine_from_body - self.com_offset_body


def body_to_world(vector_body: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a body-frame vector into the y-up world frame."""
    x_body, y_body = np.asarray(vector_body, dtype=np.float64)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    return np.array(
        [
            cos_theta * x_body - sin_theta * y_body,
            sin_theta * x_body + cos_theta * y_body,
        ],
        dtype=np.float64,
    )


def dry_body_center(state: np.ndarray, vehicle: VehicleModel) -> np.ndarray:
    """World position of the dry-body center for a combined-COM state."""
    state_array = np.asarray(state)
    com_world = state_array[[X, Y]]
    offset_world = body_to_world(vehicle.com_offset_body, float(state_array[THETA]))
    return com_world - offset_world


def body_endpoints(
    state: np.ndarray, cfg: Config, vehicle: VehicleModel
) -> tuple[np.ndarray, np.ndarray]:
    """Return the dry rocket's ``(base, tip)`` world positions.

    ``cfg`` is retained in the signature so this helper is a drop-in companion
    to :func:`rocketenv.physics.body_endpoints`.  Vehicle geometry is sourced
    from ``vehicle``.
    """
    del cfg
    center = dry_body_center(state, vehicle)
    nx, ny = nose_direction(float(state[THETA]))
    half_height = vehicle.body_height / 2.0
    offset = np.array([nx * half_height, ny * half_height])
    return center - offset, center + offset


def engine_position(state: np.ndarray, vehicle: VehicleModel) -> np.ndarray:
    """World position of the engine thrust application point."""
    engine_body = np.array(
        [vehicle.engine_body_x, vehicle.engine_body_y], dtype=np.float64
    )
    return dry_body_center(state, vehicle) + body_to_world(
        engine_body, float(state[THETA])
    )


def payload_center(state: np.ndarray, vehicle: VehicleModel) -> np.ndarray | None:
    """World position of the payload center, or ``None`` without a payload."""
    if vehicle.payload is None:
        return None
    payload_offset = np.array(
        [vehicle.payload.offset_body_x, vehicle.payload.offset_body_y],
        dtype=np.float64,
    )
    return dry_body_center(state, vehicle) + body_to_world(
        payload_offset, float(state[THETA])
    )


def payload_corners(state: np.ndarray, vehicle: VehicleModel) -> np.ndarray:
    """Four payload rectangle corners in world coordinates.

    The order is body-frame bottom-left, bottom-right, top-right, top-left.
    A vehicle without a payload returns an empty ``(0, 2)`` array.
    """
    if vehicle.payload is None:
        return np.empty((0, 2), dtype=np.float64)

    payload = vehicle.payload
    half_width = payload.width / 2.0
    half_height = payload.height / 2.0
    center_body = np.array(
        [payload.offset_body_x, payload.offset_body_y], dtype=np.float64
    )
    local_corners = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float64,
    )
    center = dry_body_center(state, vehicle)
    theta = float(state[THETA])
    return np.stack(
        [center + body_to_world(center_body + corner, theta) for corner in local_corners]
    )


def state_with_preserved_dry_body_center(
    state: np.ndarray,
    old_vehicle: VehicleModel,
    new_vehicle: VehicleModel,
) -> np.ndarray:
    """Recenter a state after a rigid mass change without moving its dry body.

    Attachment in the MVP happens while settled, so velocities are copied
    unchanged.  The environment is responsible for zeroing them before calling
    this helper if the vehicle was not already at rest.
    """
    result = np.asarray(state, dtype=np.float64).copy()
    center = dry_body_center(state, old_vehicle)
    new_com_world = center + body_to_world(
        new_vehicle.com_offset_body, float(result[THETA])
    )
    result[[X, Y]] = new_com_world
    return result


__all__ = [
    "VehicleModel",
    "body_endpoints",
    "body_to_world",
    "dry_body_center",
    "engine_position",
    "payload_center",
    "payload_corners",
    "state_with_preserved_dry_body_center",
]
