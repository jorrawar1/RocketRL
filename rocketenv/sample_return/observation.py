"""Observation construction for the sample-return Gymnasium environment."""

from __future__ import annotations

import math

import numpy as np

from ..physics import FUEL, OMEGA, THETA, VX, VY, X, Y
from .mission_types import MissionPhase


def observation_names(n_rays: int) -> tuple[str, ...]:
    """Return flat-observation labels in their exact array order."""
    if n_rays < 1:
        raise ValueError("n_rays must be positive")
    return (
        "target_dx",
        "target_dy",
        "velocity_x",
        "velocity_y",
        "sin_theta",
        "cos_theta",
        "angular_velocity",
        "fuel_fraction",
        *(f"terrain_ray_{index}" for index in range(n_rays)),
        "payload_attached",
        "phase_outbound",
        "phase_return",
    )


# Stable base-environment contract. Target deltas are divided by world width /
# height, velocities by v_ref, angular velocity by omega_ref, terrain rays by
# ray_max_range, and fuel by initial fuel. The remaining entries are already in
# [-1, 1] or are binary flags.
OBSERVATION_NAMES = observation_names(5)
OBSERVATION_DIM = len(OBSERVATION_NAMES)
OBSERVATION_INDEX = {
    name: index for index, name in enumerate(OBSERVATION_NAMES)
}

# The recurrent policy receives the complete physical observation.  The GRU
# remains useful for carrying context across the mission and for future hidden
# vehicle-property experiments.
_ACTOR_HIDDEN_NAMES: frozenset[str] = frozenset()
ACTOR_OBSERVATION_NAMES = tuple(
    name for name in OBSERVATION_NAMES if name not in _ACTOR_HIDDEN_NAMES
)
ACTOR_OBSERVATION_DIM = len(ACTOR_OBSERVATION_NAMES)
ACTOR_OBSERVATION_INDEX = {
    name: index for index, name in enumerate(ACTOR_OBSERVATION_NAMES)
}
_ACTOR_SOURCE_INDICES = np.asarray(
    [OBSERVATION_INDEX[name] for name in ACTOR_OBSERVATION_NAMES],
    dtype=np.intp,
)


def ray_distances(env) -> np.ndarray:
    """World-frame downward sensor fan, matching the original environment."""
    cfg = env.cfg
    angles = np.linspace(cfg.ray_angle_lo, cfg.ray_angle_hi, cfg.n_rays)
    return np.asarray(
        [
            env.terrain.ray_distance(
                float(env.state[X]),
                float(env.state[Y]),
                math.cos(angle),
                math.sin(angle),
                cfg.ray_max_range,
            )
            for angle in angles
        ],
        dtype=np.float64,
    )


def structured_observation(env) -> dict[str, float | np.ndarray]:
    """Named form used for inspection while the public API stays a flat Box."""
    state = env.state
    target_y = env.terrain.height_at(env.target_x)
    return {
        "target_delta": np.array(
            [
                (env.target_x - state[X]) / env.cfg.world_w,
                (target_y - state[Y]) / env.cfg.world_h,
            ],
            dtype=np.float64,
        ),
        "velocity": state[[VX, VY]] / env.cfg.v_ref,
        "attitude": np.array(
            [math.sin(state[THETA]), math.cos(state[THETA])], dtype=np.float64
        ),
        "angular_velocity": float(state[OMEGA] / env.cfg.omega_ref),
        "fuel_fraction": float(state[FUEL] / env.cfg.fuel_0),
        "rays": ray_distances(env) / env.cfg.ray_max_range,
        "payload_attached": float(env.mission_state.payload_attached),
        "phase_outbound": float(env.phase == MissionPhase.OUTBOUND),
        "phase_return": float(env.phase == MissionPhase.RETURN),
    }


def flat_observation(env) -> np.ndarray:
    named = structured_observation(env)
    flat = np.concatenate(
        [
            named["target_delta"],
            named["velocity"],
            named["attitude"],
            np.array(
                [named["angular_velocity"], named["fuel_fraction"]],
                dtype=np.float64,
            ),
            named["rays"],
            np.array(
                [
                    named["payload_attached"],
                    named["phase_outbound"],
                    named["phase_return"],
                ],
                dtype=np.float64,
            ),
        ]
    )
    return flat.astype(np.float32)


def actor_observation(full_observation: np.ndarray) -> np.ndarray:
    """Project the base observation into the recurrent policy's configured view."""
    full = np.asarray(full_observation, dtype=np.float32)
    if full.shape != (OBSERVATION_DIM,):
        raise ValueError(
            f"expected full observation shape {(OBSERVATION_DIM,)}, "
            f"got {full.shape}"
        )
    return full[_ACTOR_SOURCE_INDICES]


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "ACTOR_OBSERVATION_INDEX",
    "ACTOR_OBSERVATION_NAMES",
    "OBSERVATION_DIM",
    "OBSERVATION_INDEX",
    "OBSERVATION_NAMES",
    "actor_observation",
    "flat_observation",
    "observation_names",
    "ray_distances",
    "structured_observation",
]
