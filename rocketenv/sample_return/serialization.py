"""Stable browser-fixture export for deterministic mission rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import SampleReturnEnv
from .scripted import scripted_sample_return_action


SCHEMA_VERSION = 1


def _frame(env: SampleReturnEnv, action: np.ndarray) -> dict:
    return {
        "t": env.steps * env.cfg.dt,
        "state": [float(value) for value in env.state],
        "action": [float(value) for value in action],
        "phase": env.phase.name,
        "payload_mass": float(env.payload_mass),
        "payload_fill": float(env.sampling_progress),
        "com_offset_body": [
            float(value) for value in env.vehicle.com_offset_body
        ],
    }


def rollout_fixture(seed: int = 42, controller: str = "scripted") -> dict:
    """Run one deterministic mission and return JSON-compatible primitives."""
    if controller != "scripted":
        raise ValueError("fixture export currently supports controller='scripted'")
    env = SampleReturnEnv()
    env.reset(seed=seed)
    frames = [_frame(env, np.zeros(2, dtype=np.float64))]
    terminated = truncated = False
    while not (terminated or truncated):
        action = scripted_sample_return_action(env)
        _, _, terminated, truncated, _ = env.step(action)
        frames.append(_frame(env, action))

    payload = env.payload_spec
    return {
        "schema_version": SCHEMA_VERSION,
        "mission_seed": int(seed),
        "terrain": {"vertices": env.terrain.vertices.tolist()},
        "base_pad": {
            "x": env.base_x,
            "y": env.terrain.base_y,
            "half_width": env.cfg.pad_half_w,
        },
        "sample_pad": {
            "x": env.sample_x,
            "y": env.terrain.sample_y,
            "half_width": env.cfg.pad_half_w,
        },
        "vehicle": {
            "dry_mass": env.dry_vehicle.dry_mass,
            "dry_inertia": env.dry_vehicle.dry_inertia,
            "body_height": env.dry_vehicle.body_height,
            "max_thrust": env.cfg.T_max,
        },
        "payload": {
            "sample_id": payload.sample_id,
            "mass": payload.mass,
            "offset_body": [payload.offset_body_x, payload.offset_body_y],
            "width": payload.width,
            "height": payload.height,
        },
        "controller": controller,
        "mission_events": [dict(event) for event in env.events],
        "frames": frames,
        "outcome": env.outcome,
    }


def export_fixture(path: str | Path, *, seed: int = 42, controller: str = "scripted") -> dict:
    fixture = rollout_fixture(seed=seed, controller=controller)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8"
    )
    return fixture


__all__ = ["SCHEMA_VERSION", "export_fixture", "rollout_fixture"]
