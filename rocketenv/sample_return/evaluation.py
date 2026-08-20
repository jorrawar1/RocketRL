"""Deterministic, reward-independent evaluation for sample-return policies."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .controllers import ContinuousController
from .env import SAMPLE_RETURNED


ControllerFactory = Callable[[Any], ContinuousController]


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _touchdown_metrics(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    vx = _finite_float(value.get("vx"))
    vy = _finite_float(value.get("vy"))
    theta = _finite_float(value.get("theta"))
    omega = _finite_float(value.get("omega"))
    if None in (vx, vy, theta, omega):
        return None
    assert vx is not None and vy is not None
    assert theta is not None and omega is not None
    return {
        "vx_m_s": vx,
        "vy_m_s": vy,
        "speed_m_s": math.hypot(vx, vy),
        "tilt_deg": math.degrees(theta),
        "angular_velocity_deg_s": math.degrees(omega),
    }


def _validated_action(env: Any, action: object) -> object:
    space = env.action_space
    try:
        candidate = np.asarray(action, dtype=getattr(space, "dtype", None))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("controller returned an invalid action") from exc
    if not space.contains(candidate):
        raise ValueError(f"controller action is outside the action space: {candidate}")
    return candidate


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _max_abs_angle_deg(
    info: Mapping[str, Any],
    info_key: str,
    touchdown_component: str,
) -> float | None:
    values: list[float] = []
    current = _finite_float(info.get(info_key))
    if current is not None:
        values.append(abs(current))
    for key in ("sample_touchdown", "return_touchdown"):
        touchdown = info.get(key)
        if isinstance(touchdown, Mapping):
            component = _finite_float(touchdown.get(touchdown_component))
            if component is not None:
                values.append(abs(component))
    if not values:
        return None
    return math.degrees(max(values))


def run_episode(
    env: Any,
    controller: ContinuousController,
    *,
    seed: int,
    reset_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one episode and return JSON-compatible physical metrics."""
    observation, initial_info = env.reset(
        seed=int(seed), options=dict(reset_options or {})
    )
    controller.reset()

    terminated = truncated = False
    episode_return = 0.0
    decisions = 0
    info: Mapping[str, Any] = initial_info
    while not (terminated or truncated):
        action = _validated_action(env, controller.act(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("environment returned a non-finite reward")
        episode_return += reward_value
        decisions += 1

    outcome_value = info.get("outcome")
    outcome = None if outcome_value is None else str(outcome_value)
    task_success = info.get("task_success")
    success = (
        bool(task_success)
        if task_success is not None
        else outcome == SAMPLE_RETURNED
    )

    base = _base_env(env)
    physics_steps_value = info.get("physics_steps")
    if physics_steps_value is None:
        physics_steps_value = getattr(base, "steps", info.get("steps", decisions))
    physics_steps = int(physics_steps_value)
    decision_steps = int(info.get("decision_steps", decisions))
    dt = _finite_float(getattr(getattr(base, "cfg", None), "dt", None))

    initial_fuel = _finite_float(initial_info.get("fuel_remaining"))
    fuel_remaining = _finite_float(info.get("fuel_remaining"))
    fuel_used = (
        initial_fuel - fuel_remaining
        if initial_fuel is not None and fuel_remaining is not None
        else None
    )

    sample_touchdown = _touchdown_metrics(info.get("sample_touchdown"))
    return_touchdown = _touchdown_metrics(info.get("return_touchdown"))
    max_tilt_deg = _max_abs_angle_deg(info, "max_tilt", "theta")
    max_angular_velocity = _max_abs_angle_deg(
        info, "max_angular_velocity", "omega"
    )

    payload_offset = info.get("payload_offset_body")
    if payload_offset is not None:
        payload_offset = [
            _finite_float(value) for value in np.asarray(payload_offset).flat
        ]

    return {
        "seed": int(seed),
        "task": None if info.get("task") is None else str(info["task"]),
        "outcome": outcome,
        "success": success,
        "sample_reached": sample_touchdown is not None,
        "sample_acquired": bool(info.get("has_sample", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode_return": episode_return,
        "decision_steps": decision_steps,
        "physics_steps": physics_steps,
        "elapsed_seconds": physics_steps * dt if dt is not None else None,
        "fuel_remaining": fuel_remaining,
        "fuel_used": fuel_used,
        "outbound_fuel_used": _finite_float(info.get("outbound_fuel_used")),
        "return_fuel_used": _finite_float(info.get("return_fuel_used")),
        "payload_mass": _finite_float(
            info.get("configured_payload_mass", info.get("payload_mass"))
        ),
        "active_payload_mass": _finite_float(info.get("payload_mass")),
        "payload_offset_body": payload_offset,
        "max_abs_tilt_deg": max_tilt_deg,
        "max_abs_angular_velocity_deg_s": max_angular_velocity,
        "sample_touchdown": sample_touchdown,
        "return_touchdown": return_touchdown,
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _touchdown_summary(
    episodes: Sequence[Mapping[str, Any]], key: str
) -> dict[str, int | float | None]:
    touchdowns = [episode[key] for episode in episodes if episode.get(key) is not None]
    fields = (
        "vx_m_s",
        "vy_m_s",
        "tilt_deg",
        "angular_velocity_deg_s",
    )
    result: dict[str, int | float | None] = {"count": len(touchdowns)}
    for field in fields:
        values = [abs(float(touchdown[field])) for touchdown in touchdowns]
        result[f"mean_abs_{field}"] = _mean(values)
        result[f"max_abs_{field}"] = max(values) if values else None
    return result


def summarize_episodes(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate outcomes without treating shaped reward as task success."""
    count = len(episodes)
    successes = [episode for episode in episodes if episode.get("success")]
    sample_reached = sum(bool(episode.get("sample_reached")) for episode in episodes)
    sample_acquired = sum(
        bool(episode.get("sample_acquired")) for episode in episodes
    )

    outcome_counts: Counter[str] = Counter()
    for episode in episodes:
        outcome = episode.get("outcome")
        if outcome is None:
            outcome = "TASK_SUCCESS" if episode.get("success") else "NO_OUTCOME"
        outcome_counts[str(outcome)] += 1

    successful_fuel = [
        float(episode["fuel_remaining"])
        for episode in successes
        if episode.get("fuel_remaining") is not None
    ]
    successful_fuel_used = [
        float(episode["fuel_used"])
        for episode in successes
        if episode.get("fuel_used") is not None
    ]
    successful_outbound_fuel = [
        float(episode["outbound_fuel_used"])
        for episode in successes
        if episode.get("outbound_fuel_used") is not None
    ]
    successful_return_fuel = [
        float(episode["return_fuel_used"])
        for episode in successes
        if episode.get("return_fuel_used") is not None
    ]
    returns = [float(episode["episode_return"]) for episode in episodes]
    decision_steps = [float(episode["decision_steps"]) for episode in episodes]
    physics_steps = [float(episode["physics_steps"]) for episode in episodes]

    return {
        "episode_count": count,
        "success_count": len(successes),
        "success_rate": len(successes) / count if count else 0.0,
        "sample_reached_count": sample_reached,
        "sample_reached_rate": sample_reached / count if count else 0.0,
        "sample_acquired_count": sample_acquired,
        "sample_acquired_rate": sample_acquired / count if count else 0.0,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "mean_episode_return": _mean(returns),
        "mean_decision_steps": _mean(decision_steps),
        "mean_physics_steps": _mean(physics_steps),
        "successful_fuel_remaining_mean": _mean(successful_fuel),
        "successful_fuel_remaining_min": min(successful_fuel)
        if successful_fuel
        else None,
        "successful_fuel_used_mean": _mean(successful_fuel_used),
        "successful_outbound_fuel_used_mean": _mean(successful_outbound_fuel),
        "successful_return_fuel_used_mean": _mean(successful_return_fuel),
        "sample_touchdown": _touchdown_summary(episodes, "sample_touchdown"),
        "return_touchdown": _touchdown_summary(episodes, "return_touchdown"),
    }


def evaluate_policy(
    env: Any,
    controller_factory: ControllerFactory,
    seeds: Iterable[int],
    *,
    reset_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a controller on an exact, ordered sequence of mission seeds."""
    ordered_seeds = [int(seed) for seed in seeds]
    if not ordered_seeds:
        raise ValueError("at least one evaluation seed is required")

    controller = controller_factory(env)
    episodes = [
        run_episode(
            env,
            controller,
            seed=seed,
            reset_options=reset_options,
        )
        for seed in ordered_seeds
    ]
    return {
        "schema_version": 1,
        "seeds": ordered_seeds,
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
    }


__all__ = [
    "ControllerFactory",
    "evaluate_policy",
    "run_episode",
    "summarize_episodes",
]
