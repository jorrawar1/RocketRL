"""Controller and held-out evaluation tests."""

import dataclasses
import json

import numpy as np
import pytest

from rocketenv.sample_return.config import SampleReturnConfig
from rocketenv.sample_return.controllers import CallableController
from rocketenv.sample_return.evaluation import evaluate_policy, summarize_episodes
from rocketenv.sample_return.scripted import PayloadAwareScriptedController
from rocketenv.sample_return.training import TrainingTask, make_training_env


class CountingZeroController:
    def __init__(self):
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(2, dtype=np.float64)


def _short_training_env(max_steps: int = 5):
    base = SampleReturnConfig()
    config = dataclasses.replace(
        base, flight=dataclasses.replace(base.flight, max_steps=max_steps)
    )
    return make_training_env(
        task=TrainingTask.FULL_MISSION,
        action_repeat=4,
        config=config,
    )


def test_callable_controller_validates_shape_and_finiteness_without_clipping():
    reset_calls = []
    controller = CallableController(
        lambda observation: [1.2, -0.25],
        reset_fn=lambda: reset_calls.append(True),
    )
    controller.reset()
    assert reset_calls == [True]
    assert controller.act(np.zeros(16)) == pytest.approx([1.2, -0.25])

    with pytest.raises(ValueError, match="shape"):
        CallableController(lambda observation: [0.5]).act(np.zeros(16))
    with pytest.raises(ValueError, match="finite"):
        CallableController(lambda observation: [np.nan, 0.0]).act(np.zeros(16))


def test_evaluation_preserves_seed_order_resets_controller_and_is_deterministic():
    env = _short_training_env()
    controller = CountingZeroController()
    seeds = [12, 7]

    first = evaluate_policy(env, lambda unused_env: controller, seeds)
    second = evaluate_policy(env, lambda unused_env: CountingZeroController(), seeds)

    assert controller.reset_calls == len(seeds)
    assert first == second
    assert first["seeds"] == seeds
    assert [episode["seed"] for episode in first["episodes"]] == seeds
    assert all(episode["physics_steps"] == 5 for episode in first["episodes"])
    assert all(episode["decision_steps"] == 2 for episode in first["episodes"])
    assert first["summary"]["outcome_counts"] == {"TIMEOUT": 2}
    json.dumps(first, allow_nan=False)


def test_evaluation_rejects_action_outside_wrapped_environment_space():
    env = _short_training_env(max_steps=1)

    with pytest.raises(ValueError, match="outside the action space"):
        evaluate_policy(
            env,
            lambda unused_env: CallableController(
                lambda observation: np.array([1.01, 0.0])
            ),
            [0],
        )


def test_evaluation_keeps_configured_and_active_payload_mass_distinct():
    env = _short_training_env(max_steps=1)
    report = evaluate_policy(
        env,
        lambda unused_env: CountingZeroController(),
        [0],
        reset_options={"payload_mass": 0.4},
    )

    episode = report["episodes"][0]
    assert not episode["sample_acquired"]
    assert episode["payload_mass"] == pytest.approx(0.4)
    assert episode["active_payload_mass"] == pytest.approx(0.0)


def test_evaluation_requires_at_least_one_seed():
    env = _short_training_env(max_steps=1)
    with pytest.raises(ValueError, match="at least one"):
        evaluate_policy(env, lambda unused_env: CountingZeroController(), [])


def test_summary_uses_physical_success_and_success_only_fuel():
    touchdown = {
        "vx_m_s": -1.0,
        "vy_m_s": -2.0,
        "speed_m_s": 5**0.5,
        "tilt_deg": -3.0,
        "angular_velocity_deg_s": 4.0,
    }

    def episode(*, success, outcome, fuel, fuel_used, reached):
        return {
            "success": success,
            "outcome": outcome,
            "sample_reached": reached,
            "sample_acquired": success,
            "fuel_remaining": fuel,
            "fuel_used": fuel_used,
            "outbound_fuel_used": fuel_used / 2 if success else None,
            "return_fuel_used": fuel_used / 2 if success else None,
            "episode_return": 10.0,
            "decision_steps": 4,
            "physics_steps": 16,
            "sample_touchdown": touchdown if reached else None,
            "return_touchdown": touchdown if success else None,
        }

    episodes = [
        episode(
            success=True,
            outcome="SAMPLE_RETURNED",
            fuel=8.0,
            fuel_used=2.0,
            reached=True,
        ),
        episode(
            success=False,
            outcome="CRASHED_OUTBOUND",
            fuel=99.0,
            fuel_used=1.0,
            reached=False,
        ),
        episode(
            success=True,
            outcome=None,
            fuel=6.0,
            fuel_used=4.0,
            reached=True,
        ),
    ]

    summary = summarize_episodes(episodes)

    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["successful_fuel_remaining_mean"] == pytest.approx(7.0)
    assert summary["successful_fuel_remaining_min"] == pytest.approx(6.0)
    assert summary["successful_fuel_used_mean"] == pytest.approx(3.0)
    assert summary["successful_outbound_fuel_used_mean"] == pytest.approx(1.5)
    assert summary["successful_return_fuel_used_mean"] == pytest.approx(1.5)
    assert summary["outcome_counts"] == {
        "CRASHED_OUTBOUND": 1,
        "SAMPLE_RETURNED": 1,
        "TASK_SUCCESS": 1,
    }
    assert summary["sample_touchdown"]["count"] == 2
    assert summary["sample_touchdown"]["max_abs_vy_m_s"] == pytest.approx(2.0)
    assert summary["return_touchdown"]["count"] == 2


def test_scripted_full_mission_smoke_reports_physical_metrics():
    env = make_training_env(action_repeat=4)
    report = evaluate_policy(
        env,
        lambda wrapped: PayloadAwareScriptedController(wrapped.unwrapped),
        [42],
    )

    episode = report["episodes"][0]
    assert episode["success"]
    assert episode["outcome"] == "SAMPLE_RETURNED"
    assert episode["sample_touchdown"] is not None
    assert episode["return_touchdown"] is not None
    assert episode["outbound_fuel_used"] > 0.0
    assert episode["return_fuel_used"] > 0.0
    assert episode["decision_steps"] < episode["physics_steps"]
    json.dumps(report, allow_nan=False)
