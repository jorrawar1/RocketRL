"""Evaluate a controller on a deterministic held-out mission seed set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketenv.sample_return.evaluation import evaluate_policy
from rocketenv.sample_return.scripted import PayloadAwareScriptedController
from rocketenv.sample_return.training import TrainingTask, make_training_env


def _scripted_controller(env) -> PayloadAwareScriptedController:
    return PayloadAwareScriptedController(env.unwrapped)


def _print_summary(summary: dict) -> None:
    print(
        f"success: {summary['success_count']}/{summary['episode_count']} "
        f"({summary['success_rate']:.1%})"
    )
    print(f"sample reached: {summary['sample_reached_rate']:.1%}")
    print(f"sample acquired: {summary['sample_acquired_rate']:.1%}")
    outcomes = ", ".join(
        f"{name}={count}" for name, count in summary["outcome_counts"].items()
    )
    print(f"outcomes: {outcomes}")

    mean_fuel = summary["successful_fuel_remaining_mean"]
    minimum_fuel = summary["successful_fuel_remaining_min"]
    if mean_fuel is not None:
        print(
            f"successful fuel remaining: mean={mean_fuel:.3f}, "
            f"minimum={minimum_fuel:.3f}"
        )

    for label, key in (("sample", "sample_touchdown"), ("return", "return_touchdown")):
        touchdown = summary[key]
        if touchdown["count"]:
            print(
                f"{label} touchdown worst absolute: "
                f"vx={touchdown['max_abs_vx_m_s']:.3f} m/s, "
                f"vy={touchdown['max_abs_vy_m_s']:.3f} m/s, "
                f"tilt={touchdown['max_abs_tilt_deg']:.3f} deg, "
                f"omega={touchdown['max_abs_angular_velocity_deg_s']:.3f} deg/s"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("scripted",), default="scripted")
    parser.add_argument(
        "--task",
        choices=tuple(task.value for task in TrainingTask),
        default=TrainingTask.FULL_MISSION.value,
    )
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument(
        "--no-auto-advance-sampling",
        action="store_false",
        dest="auto_advance_sampling",
        help="retain the visual sampling dwell as policy decisions",
    )
    parser.set_defaults(auto_advance_sampling=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.action_repeat < 1:
        parser.error("--action-repeat must be positive")

    task = TrainingTask(args.task)
    env = make_training_env(
        task=task,
        action_repeat=args.action_repeat,
        auto_advance_sampling=args.auto_advance_sampling,
    )
    try:
        seeds = range(args.seed_start, args.seed_start + args.episodes)
        report = evaluate_policy(env, _scripted_controller, seeds)
    finally:
        env.close()

    report["controller"] = args.controller
    report["task"] = task.value
    report["action_repeat"] = args.action_repeat
    report["auto_advance_sampling"] = args.auto_advance_sampling
    _print_summary(report["summary"])

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
