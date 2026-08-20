"""Headless text demo of the deterministic crater mission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketenv.physics import FUEL
from rocketenv.sample_return import SampleReturnEnv, scripted_sample_return_action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("scripted",), default="scripted")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = SampleReturnEnv()
    env.reset(seed=args.seed)
    printed = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = scripted_sample_return_action(env)
        _, _, terminated, truncated, info = env.step(action)
        for event in env.events[printed:]:
            if event["label"] in {
                "BASE DEPARTURE",
                "SAMPLE PAD TOUCHDOWN",
                "SAMPLE ACQUIRED",
                "RETURN DEPARTURE",
                "BASE TOUCHDOWN",
            }:
                print(event["label"])
        printed = len(env.events)

    print(f"outcome: {info['outcome']}")
    print(f"fuel remaining: {env.state[FUEL]:.3f}")
    print(f"steps: {env.steps}")
    return 0 if info["outcome"] == "SAMPLE_RETURNED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
