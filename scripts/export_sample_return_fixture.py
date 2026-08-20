"""Export a deterministic crater sample-return rollout for browser work."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rocketenv.sample_return.serialization import export_fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller", choices=("scripted",), default="scripted")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = export_fixture(
        args.output, seed=args.seed, controller=args.controller
    )
    print(f"wrote {args.output}")
    print(f"frames: {len(fixture['frames'])}")
    print(f"outcome: {fixture['outcome']}")
    return 0 if fixture["outcome"] == "SAMPLE_RETURNED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
