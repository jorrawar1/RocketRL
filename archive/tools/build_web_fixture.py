"""Round a mission fixture's floats for web delivery.

The full-precision export in artifacts/ stays the parity reference; this
produces a smaller copy for the browser, where four decimals is far beyond
what a canvas can show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _round(value, places):
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, list):
        return [_round(item, places) for item in value]
    if isinstance(value, dict):
        return {key: _round(item, places) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/mission_42.json"))
    parser.add_argument("--output", type=Path, default=Path("web/data/mission_42.json"))
    parser.add_argument("--places", type=int, default=4)
    args = parser.parse_args()

    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    compact = _round(fixture, args.places)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compact, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )

    before = args.input.stat().st_size / 1024
    after = args.output.stat().st_size / 1024
    print(f"{args.input} {before:,.0f} KB -> {args.output} {after:,.0f} KB "
          f"({after / before:.0%})")
    print(f"frames: {len(compact['frames'])}  outcome: {compact['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
