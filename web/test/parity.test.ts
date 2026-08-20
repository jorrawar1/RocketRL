import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { stepDynamics } from "../src/sim/physics";
import { VehicleModel } from "../src/sim/vehicle";
import type { Fixture, StateVec } from "../src/types";

/* The full-precision export is the golden reference. `artifacts/` is
 * gitignored, so regenerate it with
 *   python scripts/export_sample_return_fixture.py
 * before running this suite. The web copy under public/ is rounded to four
 * decimals and is useless for parity. */
const REFERENCE = fileURLToPath(
  new URL("../../artifacts/mission_42.json", import.meta.url),
);

const available = existsSync(REFERENCE);
const suite = available ? describe : describe.skip;

suite("physics parity with the Python recording", () => {
  const fx = available
    ? (JSON.parse(readFileSync(REFERENCE, "utf8")) as Fixture)
    : ({ frames: [], mission_events: [] } as unknown as Fixture);

  /** Steps where the transition is pure dynamics: airborne, no pad snap. */
  const airborneWindows = (): [number, number][] => {
    const stepOf = (label: string) =>
      fx.mission_events.find((e) => e.label === label)?.step;
    const windows: [number, number][] = [];
    const legs: [string, string][] = [
      ["BASE DEPARTURE", "SAMPLE PAD TOUCHDOWN"],
      ["RETURN DEPARTURE", "BASE TOUCHDOWN"],
    ];
    for (const [from, to] of legs) {
      const a = stepOf(from);
      const b = stepOf(to);
      // Stop one short of touchdown: that frame is replaced by the rest pose.
      if (a !== undefined && b !== undefined) windows.push([a, b - 2]);
    }
    return windows;
  };

  it("finds both airborne legs in the recording", () => {
    expect(airborneWindows()).toHaveLength(2);
  });

  it("reproduces every airborne step to float64 precision", () => {
    let checked = 0;
    let worst = 0;
    let worstAt = -1;

    for (const [lo, hi] of airborneWindows()) {
      for (let i = lo; i <= hi; i++) {
        const frame = fx.frames[i]!;
        const next = fx.frames[i + 1]!;
        const vehicle = new VehicleModel(
          fx.vehicle.dry_mass,
          fx.vehicle.dry_inertia,
          fx.vehicle.body_height,
          frame.payload_mass > 0
            ? {
                mass: fx.payload.mass,
                offsetBody: fx.payload.offset_body,
                width: fx.payload.width,
                height: fx.payload.height,
              }
            : null,
        );
        // Each exported frame stores the action that produced that frame, so
        // the transition frame[i] -> frame[i+1] uses next.action.
        const got = stepDynamics(frame.state, next.action, vehicle);
        for (let k = 0; k < 7; k++) {
          const err = Math.abs(got[k]! - (next.state as StateVec)[k]!);
          if (err > worst) {
            worst = err;
            worstAt = i;
          }
        }
        checked++;
      }
    }

    // Both sides are IEEE-754 float64 running the same operation order, so the
    // only drift available is the last bit or two of each accumulator.
    expect(checked).toBeGreaterThan(1000);
    expect({ worst, worstAt }).toMatchObject({ worst: expect.any(Number) });
    expect(worst).toBeLessThan(1e-9);
  });
});
