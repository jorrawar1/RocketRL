/* Port of the tip-over model in rocketenv/sample_return/reward.py.
 *
 * Only the classification half is ported. Reward shaping stays in Python —
 * nothing in the browser needs it. */

import { OMEGA, THETA, VX, VY, type StateVec, type Vec2 } from "../types";
import { SIM } from "./config";
import type { VehicleModel } from "./vehicle";

/** World-y component of a body-frame vector. */
function bodyVertical(v: Vec2, theta: number): number {
  return Math.sin(theta) * v[0] + Math.cos(theta) * v[1];
}

/** Touchdown rotational energy about a leg pivot, and the potential barrier. */
export function tipEnergyAndBarrier(
  state: StateVec,
  vehicle: VehicleModel,
): { energy: number; barrier: number } {
  const halfBase = SIM.legHalfW;
  const [comX, comY] = vehicle.comOffsetBody;
  const theta = state[THETA];

  // Both leg pivots, in body coordinates relative to the combined COM. The
  // lower barrier is the conservative side for an off-centre pod, and reduces
  // exactly to the symmetric case when the pod is absent.
  const candidates: Vec2[] = [
    [-halfBase - comX, -vehicle.bodyHeight / 2 - comY],
    [halfBase - comX, -vehicle.bodyHeight / 2 - comY],
  ];
  let pivot = candidates[0]!;
  let best = Infinity;
  for (const p of candidates) {
    const cost =
      vehicle.totalMass * SIM.g * (Math.hypot(p[0], p[1]) + bodyVertical(p, theta));
    if (cost < best) {
      best = cost;
      pivot = p;
    }
  }

  const pivotRadius = Math.hypot(pivot[0], pivot[1]);
  const comHeight = -bodyVertical(pivot, theta);
  const barrier = vehicle.totalMass * SIM.g * (pivotRadius - comHeight);
  if (barrier <= 0) return { energy: 0, barrier };

  const pivotInertia =
    vehicle.totalInertia + vehicle.totalMass * pivotRadius * pivotRadius;
  const worldPivotY = bodyVertical(pivot, theta);
  const angularMomentum =
    vehicle.totalInertia * Math.abs(state[OMEGA]) +
    vehicle.totalMass * Math.abs(worldPivotY) * Math.abs(state[VX]);
  return {
    energy: (angularMomentum * angularMomentum) / (2 * pivotInertia),
    barrier,
  };
}

export function sticksUpright(state: StateVec, vehicle: VehicleModel): boolean {
  const { energy, barrier } = tipEnergyAndBarrier(state, vehicle);
  return barrier > 0 && energy < barrier;
}

export type ContactKind = "TOUCHDOWN" | "TIPPED" | "CRASHED";

/** Classify a pad contact. `bodyCenterX` is the *dry* body centre. */
export function landingResult(
  state: StateVec,
  bodyCenterX: number,
  targetX: number,
  vehicle: VehicleModel,
): { safe: boolean; kind: ContactKind } {
  const onPad = Math.abs(bodyCenterX - targetX) < SIM.padHalfW;
  const survivable =
    Math.abs(state[VY]) <= SIM.landVyMax && Math.abs(state[VX]) <= SIM.landVxMax;
  if (onPad && survivable && sticksUpright(state, vehicle)) {
    return { safe: true, kind: "TOUCHDOWN" };
  }
  if (onPad && survivable) return { safe: false, kind: "TIPPED" };
  return { safe: false, kind: "CRASHED" };
}
