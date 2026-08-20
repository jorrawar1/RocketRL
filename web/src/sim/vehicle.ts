/* Port of rocketenv/sample_return/vehicle.py — mass properties only.
 * The render-side geometry helpers live in src/geometry.ts. */

import type { Vec2 } from "../types";
import { SIM } from "./config";

export interface PayloadSpec {
  mass: number;
  offsetBody: Vec2;
  width: number;
  height: number;
}

export class VehicleModel {
  readonly totalMass: number;
  readonly comOffsetBody: Vec2;
  readonly totalInertia: number;

  constructor(
    readonly dryMass: number,
    readonly dryInertia: number,
    readonly bodyHeight: number,
    readonly payload: PayloadSpec | null = null,
  ) {
    const pm = payload?.mass ?? 0;
    this.totalMass = dryMass + pm;

    if (!payload || pm === 0) {
      this.comOffsetBody = [0, 0];
      this.totalInertia = dryInertia;
      return;
    }

    const k = pm / this.totalMass;
    const com: Vec2 = [k * payload.offsetBody[0], k * payload.offsetBody[1]];
    this.comOffsetBody = com;

    // Parallel-axis on both sides: the dry body about the new combined COM,
    // plus the payload's own rectangular inertia carried out to its offset.
    const dryParallel = dryMass * (com[0] * com[0] + com[1] * com[1]);
    const payloadLocal =
      (pm * (payload.width * payload.width + payload.height * payload.height)) /
      12;
    const dx = payload.offsetBody[0] - com[0];
    const dy = payload.offsetBody[1] - com[1];
    const payloadParallel = pm * (dx * dx + dy * dy);
    this.totalInertia = dryInertia + dryParallel + payloadLocal + payloadParallel;
  }

  /** Engine position relative to the combined COM, in body coordinates. */
  get engineOffsetFromCom(): Vec2 {
    return [
      0 - this.comOffsetBody[0],
      -this.bodyHeight / 2 - this.comOffsetBody[1],
    ];
  }

  withPayload(payload: PayloadSpec | null): VehicleModel {
    return new VehicleModel(
      this.dryMass,
      this.dryInertia,
      this.bodyHeight,
      payload,
    );
  }
}

export function dryVehicle(): VehicleModel {
  return new VehicleModel(SIM.dryMass, SIM.dryInertia, SIM.bodyHeight, null);
}
