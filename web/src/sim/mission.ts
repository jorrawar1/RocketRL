/* Port of the mission machine in rocketenv/sample_return/env.py.
 *
 * Reward shaping is deliberately absent — the browser only needs the physical
 * mission: fly out, settle on the sample pad, dwell, fly home. Outcomes and
 * event labels match the Python side so a live flight reads the same as a
 * replayed one. */

import { terrainHeightAt } from "../geometry";
import {
  FUEL,
  THETA,
  X,
  Y,
  type MissionFixture,
  type Frame,
  type MissionPhase,
  type StateVec,
  type Vec2,
} from "../types";
import { SIM, T_MAX } from "./config";
import { stepDynamics } from "./physics";
import { buildActorObservation } from "./observation";
import { landingResult } from "./stability";
import { VehicleModel, type PayloadSpec } from "./vehicle";

type PadName = "BASE" | "SAMPLE";

export interface MissionEventLive {
  step: number;
  t: number;
  label: string;
}

export class Mission {
  state: StateVec = [0, 0, 0, 0, 0, 0, 0];
  vehicle: VehicleModel;
  phase: MissionPhase = "OUTBOUND";
  outcome: string | null = null;
  events: MissionEventLive[] = [];
  steps = 0;
  done = false;

  private readonly dry: VehicleModel;
  private readonly loaded: VehicleModel;
  private readonly terrain: Vec2[];
  private readonly baseX: number;
  private readonly sampleX: number;
  private readonly payload: PayloadSpec;

  private groundedPad: PadName | null = null;
  private departurePad: PadName | null = null;
  private contactArmed = false;
  private samplingLeft = 0;
  private lastAction: [number, number] = [0, 0];

  constructor(fx: MissionFixture) {
    this.terrain = fx.terrain.vertices;
    this.baseX = fx.base_pad.x;
    this.sampleX = fx.sample_pad.x;
    this.payload = {
      mass: fx.payload.mass,
      offsetBody: fx.payload.offset_body,
      width: fx.payload.width,
      height: fx.payload.height,
    };
    this.dry = new VehicleModel(
      fx.vehicle.dry_mass,
      fx.vehicle.dry_inertia,
      fx.vehicle.body_height,
      null,
    );
    this.loaded = this.dry.withPayload(this.payload);
    this.vehicle = this.dry;
    this.reset();
  }

  reset(): void {
    this.vehicle = this.dry;
    this.state = this.restState("BASE", SIM.fuel0, this.dry);
    this.phase = "OUTBOUND";
    this.outcome = null;
    this.events = [];
    this.steps = 0;
    this.done = false;
    this.groundedPad = "BASE";
    this.departurePad = "BASE";
    this.contactArmed = false;
    this.samplingLeft = 0;
    this.lastAction = [0, 0];
  }

  get targetPad(): PadName {
    return this.phase === "RETURN" || this.phase === "SUCCESS"
      ? "BASE"
      : "SAMPLE";
  }

  get samplingProgress(): number {
    if (this.phase !== "SAMPLING") return 0;
    return 1 - this.samplingLeft / SIM.samplingSteps;
  }

  /** Full 16-value actor observation in the Python contract's exact order. */
  actorObservation(): Float32Array {
    const targetX = this.targetPad === "BASE" ? this.baseX : this.sampleX;
    const payloadAttached = this.phase === "RETURN" || this.phase === "SUCCESS";
    return buildActorObservation(
      this.state,
      this.terrain,
      targetX,
      this.phase,
      payloadAttached,
    );
  }

  /** One physics frame. `action` is [throttle 0..1, gimbal -1..1]. */
  step(action: readonly [number, number]): void {
    if (this.done) return;
    const a: [number, number] = [
      Math.min(Math.max(action[0], 0), 1),
      Math.min(Math.max(action[1], -1), 1),
    ];
    this.lastAction = a;
    this.steps += 1;

    if (this.phase === "SAMPLING") this.stepSampling();
    else if (this.groundedPad !== null) this.stepSupported(a);
    else this.stepAirborne(a);

    if (!this.done && this.state[FUEL] <= 0) this.fail("OUT_OF_FUEL");
    if (!this.done && this.steps >= SIM.maxSteps) {
      this.outcome = "TIMEOUT";
      this.done = true;
    }
  }

  /** The same shape the replay renderer consumes, so both paths share code. */
  toFrame(): Frame {
    return {
      t: this.steps * SIM.dt,
      state: [...this.state] as StateVec,
      action: [...this.lastAction] as [number, number],
      phase: this.phase,
      payload_mass: this.vehicle.payload?.mass ?? 0,
      payload_fill: this.samplingProgress,
      com_offset_body: [...this.vehicle.comOffsetBody] as Vec2,
    };
  }

  // --- dynamics -----------------------------------------------------------

  private stepSupported(action: [number, number]): void {
    const throttle = action[0];
    const phi = action[1] * SIM.phiMax;
    const available =
      this.state[FUEL] > 0 ? throttle * T_MAX * SIM.thrustMultiplier : 0;
    const vertical = available * Math.cos(this.state[THETA] + phi);
    const canLift = vertical > this.vehicle.totalMass * SIM.g;

    if (canLift) {
      const departure = this.groundedPad;
      this.state = stepDynamics(this.state, action, this.vehicle);
      this.groundedPad = null;
      this.departurePad = departure;
      this.contactArmed = false;
      this.post(
        this.phase === "RETURN" ? "RETURN DEPARTURE" : "BASE DEPARTURE",
      );
    } else {
      // The pad reaction holds the pose, but propellant still burns: an
      // undersized launch command is not free.
      this.state[FUEL] = Math.max(
        0,
        this.state[FUEL] - SIM.burnRate * throttle * SIM.dt,
      );
    }
  }

  private stepAirborne(action: [number, number]): void {
    this.state = stepDynamics(this.state, action, this.vehicle);
    this.updateContactArming();

    if (this.outOfBounds()) {
      this.fail("OUT_OF_BOUNDS");
      return;
    }
    if (!this.bodyContact()) return;

    const bodyX = this.dryCenter()[0];

    // Before arming, a short hop that settles back on the departure pad counts
    // as supported contact rather than a landing attempt.
    if (!this.contactArmed && this.insideDeparturePad()) {
      const pad = this.departurePad!;
      const padX = pad === "BASE" ? this.baseX : this.sampleX;
      const { safe, kind } = landingResult(
        this.state,
        bodyX,
        padX,
        this.vehicle,
      );
      if (!safe) {
        this.fail(this.phaseFailure(kind));
        return;
      }
      this.state = this.restState(pad, this.state[FUEL], this.vehicle);
      this.groundedPad = pad;
      return;
    }

    const contacted = this.contactedPad(bodyX);
    if (contacted === null) {
      this.fail(this.phaseFailure("CRASHED"));
      return;
    }

    const padX = contacted === "BASE" ? this.baseX : this.sampleX;
    const { safe, kind } = landingResult(this.state, bodyX, padX, this.vehicle);
    if (!safe) {
      this.fail(this.phaseFailure(kind));
      return;
    }
    if (contacted !== this.targetPad) {
      this.fail("ABORTED");
      return;
    }

    this.state = this.restState(contacted, this.state[FUEL], this.vehicle);
    this.groundedPad = contacted;
    this.departurePad = contacted;
    this.contactArmed = false;

    if (this.phase === "OUTBOUND") {
      this.phase = "SAMPLING";
      this.samplingLeft = SIM.samplingSteps;
      this.post("SAMPLE PAD TOUCHDOWN");
      this.post("SAMPLE ACQUISITION");
      return;
    }

    this.phase = "SUCCESS";
    this.outcome = "SAMPLE_RETURNED";
    this.post("BASE TOUCHDOWN");
    this.done = true;
  }

  private stepSampling(): void {
    this.samplingLeft -= 1;
    if (this.samplingLeft > 0) return;

    const fuel = this.state[FUEL];
    this.vehicle = this.loaded;
    // Re-assert the exact rest pose once the COM convention switches.
    this.state = this.restState("SAMPLE", fuel, this.vehicle);

    this.phase = "RETURN";
    this.contactArmed = false;
    this.samplingLeft = 0;
    this.groundedPad = "SAMPLE";
    this.departurePad = "SAMPLE";
    this.post("SAMPLE ACQUIRED");
    this.post("PAYLOAD LOCKED");
  }

  // --- geometry -----------------------------------------------------------

  private groundAt(x: number): number {
    return terrainHeightAt(this.terrain, x);
  }

  private restState(pad: PadName, fuel: number, v: VehicleModel): StateVec {
    const x = pad === "BASE" ? this.baseX : this.sampleX;
    const groundY = this.groundAt(x);
    // theta = 0, so the body-frame COM offset is already world-aligned.
    const cx = x + v.comOffsetBody[0];
    const cy = groundY + v.bodyHeight / 2 + v.comOffsetBody[1];
    return [cx, cy, 0, 0, 0, 0, fuel];
  }

  /** World position of the dry body centre for the current state. */
  private dryCenter(): Vec2 {
    const theta = this.state[THETA];
    const [ox, oy] = this.vehicle.comOffsetBody;
    const s = Math.sin(theta);
    const c = Math.cos(theta);
    return [this.state[X] - (c * ox - s * oy), this.state[Y] - (s * ox + c * oy)];
  }

  private endpoints(): [Vec2, Vec2] {
    const center = this.dryCenter();
    const theta = this.state[THETA];
    const nx = -Math.sin(theta);
    const ny = Math.cos(theta);
    const h = this.vehicle.bodyHeight / 2;
    return [
      [center[0] - nx * h, center[1] - ny * h],
      [center[0] + nx * h, center[1] + ny * h],
    ];
  }

  private bodyContact(): boolean {
    const [base, tip] = this.endpoints();
    return base[1] <= this.groundAt(base[0]) || tip[1] <= this.groundAt(tip[0]);
  }

  private updateContactArming(): void {
    if (this.contactArmed || this.departurePad === null) return;
    const [base] = this.endpoints();
    const clearance = base[1] - this.groundAt(base[0]);
    const departureX =
      this.departurePad === "BASE" ? this.baseX : this.sampleX;
    if (
      clearance > SIM.contactArmClearance ||
      Math.abs(this.dryCenter()[0] - departureX) > SIM.departureRadius
    ) {
      this.contactArmed = true;
    }
  }

  private insideDeparturePad(): boolean {
    if (this.departurePad === null) return false;
    const x = this.departurePad === "BASE" ? this.baseX : this.sampleX;
    return Math.abs(this.dryCenter()[0] - x) < SIM.padHalfW;
  }

  private contactedPad(bodyX: number): PadName | null {
    if (Math.abs(bodyX - this.baseX) < SIM.padHalfW) return "BASE";
    if (Math.abs(bodyX - this.sampleX) < SIM.padHalfW) return "SAMPLE";
    return null;
  }

  private outOfBounds(): boolean {
    return (
      this.state[X] < 0 || this.state[X] > SIM.worldW || this.state[Y] > SIM.worldH
    );
  }

  // --- outcomes -----------------------------------------------------------

  private phaseFailure(kind: string): string {
    return `${kind}_${this.phase === "RETURN" ? "RETURN" : "OUTBOUND"}`;
  }

  private fail(outcome: string): void {
    this.phase = "FAILURE";
    this.outcome = outcome;
    this.post(outcome.replace(/_/g, " "));
    this.done = true;
  }

  private post(label: string): void {
    this.events.push({ step: this.steps, t: this.steps * SIM.dt, label });
  }
}
