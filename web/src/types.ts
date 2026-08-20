/** Mirror of rocketenv/sample_return/serialization.py, schema_version 1. */

export const SCHEMA_VERSION = 1;

/** State layout: rocketenv.physics X, Y, VX, VY, THETA, OMEGA, FUEL = range(7) */
export const X = 0, Y = 1, VX = 2, VY = 3, THETA = 4, OMEGA = 5, FUEL = 6;

export type Vec2 = [number, number];
export type StateVec = [number, number, number, number, number, number, number];

export type MissionPhase =
  | "OUTBOUND"
  | "SAMPLING"
  | "RETURN"
  | "SUCCESS"
  | "FAILURE";

export interface Frame {
  t: number;
  state: StateVec;
  /** [throttle 0..1, gimbal -1..1] */
  action: [number, number];
  phase: MissionPhase;
  payload_mass: number;
  /** 0..1 progress through the sampling dwell */
  payload_fill: number;
  com_offset_body: Vec2;
}

export interface Pad {
  x: number;
  y: number;
  half_width: number;
}

export interface MissionEvent {
  label: string;
  step: number;
  t: number;
}

/** Static mission definition consumed by the live browser simulator. */
export interface MissionFixture {
  schema_version: number;
  mission_seed: number;
  terrain: { vertices: Vec2[] };
  base_pad: Pad;
  sample_pad: Pad;
  vehicle: {
    dry_mass: number;
    dry_inertia: number;
    body_height: number;
    max_thrust: number;
  };
  payload: {
    sample_id: string;
    mass: number;
    offset_body: Vec2;
    width: number;
    height: number;
  };
}

/** Recorded fixture retained as the Python/TypeScript physics parity format. */
export interface Fixture extends MissionFixture {
  controller: string;
  mission_events: MissionEvent[];
  frames: Frame[];
  outcome: string;
}

/** Constants the fixture does not carry, taken from rocketenv Config. */
export const DT = 1 / 60;
export const PHI_MAX = (10 * Math.PI) / 180;
export const WORLD_W = 220;
export const WORLD_H = 140;
/** Hull half-width and landing-leg span, matching the pygame silhouette. */
export const BODY_HALF_W = 0.5;
export const LEG_HALF_W = 1.5;
