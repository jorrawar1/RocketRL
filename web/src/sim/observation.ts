import { terrainHeightAt, terrainRayDistance } from "../geometry";
import {
  FUEL,
  OMEGA,
  THETA,
  VX,
  VY,
  X,
  Y,
  type MissionPhase,
  type StateVec,
  type Vec2,
} from "../types";
import { SIM } from "./config";

/** Construct the full 16-value actor observation in Python's exact order. */
export function buildActorObservation(
  state: StateVec,
  terrain: Vec2[],
  targetX: number,
  phase: MissionPhase,
  payloadAttached: boolean,
): Float32Array {
  const targetY = terrainHeightAt(terrain, targetX);
  const rays: number[] = [];
  for (let i = 0; i < SIM.rayCount; i++) {
    const mix = i / (SIM.rayCount - 1);
    const angle = SIM.rayAngleLo + mix * (SIM.rayAngleHi - SIM.rayAngleLo);
    rays.push(
      terrainRayDistance(
        terrain,
        state[X],
        state[Y],
        Math.cos(angle),
        Math.sin(angle),
        SIM.rayMaxRange,
      ) / SIM.rayMaxRange,
    );
  }
  return new Float32Array([
    (targetX - state[X]) / SIM.worldW,
    (targetY - state[Y]) / SIM.worldH,
    state[VX] / SIM.vRef,
    state[VY] / SIM.vRef,
    Math.sin(state[THETA]),
    Math.cos(state[THETA]),
    state[OMEGA] / SIM.omegaRef,
    state[FUEL] / SIM.fuel0,
    ...rays,
    payloadAttached ? 1 : 0,
    phase === "OUTBOUND" ? 1 : 0,
    phase === "RETURN" ? 1 : 0,
  ]);
}
