/* Port of rocketenv/sample_return/physics.py.
 *
 * Line-for-line, including the order of the semi-implicit Euler updates —
 * velocity before position, and omega before theta. JavaScript numbers are
 * IEEE-754 float64 and the Python side is float64 throughout, so this is
 * expected to agree to near machine precision. test/parity.test.ts checks it
 * against the recorded trajectory.
 */

import type { StateVec, Vec2 } from "../types";
import { SIM } from "./config";
import type { VehicleModel } from "./vehicle";

export function thrustDirection(theta: number, phi: number): Vec2 {
  return [-Math.sin(theta + phi), Math.cos(theta + phi)];
}

/** Torque about the combined COM from thrust applied at the engine. */
export function thrustTorque(
  vehicle: VehicleModel,
  thrust: number,
  phi: number,
): number {
  const [ex, ey] = vehicle.engineOffsetFromCom;
  const fx = -thrust * Math.sin(phi);
  const fy = thrust * Math.cos(phi);
  return ex * fy - ey * fx;
}

const clamp = (v: number, lo: number, hi: number) =>
  Math.min(Math.max(v, lo), hi);

/** Advance one fixed timestep. Pure: returns a new array. */
export function stepDynamics(
  state: StateVec,
  action: readonly [number, number],
  vehicle: VehicleModel,
): StateVec {
  let [x, y, vx, vy, theta, omega, fuel] = state;

  const throttle = clamp(action[0], 0, 1);
  const phi = clamp(action[1], -1, 1) * SIM.phiMax;

  const dryMaxThrust = SIM.twr * vehicle.dryMass * SIM.g;
  const thrust = fuel > 0 ? throttle * dryMaxThrust * SIM.thrustMultiplier : 0;

  const totalMass = vehicle.totalMass;
  const dir = thrustDirection(theta, phi);
  let fx = thrust * dir[0] + SIM.windX;
  let fy = thrust * dir[1] - totalMass * SIM.g;
  if (SIM.dragCoeff > 0) {
    const speed = Math.hypot(vx, vy);
    fx -= SIM.dragCoeff * speed * vx;
    fy -= SIM.dragCoeff * speed * vy;
  }

  const tau = thrustTorque(vehicle, thrust, phi);

  vx += (fx / totalMass) * SIM.dt;
  vy += (fy / totalMass) * SIM.dt;
  x += vx * SIM.dt;
  y += vy * SIM.dt;
  omega += (tau / vehicle.totalInertia) * SIM.dt;
  theta += omega * SIM.dt;

  fuel = Math.max(0, fuel - SIM.burnRate * throttle * SIM.dt);

  return [x, y, vx, vy, theta, omega, fuel];
}
