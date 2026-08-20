/* Vehicle geometry, ported from rocketenv/sample_return/vehicle.py.
 *
 * The state stores the *combined* centre of mass; render and contact geometry
 * belong to the dry body, so every helper recovers the dry-body centre first.
 * These are the first pieces of the physics port — src/sim/ will grow the
 * integrator next, and this module is what it will share.
 */
import { PHI_MAX, THETA, X, Y, type Frame, type Vec2 } from "./types";

/** Body-up (nose) unit vector in the world frame. */
export function noseDirection(theta: number): Vec2 {
  return [-Math.sin(theta), Math.cos(theta)];
}

/** Body-right unit vector in the world frame. */
export function rightDirection(theta: number): Vec2 {
  return [Math.cos(theta), Math.sin(theta)];
}

/** Rotate a body-frame vector into the y-up world frame. */
export function bodyToWorld(v: Vec2, theta: number): Vec2 {
  const s = Math.sin(theta);
  const c = Math.cos(theta);
  return [c * v[0] - s * v[1], s * v[0] + c * v[1]];
}

export function dryBodyCenter(frame: Frame): Vec2 {
  const theta = frame.state[THETA];
  const off = bodyToWorld(frame.com_offset_body, theta);
  return [frame.state[X] - off[0], frame.state[Y] - off[1]];
}

/** `[base, tip]` world positions of the rocket segment. */
export function bodyEndpoints(frame: Frame, bodyHeight: number): [Vec2, Vec2] {
  const c = dryBodyCenter(frame);
  const [nx, ny] = noseDirection(frame.state[THETA]);
  const h = bodyHeight / 2;
  return [
    [c[0] - nx * h, c[1] - ny * h],
    [c[0] + nx * h, c[1] + ny * h],
  ];
}

/** Thrust application point: the engine sits at body `(0, -H/2)`. */
export function enginePosition(frame: Frame, bodyHeight: number): Vec2 {
  const c = dryBodyCenter(frame);
  const e = bodyToWorld([0, -bodyHeight / 2], frame.state[THETA]);
  return [c[0] + e[0], c[1] + e[1]];
}

/** Unit vector the exhaust travels along (opposite the thrust force). */
export function plumeDirection(frame: Frame): Vec2 {
  const a = frame.state[THETA] + frame.action[1] * PHI_MAX;
  return [Math.sin(a), -Math.cos(a)];
}

/**
 * Point on the hull, addressed the way the pygame renderer does it:
 * `along` measured up from the base, `across` measured to body-right.
 */
export function bodyPoint(
  frame: Frame,
  bodyHeight: number,
  along: number,
  across: number,
): Vec2 {
  const [base] = bodyEndpoints(frame, bodyHeight);
  const theta = frame.state[THETA];
  const n = noseDirection(theta);
  const r = rightDirection(theta);
  return [
    base[0] + n[0] * along + r[0] * across,
    base[1] + n[1] * along + r[1] * across,
  ];
}

/** Payload rectangle corners, body order BL, BR, TR, TL. */
export function payloadCorners(
  frame: Frame,
  offsetBody: Vec2,
  width: number,
  height: number,
): Vec2[] {
  const c = dryBodyCenter(frame);
  const theta = frame.state[THETA];
  const hw = width / 2;
  const hh = height / 2;
  const local: Vec2[] = [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
  ];
  return local.map((p) => {
    const w = bodyToWorld([offsetBody[0] + p[0], offsetBody[1] + p[1]], theta);
    return [c[0] + w[0], c[1] + w[1]] as Vec2;
  });
}

/** Linear interpolation of the terrain polyline; clamps outside the span. */
export function terrainHeightAt(vertices: Vec2[], x: number): number {
  if (vertices.length === 0) return 0;
  const first = vertices[0]!;
  const last = vertices[vertices.length - 1]!;
  if (x <= first[0]) return first[1];
  if (x >= last[0]) return last[1];
  let lo = 0;
  let hi = vertices.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (vertices[mid]![0] <= x) lo = mid;
    else hi = mid;
  }
  const a = vertices[lo]!;
  const b = vertices[hi]!;
  const span = b[0] - a[0];
  if (span === 0) return a[1];
  return a[1] + ((x - a[0]) / span) * (b[1] - a[1]);
}

/** Exact ray-versus-polyline distance, matching PolylineTerrain.ray_distance. */
export function terrainRayDistance(
  vertices: Vec2[],
  ox: number,
  oy: number,
  dx: number,
  dy: number,
  maxRange: number,
): number {
  if (oy <= terrainHeightAt(vertices, ox)) return 0;

  let nearest = maxRange;
  for (let i = 0; i + 1 < vertices.length; i++) {
    const p = vertices[i]!;
    const next = vertices[i + 1]!;
    const sx = next[0] - p[0];
    const sy = next[1] - p[1];
    const denom = dx * sy - dy * sx;
    if (Math.abs(denom) <= 1e-12) continue;

    const qpx = p[0] - ox;
    const qpy = p[1] - oy;
    const t = (qpx * sy - qpy * sx) / denom;
    const u = (qpx * dy - qpy * dx) / denom;
    if (t > 1e-9 && u >= 0 && u <= 1 && t < nearest) nearest = t;
  }
  return nearest;
}
