import { describe, expect, it } from "vitest";

import {
  bodyEndpoints,
  bodyToWorld,
  dryBodyCenter,
  noseDirection,
  payloadCorners,
  plumeDirection,
  terrainHeightAt,
} from "../src/geometry";
import type { Frame, StateVec, Vec2 } from "../src/types";

function frame(over: Partial<Frame> = {}): Frame {
  return {
    t: 0,
    state: [10, 20, 0, 0, 0, 0, 100] as StateVec,
    action: [0, 0],
    phase: "OUTBOUND",
    payload_mass: 0,
    payload_fill: 0,
    com_offset_body: [0, 0],
    ...over,
  };
}

const near = (a: number, b: number) => expect(a).toBeCloseTo(b, 9);

describe("body frame", () => {
  it("points the nose straight up at zero pitch", () => {
    const [nx, ny] = noseDirection(0);
    near(nx, 0);
    near(ny, 1);
  });

  it("rotates body vectors counter-clockwise", () => {
    const [x, y] = bodyToWorld([1, 0], Math.PI / 2);
    near(x, 0);
    near(y, 1);
  });

  it("recovers the dry body centre from the combined COM", () => {
    // At theta = 0 a body-x offset is a world-x offset, so the dry centre
    // sits that far to the -x side of the stored COM.
    const c = dryBodyCenter(frame({ com_offset_body: [0.28, 0] }));
    near(c[0], 9.72);
    near(c[1], 20);
  });

  it("places the endpoints half a body height either side", () => {
    const [base, tip] = bodyEndpoints(frame(), 4);
    near(base[1], 18);
    near(tip[1], 22);
  });
});

describe("plume", () => {
  it("fires straight down with no gimbal and no pitch", () => {
    const [dx, dy] = plumeDirection(frame());
    near(dx, 0);
    near(dy, -1);
  });

  it("deflects with the gimbal command", () => {
    const [dx] = plumeDirection(frame({ action: [1, 1] }));
    expect(dx).toBeGreaterThan(0);
  });
});

describe("payload", () => {
  it("returns four corners around the body offset", () => {
    const corners = payloadCorners(frame(), [0.8, 0], 0.65, 0.85);
    expect(corners).toHaveLength(4);
    const xs = corners.map((c) => c[0]);
    near(Math.min(...xs), 10 + 0.8 - 0.325);
    near(Math.max(...xs), 10 + 0.8 + 0.325);
  });
});

describe("terrain sampling", () => {
  const v: Vec2[] = [
    [0, 10],
    [2, 20],
    [4, 20],
  ];

  it("interpolates between vertices", () => {
    near(terrainHeightAt(v, 1), 15);
    near(terrainHeightAt(v, 3), 20);
  });

  it("clamps outside the span", () => {
    near(terrainHeightAt(v, -5), 10);
    near(terrainHeightAt(v, 99), 20);
  });
});
