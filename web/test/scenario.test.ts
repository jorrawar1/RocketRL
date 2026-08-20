import { describe, expect, it } from "vitest";

import { createScenario, generateTerrain } from "../src/scenario";

describe("seeded browser scenarios", () => {
  it("replays the same terrain and payload for the same seed", () => {
    expect(createScenario(42)).toEqual(createScenario(42));
  });

  it("changes both terrain and payload when the seed changes", () => {
    const a = createScenario(42);
    const b = createScenario(43);
    expect(a.terrain.vertices).not.toEqual(b.terrain.vertices);
    expect(a.payload).not.toEqual(b.payload);
  });

  it("keeps pads exact and payloads inside the trained envelope", () => {
    for (const seed of [0, 1, 42, 0xffff_ffff]) {
      const scenario = createScenario(seed);
      const terrain = generateTerrain(seed);
      const at = (x: number) => terrain.find(([tx]) => tx === x)?.[1];
      expect(at(28)).toBe(30);
      expect(at(145)).toBe(8);
      expect(scenario.payload.mass).toBeGreaterThanOrEqual(0.25);
      expect(scenario.payload.mass).toBeLessThanOrEqual(0.35);
      expect(scenario.payload.offset_body[0]).toBeGreaterThanOrEqual(0.55);
      expect(scenario.payload.offset_body[0]).toBeLessThanOrEqual(0.8);
    }
  });
});
