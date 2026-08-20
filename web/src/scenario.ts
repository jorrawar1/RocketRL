import {
  BODY_HALF_W,
  SCHEMA_VERSION,
  type MissionFixture,
  type Vec2,
} from "./types";

const WORLD_W = 220;
const BASE_X = 28;
const SAMPLE_X = 145;
const CRATER_HALF_W = 62;
const FLOOR_Y = 8;
const RIM_Y = 45;
const OUTER_Y = 30;
const PAD_HALF_W = 5;
const RESOLUTION = 2;
const ROUGHNESS = 0.7;

/** Small deterministic generator: the same uint32 seed gives the same mission. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

function normal(random: () => number): number {
  const u = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

function smoothstep(value: number): number {
  const t = Math.min(Math.max(value, 0), 1);
  return t * t * (3 - 2 * t);
}

function uniqueSorted(values: number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

/** Browser port of the crater silhouette, with independently seeded roughness. */
export function generateTerrain(seed: number): Vec2[] {
  const leftRim = SAMPLE_X - CRATER_HALF_W;
  const rightRim = SAMPLE_X + CRATER_HALF_W;
  const regular = Array.from(
    { length: Math.ceil(WORLD_W / RESOLUTION) },
    (_, i) => i * RESOLUTION,
  );
  const xs = uniqueSorted([
    ...regular,
    0,
    WORLD_W,
    BASE_X - PAD_HALF_W,
    BASE_X,
    BASE_X + PAD_HALF_W,
    leftRim,
    SAMPLE_X - PAD_HALF_W,
    SAMPLE_X,
    SAMPLE_X + PAD_HALF_W,
    rightRim,
  ]);

  const ys = xs.map((x) => {
    if (x < leftRim) {
      const start = leftRim - CRATER_HALF_W / 2;
      return OUTER_Y + (RIM_Y - OUTER_Y) * smoothstep((x - start) / (leftRim - start));
    }
    if (x > rightRim) {
      const end = rightRim + CRATER_HALF_W / 2;
      return OUTER_Y + (RIM_Y - OUTER_Y) * smoothstep((end - x) / (end - rightRim));
    }
    const radius = Math.abs(x - SAMPLE_X) / CRATER_HALF_W;
    return FLOOR_Y + (RIM_Y - FLOOR_Y) * smoothstep(radius);
  });

  const random = mulberry32(seed ^ 0x74657272);
  const raw = xs.map(() => normal(random));
  const padded = [raw[0]!, raw[0]!, ...raw, raw.at(-1)!, raw.at(-1)!];
  const noise = raw.map((_, i) => {
    let total = 0;
    for (let k = 0; k < 5; k++) total += padded[i + k]!;
    return total / 5;
  });
  const peak = Math.max(...noise.map(Math.abs), Number.EPSILON);

  return xs.map((x, i): Vec2 => {
    const onBase = Math.abs(x - BASE_X) <= PAD_HALF_W;
    const onSample = Math.abs(x - SAMPLE_X) <= PAD_HALF_W;
    if (onBase) return [x, OUTER_Y];
    if (onSample) return [x, FLOOR_Y];
    if (x === leftRim || x === rightRim) return [x, RIM_Y];

    const clearance = Math.min(
      Math.abs(x - BASE_X) - PAD_HALF_W,
      Math.abs(x - SAMPLE_X) - PAD_HALF_W,
    );
    const fade = smoothstep(clearance / (2 * RESOLUTION));
    return [x, ys[i]! + (noise[i]! / peak) * ROUGHNESS * fade];
  });
}

/** Generate terrain and a payload inside the policy's trained parameter range. */
export function createScenario(seed: number): MissionFixture {
  const normalizedSeed = seed >>> 0;
  const random = mulberry32(normalizedSeed ^ 0x7061796c);
  // Showcase the harder end of the trained payload envelope without making
  // every mission identical. sqrt(U) biases samples toward the upper bound.
  const payloadMass = 0.25 + Math.sqrt(random()) * 0.1;
  const payloadOffsetX = 0.55 + Math.sqrt(random()) * 0.25;

  return {
    schema_version: SCHEMA_VERSION,
    mission_seed: normalizedSeed,
    terrain: { vertices: generateTerrain(normalizedSeed) },
    base_pad: { x: BASE_X, y: OUTER_Y, half_width: PAD_HALF_W },
    sample_pad: { x: SAMPLE_X, y: FLOOR_Y, half_width: PAD_HALF_W },
    vehicle: {
      dry_mass: 1,
      dry_inertia: 4 / 3,
      body_height: BODY_HALF_W * 8,
      max_thrust: 22.563,
    },
    payload: {
      sample_id: `regolith-${normalizedSeed.toString(16).padStart(8, "0")}`,
      mass: payloadMass,
      offset_body: [payloadOffsetX, 0],
      width: 0.65,
      height: 0.85,
    },
  };
}

export function randomSeed(): number {
  const value = new Uint32Array(1);
  crypto.getRandomValues(value);
  return value[0]!;
}
