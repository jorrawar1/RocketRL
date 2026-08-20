import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ActorPolicy, type PolicyMetadata } from "../src/policy";
import { createScenario } from "../src/scenario";
import { Mission } from "../src/sim/mission";
import { buildActorObservation } from "../src/sim/observation";
import type { Fixture, MissionPhase, StateVec, Vec2 } from "../src/types";

interface ReferenceStep {
  observation: number[];
  mean: number[];
  std: number[];
  action: number[];
  next_hidden: number[];
  layer1: number[];
  layer2: number[];
}

interface PolicyReference {
  schema_version: number;
  model_sha256: string;
  absolute_tolerance: number;
  steps: ReferenceStep[];
  observation_contract: {
    terrain: Vec2[];
    cases: Array<{
      state: StateVec;
      target_x: number;
      phase: MissionPhase;
      payload_attached: boolean;
      expected: number[];
    }>;
  };
}

const MODEL = fileURLToPath(
  new URL("../public/models/final_actor.onnx", import.meta.url),
);
const METADATA = fileURLToPath(
  new URL("../public/models/final_actor.json", import.meta.url),
);
const REFERENCE = fileURLToPath(
  new URL("./data/policy_reference.json", import.meta.url),
);
const MISSION = fileURLToPath(
  new URL("../../artifacts/mission_42.json", import.meta.url),
);

const maxError = (actual: ArrayLike<number>, expected: number[]): number => {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    worst = Math.max(worst, Math.abs(actual[i]! - expected[i]!));
  }
  return worst;
};

describe("exported recurrent actor", () => {
  it("matches PyTorch through a recurrent decision sequence", async () => {
    const model = new Uint8Array(readFileSync(MODEL));
    const metadata = JSON.parse(
      readFileSync(METADATA, "utf8"),
    ) as PolicyMetadata;
    const reference = JSON.parse(
      readFileSync(REFERENCE, "utf8"),
    ) as PolicyReference;
    expect(reference.schema_version).toBe(1);
    expect(reference.model_sha256).toBe(metadata.model_sha256);

    const policy = await ActorPolicy.fromModelBytes(model, metadata);
    policy.reset();
    let worst = 0;
    for (const step of reference.steps) {
      const decision = await policy.decide(
        new Float32Array(step.observation),
        false,
      );
      const activations = policy.activations();
      const snapshot = policy.snapshot();
      worst = Math.max(
        worst,
        maxError(decision.mean, step.mean),
        maxError(decision.std, step.std),
        maxError(decision.action, step.action),
        maxError(decision.nextHidden, step.next_hidden),
        maxError(snapshot.layer1, step.layer1),
        maxError(activations[1]!, step.layer2),
        maxError(activations[2]!, step.next_hidden),
      );
    }
    expect(reference.steps.length).toBeGreaterThanOrEqual(20);
    expect(worst).toBeLessThan(reference.absolute_tolerance);
  });

  it("constructs the same observations as Python", () => {
    const reference = JSON.parse(
      readFileSync(REFERENCE, "utf8"),
    ) as PolicyReference;
    let worst = 0;
    for (const sample of reference.observation_contract.cases) {
      const observation = buildActorObservation(
        sample.state,
        reference.observation_contract.terrain,
        sample.target_x,
        sample.phase,
        sample.payload_attached,
      );
      worst = Math.max(worst, maxError(observation, sample.expected));
    }
    expect(reference.observation_contract.cases.length).toBeGreaterThanOrEqual(5);
    expect(worst).toBeLessThan(2e-6);
  });

  it("completes a browser-simulated full mission", async () => {
    const model = new Uint8Array(readFileSync(MODEL));
    const metadata = JSON.parse(
      readFileSync(METADATA, "utf8"),
    ) as PolicyMetadata;
    const fixture = JSON.parse(readFileSync(MISSION, "utf8")) as Fixture;
    const policy = await ActorPolicy.fromModelBytes(model, metadata);
    const mission = new Mission(fixture);

    let decisions = 0;
    while (!mission.done && decisions < 2_400) {
      const { action } = await policy.decide(mission.actorObservation(), false);
      decisions += 1;
      for (let frame = 0; frame < metadata.action_repeat && !mission.done; frame++) {
        mission.step(action);
        if (mission.phase === "SAMPLING") {
          while (mission.phase === "SAMPLING" && !mission.done) {
            mission.step([0, 0]);
          }
          policy.reset();
          break;
        }
      }
    }

    expect(mission.outcome).toBe("SAMPLE_RETURNED");
    expect(decisions).toBeLessThan(600);
  });

  it("runs the learned policy on newly generated terrain and payload", async () => {
    const model = new Uint8Array(readFileSync(MODEL));
    const metadata = JSON.parse(
      readFileSync(METADATA, "utf8"),
    ) as PolicyMetadata;
    const policy = await ActorPolicy.fromModelBytes(model, metadata);
    const mission = new Mission(createScenario(42));

    let decisions = 0;
    while (!mission.done && decisions < 2_400) {
      const { action } = await policy.decide(mission.actorObservation(), false);
      decisions += 1;
      for (let frame = 0; frame < metadata.action_repeat && !mission.done; frame++) {
        mission.step(action);
        if (mission.phase === "SAMPLING") {
          while (mission.phase === "SAMPLING" && !mission.done) {
            mission.step([0, 0]);
          }
          policy.reset();
          break;
        }
      }
    }

    expect(mission.outcome).toBe("SAMPLE_RETURNED");
  });

  it("samples reproducibly from raw learned action deviations", async () => {
    const model = new Uint8Array(readFileSync(MODEL));
    const metadata = JSON.parse(
      readFileSync(METADATA, "utf8"),
    ) as PolicyMetadata;
    const reference = JSON.parse(
      readFileSync(REFERENCE, "utf8"),
    ) as PolicyReference;
    const policy = await ActorPolicy.fromModelBytes(model, metadata);
    const observation = new Float32Array(reference.steps[0]!.observation);

    policy.reset();
    policy.setSamplingSeed(12345);
    const first = await policy.decide(observation);
    policy.reset();
    policy.setSamplingSeed(12345);
    const replay = await policy.decide(observation);

    expect(replay.action).toEqual(first.action);
    expect(first.std[0]).toBeGreaterThan(0);
    expect(first.std[1]).toBeGreaterThan(0);
    expect(first.std[1]).toBeLessThan(0.30);
    expect(first.action).not.toEqual([
      (Math.tanh(first.mean[0]!) + 1) / 2,
      Math.tanh(first.mean[1]!),
    ]);
    expect(first.latent[0]).toBeCloseTo(
      first.mean[0]! + first.std[0]! * first.noise[0]!,
      6,
    );
    expect(first.latent[1]).toBeCloseTo(
      first.mean[1]! + first.std[1]! * first.noise[1]!,
      6,
    );
    expect(policy.snapshot().action).toEqual(new Float32Array(first.action));
  });
});
