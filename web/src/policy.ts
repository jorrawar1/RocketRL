import * as ort from "onnxruntime-web/wasm";

export interface PolicyMetadata {
  schema_version: number;
  checkpoint: string;
  checkpoint_training_mode: string;
  checkpoint_update: number;
  model_sha256: string;
  actor_observation_dim: number;
  observation_names: string[];
  hidden_width: number;
  action_dim: number;
  action_repeat: number;
  action_sampling?: string;
  std_source?: string;
}

export interface PolicyDecision {
  action: [number, number];
  mean: Float32Array;
  std: Float32Array;
  noise: Float32Array;
  latent: Float32Array;
  nextHidden: Float32Array;
}

/** Complete state of the most recently committed recurrent policy decision. */
export interface PolicySnapshot {
  observation: Float32Array;
  previousAction: Float32Array;
  layer1: Float32Array;
  layer2: Float32Array;
  hidden: Float32Array;
  mean: Float32Array;
  std: Float32Array;
  noise: Float32Array;
  latent: Float32Array;
  action: Float32Array;
}

const copyFloat32 = (value: ort.Tensor["data"]): Float32Array => {
  if (!(value instanceof Float32Array)) {
    throw new Error("actor output is not float32");
  }
  return new Float32Array(value);
};

const sha256 = async (bytes: ArrayBuffer): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
};

/** Stateful recurrent Gaussian policy exported from PPO. */
export class ActorPolicy {
  private generation = 0;
  private previousAction: Float32Array;
  private hidden: Float32Array;

  private lastObservation: Float32Array;
  private lastPreviousAction: Float32Array;
  private lastLayer1: Float32Array;
  private lastLayer2: Float32Array;
  private lastHidden: Float32Array;
  private lastMean: Float32Array;
  private lastStd: Float32Array;
  private lastNoise: Float32Array;
  private lastLatent: Float32Array;
  private lastAction: Float32Array;
  private samplingState = 0x6d2b79f5;
  private spareNormal: number | null = null;

  private constructor(
    private session: ort.InferenceSession,
    readonly metadata: PolicyMetadata,
  ) {
    this.previousAction = new Float32Array(metadata.action_dim);
    this.hidden = new Float32Array(metadata.hidden_width);
    this.lastObservation = new Float32Array(metadata.actor_observation_dim);
    this.lastPreviousAction = new Float32Array(metadata.action_dim);
    this.lastLayer1 = new Float32Array(metadata.hidden_width);
    this.lastLayer2 = new Float32Array(metadata.hidden_width);
    this.lastHidden = new Float32Array(metadata.hidden_width);
    this.lastMean = new Float32Array(metadata.action_dim);
    this.lastStd = new Float32Array(metadata.action_dim);
    this.lastNoise = new Float32Array(metadata.action_dim);
    this.lastLatent = new Float32Array(metadata.action_dim);
    this.lastAction = new Float32Array(metadata.action_dim);
  }

  static async load(
    modelUrl = `${import.meta.env.BASE_URL}models/final_actor.onnx`,
    metadataUrl = `${import.meta.env.BASE_URL}models/final_actor.json`,
  ): Promise<ActorPolicy> {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
      mjs: new URL(
        "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
        import.meta.url,
      ).href,
      wasm: new URL(
        "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
        import.meta.url,
      ).href,
    };
    // Metadata is tiny and authoritative. Fetch it first, then version the
    // model URL by its checksum so a deployment cannot pair new code with a
    // stale cached ONNX graph.
    const metadataResponse = await fetch(metadataUrl, { cache: "no-store" });
    if (!metadataResponse.ok) {
      throw new Error(`actor metadata: HTTP ${metadataResponse.status}`);
    }
    const metadata = (await metadataResponse.json()) as PolicyMetadata;
    if (metadata.schema_version !== 1) {
      throw new Error(`unsupported actor schema ${metadata.schema_version}`);
    }
    const separator = modelUrl.includes("?") ? "&" : "?";
    const versionedModelUrl = `${modelUrl}${separator}v=${encodeURIComponent(metadata.model_sha256)}`;
    const modelResponse = await fetch(versionedModelUrl);
    if (!modelResponse.ok) {
      throw new Error(`actor model: HTTP ${modelResponse.status}`);
    }
    const model = await modelResponse.arrayBuffer();
    const digest = await sha256(model);
    if (digest !== metadata.model_sha256) {
      throw new Error("actor model checksum does not match metadata");
    }
    return ActorPolicy.fromModelBytes(new Uint8Array(model), metadata);
  }

  static async fromModelBytes(
    model: Uint8Array,
    metadata: PolicyMetadata,
  ): Promise<ActorPolicy> {
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new ActorPolicy(session, metadata);
  }

  reset(): void {
    this.generation += 1;
    this.previousAction.fill(0);
    this.hidden.fill(0);
    this.lastObservation.fill(0);
    this.lastPreviousAction.fill(0);
    this.lastLayer1.fill(0);
    this.lastLayer2.fill(0);
    this.lastHidden.fill(0);
    this.lastMean.fill(0);
    this.lastStd.fill(0);
    this.lastNoise.fill(0);
    this.lastLatent.fill(0);
    this.lastAction.fill(0);
  }

  /** Seed only the action-noise stream; recurrent memory is reset separately. */
  setSamplingSeed(seed: number): void {
    this.samplingState = (seed >>> 0) || 0x6d2b79f5;
    this.spareNormal = null;
  }

  private uniform(): number {
    let x = this.samplingState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.samplingState = x >>> 0;
    return (this.samplingState + 0.5) / 0x1_0000_0000;
  }

  private normal(): number {
    if (this.spareNormal !== null) {
      const value = this.spareNormal;
      this.spareNormal = null;
      return value;
    }
    const radius = Math.sqrt(-2 * Math.log(this.uniform()));
    const angle = 2 * Math.PI * this.uniform();
    this.spareNormal = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  }

  async decide(
    observation: Float32Array,
    sample = true,
  ): Promise<PolicyDecision> {
    if (observation.length !== this.metadata.actor_observation_dim) {
      throw new Error(
        `actor expected ${this.metadata.actor_observation_dim} observations, got ${observation.length}`,
      );
    }
    const generation = this.generation;
    const feeds: Record<string, ort.Tensor> = {
      observation: new ort.Tensor(
        "float32",
        new Float32Array(observation),
        [1, observation.length],
      ),
      previous_action: new ort.Tensor(
        "float32",
        new Float32Array(this.previousAction),
        [1, 2],
      ),
      hidden: new ort.Tensor(
        "float32",
        new Float32Array(this.hidden),
        [1, this.hidden.length],
      ),
    };
    const output = await this.session.run(feeds);
    const mean = copyFloat32(output.mean!.data);
    if (!output.std) throw new Error("actor model is missing its std output");
    const std = copyFloat32(output.std.data);
    const nextHidden = copyFloat32(output.next_hidden!.data);
    const noise = new Float32Array([
      sample ? this.normal() : 0,
      sample ? this.normal() : 0,
    ]);
    const latent = new Float32Array([
      mean[0]! + std[0]! * noise[0]!,
      mean[1]! + std[1]! * noise[1]!,
    ]);
    const throttleLatent = latent[0]!;
    const gimbalLatent = latent[1]!;
    const throttle = (Math.tanh(throttleLatent) + 1) / 2;
    const gimbal = Math.tanh(gimbalLatent);
    const action: [number, number] = [throttle, gimbal];

    if (generation === this.generation) {
      this.lastPreviousAction = new Float32Array(this.previousAction);
      this.previousAction.set(action);
      this.hidden.set(nextHidden);
      this.lastObservation = new Float32Array(observation);
      this.lastLayer1 = copyFloat32(output.layer1!.data);
      this.lastLayer2 = copyFloat32(output.layer2!.data);
      this.lastHidden = new Float32Array(nextHidden);
      this.lastMean = new Float32Array(mean);
      this.lastStd = new Float32Array(std);
      this.lastNoise = new Float32Array(noise);
      this.lastLatent = new Float32Array(latent);
      this.lastAction = new Float32Array(action);
    }
    return { action, mean, std, noise, latent, nextHidden };
  }

  /** Legacy compact activation list retained for policy parity tests. */
  activations(): Float32Array[] {
    return [
      this.lastObservation,
      this.lastLayer2,
      this.lastHidden,
      this.lastMean,
    ];
  }

  /** Read-only copies are unnecessary: the view never mutates these arrays. */
  snapshot(): PolicySnapshot {
    return {
      observation: this.lastObservation,
      previousAction: this.lastPreviousAction,
      layer1: this.lastLayer1,
      layer2: this.lastLayer2,
      hidden: this.lastHidden,
      mean: this.lastMean,
      std: this.lastStd,
      noise: this.lastNoise,
      latent: this.lastLatent,
      action: this.lastAction,
    };
  }
}
