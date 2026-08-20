import "@fontsource-variable/martian-mono";
import "@fontsource-variable/jetbrains-mono";
import "./style.css";

import { Camera, contentBounds } from "./camera";
import { Panels } from "./panels";
import { ActorPolicy } from "./policy";
import { ActivationView } from "./render/activations";
import { WorldRenderer } from "./render/world";
import { createScenario, randomSeed } from "./scenario";
import { SIM } from "./sim/config";
import { Mission } from "./sim/mission";
import { type Frame, type MissionFixture, type Vec2 } from "./types";

/** Seconds to dwell on the final frame before the same mission runs again. */
const LOOP_HOLD = 0.5;
const INITIAL_SEED = 42;

/** Control ramps, in units per second. Instant input makes the gimbal twitchy. */
const THROTTLE_UP = 2.6;
const THROTTLE_DOWN = 3.4;
/** Keyboard steering is deliberately gentle: ±3.5° rather than full ±10°. */
const HUMAN_GIMBAL_LIMIT = 0.35;
const GIMBAL_IN = 0.85;
const GIMBAL_OUT = 1.8;

type Mode = "FLY" | "POLICY";

function el<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

function boot(policy: ActorPolicy): void {
  const activations = new ActivationView(
    el<HTMLCanvasElement>("activations"),
    el("policy-inputs"),
    el("policy-actions"),
  );
  activations.setSource(() => policy.snapshot());

  let fx: MissionFixture;
  let camera: Camera;
  let world: WorldRenderer;
  let panels: Panels;
  let mission: Mission;
  let mode: Mode = "POLICY";
  let carry = 0;
  let throttle = 0;
  let gimbal = 0;
  let holdLeft = 0;
  let snapCamera = false;
  let policyAction: [number, number] = [0, 0];
  let policyFramesLeft = 0;
  let policyPending = false;
  let policyGeneration = 0;
  let policyFault: string | null = null;
  let lastTick = performance.now();

  const held = new Set<string>();
  const statusLine = el("statusline");
  const statusLeft = el("status-left");
  const foot = el("hud-foot");
  const packet = el("packet");
  const packetBody = el("packet-body");
  const net = el("net");
  const policyButton = el<HTMLButtonElement>("control-policy");
  const humanButton = el<HTMLButtonElement>("control-human");
  const netButton = el<HTMLButtonElement>("control-net");
  const setButtonState = (button: HTMLButtonElement, active: boolean) => {
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  };

  const syncControls = () => {
    setButtonState(policyButton, mode === "POLICY");
    setButtonState(humanButton, mode === "FLY");
    setButtonState(netButton, !net.hidden);
  };

  const clearController = () => {
    throttle = 0;
    gimbal = 0;
    carry = 0;
    holdLeft = 0;
    policyAction = [0, 0];
    policyFramesLeft = 0;
    policyPending = false;
    policyFault = null;
    policyGeneration += 1;
    policy.reset();
  };

  const resetLive = () => {
    mission.reset();
    policy.setSamplingSeed(fx.mission_seed ^ 0x6163746e);
    clearController();
    panels.setOutcome(null);
    snapCamera = true;
  };

  const installScenario = (seed: number) => {
    fx = createScenario(seed);
    // The full mission remains visible; the upper point leaves room for arcs.
    const framingPath: Vec2[] = [[0, 0], [220, 75]];
    camera = new Camera(contentBounds(fx.terrain.vertices, framingPath));
    world = new WorldRenderer(el<HTMLCanvasElement>("world"), fx, camera);
    panels = new Panels(fx);
    mission = new Mission(fx);
    policy.setSamplingSeed(fx.mission_seed ^ 0x6163746e);
    clearController();
    panels.setPilot(mode === "POLICY" ? "PPO GRU" : "HUMAN");
    panels.setOutcome(null);
    statusLeft.textContent = "";
    statusLine.hidden = true;
    snapCamera = true;
  };

  const setMode = (next: Mode) => {
    mode = next;
    el("hdr-mode").textContent = next;
    resetLive();
    panels.setPilot(next === "POLICY" ? "PPO GRU" : "HUMAN");
    syncControls();
  };

  const useNewSeed = () => installScenario(randomSeed());
  const toggleNetwork = () => {
    net.hidden = !net.hidden;
    syncControls();
  };

  // --- input --------------------------------------------------------------

  window.addEventListener("keydown", (ev) => {
    held.add(ev.key.toLowerCase());
    switch (ev.key.toLowerCase()) {
      case "f":
        setMode("FLY");
        break;
      case "p":
        setMode("POLICY");
        break;
      case "s":
        useNewSeed();
        break;
      case "n":
        toggleNetwork();
        break;
      case "`":
        packet.hidden = !packet.hidden;
        break;
      case "r":
        resetLive();
        break;
    }
  });
  window.addEventListener("keyup", (ev) => held.delete(ev.key.toLowerCase()));
  window.addEventListener("blur", () => held.clear());
  el<HTMLButtonElement>("seed-control").addEventListener("click", useNewSeed);
  policyButton.addEventListener("click", () => setMode("POLICY"));
  humanButton.addEventListener("click", () => setMode("FLY"));
  el<HTMLButtonElement>("control-restart").addEventListener("click", resetLive);
  el<HTMLButtonElement>("control-seed").addEventListener("click", useNewSeed);
  netButton.addEventListener("click", toggleNetwork);
  const approach = (value: number, target: number, rate: number, dt: number) => {
    const step = rate * dt;
    if (value < target) return Math.min(value + step, target);
    return Math.max(value - step, target);
  };

  /** Ramped stick, so a keyboard does not produce square-wave commands. */
  const pilotAction = (dt: number): [number, number] => {
    const up = held.has("w") || held.has("arrowup");
    throttle = approach(throttle, up ? 1 : 0, up ? THROTTLE_UP : THROTTLE_DOWN, dt);

    const left = held.has("a") || held.has("arrowleft");
    const right = held.has("d") || held.has("arrowright");
    // Positive gimbal is screen-right in the human control convention.
    const target = left && !right
      ? -HUMAN_GIMBAL_LIMIT
      : right && !left
        ? HUMAN_GIMBAL_LIMIT
        : 0;
    gimbal = approach(gimbal, target, target === 0 ? GIMBAL_OUT : GIMBAL_IN, dt);
    return [throttle, gimbal];
  };

  const requestPolicyDecision = () => {
    if (policyPending || policyFramesLeft > 0 || policyFault) return;
    policyPending = true;
    const generation = policyGeneration;
    void policy
      .decide(mission.actorObservation())
      .then(({ action }) => {
        if (mode !== "POLICY" || generation !== policyGeneration) return;
        policyAction = action;
        policyFramesLeft = policy.metadata.action_repeat;
        policyPending = false;
      })
      .catch((error: unknown) => {
        if (generation !== policyGeneration) return;
        policyPending = false;
        policyFault = error instanceof Error ? error.message : String(error);
        console.error(error);
      });
  };

  // --- loop ---------------------------------------------------------------

  const tick = (now: number) => {
    const elapsed = Math.min((now - lastTick) / 1000, 0.1);
    lastTick = now;

    if (mode === "FLY") {
      const action = pilotAction(elapsed);
      carry = Math.min(carry + elapsed, SIM.dt * 8);
      while (carry >= SIM.dt && !mission.done) {
        mission.step(action);
        carry -= SIM.dt;
      }
    } else {
      carry = Math.min(carry + elapsed, SIM.dt * 8);
      while (carry >= SIM.dt && !mission.done) {
        // Unlike training, the demo renders the complete acquisition dwell.
        // The policy does not act during it; memory resets only at the exact
        // transition to the loaded return leg.
        if (mission.phase === "SAMPLING") {
          mission.step([0, 0]);
          carry -= SIM.dt;
          if (mission.phase !== "SAMPLING") {
            policy.reset();
            policyFramesLeft = 0;
            policyAction = [0, 0];
            policyGeneration += 1;
          }
          continue;
        }
        if (policyFramesLeft === 0) {
          requestPolicyDecision();
          break;
        }
        mission.step(policyAction);
        policyFramesLeft -= 1;
        carry -= SIM.dt;
      }
    }

    if (mission.done) {
      holdLeft = holdLeft > 0 ? holdLeft - elapsed : LOOP_HOLD;
      if (holdLeft <= 0) installScenario(randomSeed());
    }

    const frame: Frame = mission.toFrame();
    panels.setOutcome(mission.outcome);
    statusLine.hidden = !policyFault;
    statusLeft.textContent = policyFault ? `POLICY FAULT · ${policyFault.toUpperCase()}` : "";

    const frameRect = foot.getBoundingClientRect();
    const viewport: { left?: number; right?: number; bottom?: number } = {
      left: Math.max(0, frameRect.left),
      right: Math.max(0, window.innerWidth - frameRect.right),
    };
    if (!net.hidden) {
      const rect = net.getBoundingClientRect();
      viewport.bottom = Math.max(0, window.innerHeight - rect.top + 8);
    }
    world.setHighlightedRay(activations.highlightedRay);
    world.draw(frame, !snapCamera, viewport);
    snapCamera = false;
    panels.update(frame);
    panels.renderFeed(mission.events, frame.t);
    if (!net.hidden) activations.draw(mission.steps, frame);
    if (!packet.hidden) packetBody.textContent = JSON.stringify(frame, null, 1);

    requestAnimationFrame(tick);
  };

  installScenario(INITIAL_SEED);
  setMode("POLICY");
  requestAnimationFrame(tick);
}

ActorPolicy.load()
  .then((policy) => boot(policy))
  .catch((err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    const status = el("status-left");
    el("statusline").hidden = false;
    status.textContent = `FAULT · ${message.toUpperCase()}`;
    status.style.color = "var(--red)";
    console.error(err);
  });
