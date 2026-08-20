import { SIM } from "../sim/config";
import { alpha, paletteFor, theme, type Aesthetic } from "../theme";
import type { PolicySnapshot } from "../policy";
import type { Frame } from "../types";
import { fitCanvas } from "./canvas";

interface ObservationDescriptor {
  label: string;
  description: string;
  format: (value: number) => string;
}

const signed = (value: number, digits = 2): string =>
  `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;

const OBSERVATIONS: ObservationDescriptor[] = [
  { label: "Target horizontal", description: "Horizontal displacement from the active landing pad.", format: (v) => `${signed(v * SIM.worldW, 1)} m` },
  { label: "Target vertical", description: "Vertical displacement from the active landing pad.", format: (v) => `${signed(v * SIM.worldH, 1)} m` },
  { label: "Velocity horizontal", description: "World-frame horizontal velocity; positive points right.", format: (v) => `${signed(v * SIM.vRef, 1)} m/s` },
  { label: "Velocity vertical", description: "World-frame vertical velocity; positive points up.", format: (v) => `${signed(v * SIM.vRef, 1)} m/s` },
  { label: "Pitch sine", description: "Sine of the rocket pitch, continuous through angle wrapping.", format: (v) => signed(v, 3) },
  { label: "Pitch cosine", description: "Cosine of the rocket pitch, paired with pitch sine.", format: (v) => signed(v, 3) },
  { label: "Angular rate", description: "Rocket angular velocity.", format: (v) => `${signed(v * SIM.omegaRef, 2)} rad/s` },
  { label: "Fuel remaining", description: "Remaining propellant as a fraction of initial fuel.", format: (v) => `${(v * 100).toFixed(1)}%` },
  ...[-150, -120, -90, -60, -30].map((angle): ObservationDescriptor => ({
    label: `Terrain ray ${angle}°`,
    description: `Terrain clearance along the ${angle} degree sensor ray.`,
    format: (v) => `${(v * SIM.rayMaxRange).toFixed(1)} m`,
  })),
  { label: "Payload attached", description: "One when the sample payload is attached to the vehicle.", format: (v) => (v >= 0.5 ? "YES" : "NO") },
  { label: "Outbound phase", description: "One while travelling from base to the sample site.", format: (v) => (v >= 0.5 ? "ACTIVE" : "OFF") },
  { label: "Return phase", description: "One while returning the payload to base.", format: (v) => (v >= 0.5 ? "ACTIVE" : "OFF") },
];

const PREVIOUS_ACTIONS: ObservationDescriptor[] = [
  { label: "Previous throttle", description: "Bounded throttle command issued at the previous policy decision.", format: (v) => `${(v * 100).toFixed(1)}%` },
  { label: "Previous gimbal", description: "Bounded gimbal command issued at the previous policy decision.", format: (v) => `${signed(v * SIM.phiMax * 180 / Math.PI, 2)}°` },
];

interface LayerLayout {
  name: string;
  note: string;
  values: Float32Array;
  x: number;
  y: number;
  width: number;
  height: number;
  points: Array<[number, number]>;
}

export type ActivationSource = (index: number, frame: Frame) => PolicySnapshot | null;

/** Live, exact policy state plus a compact architectural flow diagram. */
export class ActivationView {
  private ctx: CanvasRenderingContext2D;
  private source: ActivationSource | null = null;
  private inputRows: HTMLElement[] = [];
  private aesthetic: Aesthetic = "minimal";
  private hoveredObservation = -1;

  constructor(
    private canvas: HTMLCanvasElement,
    private inputRoot: HTMLElement,
    private actionRoot: HTMLElement,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    this.ctx = ctx;
    this.buildInputs();
  }

  setSource(source: ActivationSource | null): void { this.source = source; }
  setAesthetic(aesthetic: Aesthetic): void { this.aesthetic = aesthetic; }
  get bound(): boolean { return this.source !== null; }
  get highlightedRay(): number | null {
    return this.hoveredObservation >= 8 && this.hoveredObservation <= 12
      ? this.hoveredObservation - 8
      : null;
  }

  private buildInputs(): void {
    for (const [index, descriptor] of [...OBSERVATIONS, ...PREVIOUS_ACTIONS].entries()) {
      const row = document.createElement("div");
      row.className = "policy-input";
      row.title = descriptor.description;
      row.innerHTML = `
        <span class="policy-input__signal" aria-hidden="true"></span>
        <span class="policy-input__label">${descriptor.label}</span>
        <span class="policy-input__value">—</span>
      `;
      row.addEventListener("mouseenter", () => { this.hoveredObservation = index; });
      row.addEventListener("mouseleave", () => { this.hoveredObservation = -1; });
      this.inputRoot.append(row);
      this.inputRows.push(row);
    }
  }

  private updateInputs(snapshot: PolicySnapshot): void {
    const palette = paletteFor(this.aesthetic);
    const values = [...snapshot.observation, ...snapshot.previousAction];
    const descriptors = [...OBSERVATIONS, ...PREVIOUS_ACTIONS];
    for (let index = 0; index < descriptors.length; index++) {
      const row = this.inputRows[index]!;
      const value = values[index] ?? 0;
      const dot = row.querySelector<HTMLElement>(".policy-input__signal")!;
      const output = row.querySelector<HTMLElement>(".policy-input__value")!;
      const magnitude = Math.min(Math.abs(value), 1);
      const color = value >= 0 ? palette.amber : palette.blue;
      dot.style.background = color;
      dot.style.opacity = String(0.22 + 0.78 * magnitude);
      dot.style.boxShadow = `0 0 ${2 + magnitude * 7}px ${alpha(color, 0.15 + magnitude * 0.4)}`;
      output.textContent = descriptors[index]!.format(value);
    }
  }

  private updateAction(name: "throttle" | "gimbal", actionIndex: number, snapshot: PolicySnapshot): void {
    const mean = snapshot.mean[actionIndex] ?? 0;
    const std = snapshot.std[actionIndex] ?? 0;
    const noise = snapshot.noise[actionIndex] ?? 0;
    const latent = snapshot.latent[actionIndex] ?? 0;
    const action = snapshot.action[actionIndex] ?? 0;
    const card = this.actionRoot.querySelector<HTMLElement>(`[data-action="${name}"]`)!;
    const set = (field: string, value: string) => {
      card.querySelector<HTMLElement>(`[data-field="${field}"]`)!.textContent = value;
    };
    set("mean", signed(mean, 3));
    set("std", std.toFixed(3));
    set("noise", signed(noise, 3));
    set("latent", signed(latent, 3));
    set("command", name === "throttle"
      ? `${(action * 100).toFixed(1)}%`
      : `${signed(action * SIM.phiMax * 180 / Math.PI, 2)}°`);

    // Both heads share a fixed latent scale. A stable axis makes movement of
    // the mean and sample comparable across decisions and between controls.
    const range = 4;
    const toPercent = (value: number) => `${Math.max(0, Math.min(100, 50 + value / range * 50))}%`;
    const band = card.querySelector<HTMLElement>(".policy-gaussian__band")!;
    const bandLeft = Math.max(-range, mean - std);
    const bandRight = Math.min(range, mean + std);
    band.style.left = toPercent(bandLeft);
    band.style.width = `${Math.max(0, (bandRight - bandLeft) / (2 * range) * 100)}%`;
    card.querySelector<HTMLElement>(".policy-gaussian__mean")!.style.left = toPercent(mean);
    card.querySelector<HTMLElement>(".policy-gaussian__sample")!.style.left = toPercent(latent);
  }

  private layerLayout(name: string, note: string, values: Float32Array, x: number, y: number, width: number, height: number): LayerLayout {
    const rows = 16;
    const cols = 8;
    const cellX = width / cols;
    const cellY = height / rows;
    const points: Array<[number, number]> = [];
    for (let unit = 0; unit < values.length; unit++) {
      const col = Math.floor(unit / rows);
      const row = unit % rows;
      points.push([x + (col + 0.5) * cellX, y + (row + 0.5) * cellY]);
    }
    return { name, note, values, x, y, width, height, points };
  }

  private drawFlow(a: LayerLayout, b: LayerLayout, now: number, flowOffset: number): void {
    const { ctx } = this;
    const palette = paletteFor(this.aesthetic);
    const strongest = (layer: LayerLayout, offset: number): number[] =>
      Array.from({ length: 16 }, (_, row) => offset + row)
        .sort((left, right) => Math.abs(layer.values[right] ?? 0) - Math.abs(layer.values[left] ?? 0))
        .slice(0, 12);
    const sources = strongest(a, 7 * 16);
    const targets = strongest(b, 0);
    for (let path = 0; path < sources.length; path++) {
      const from = sources[path]!;
      const to = targets[(path * 5 + flowOffset) % targets.length]!;
      const p = a.points[from]!;
      const q = b.points[to]!;
      const sourceValue = a.values[from] ?? 0;
      const targetValue = b.values[to] ?? 0;
      const magnitude = Math.min(1, (Math.abs(sourceValue) + Math.abs(targetValue)) / 2);
      const sourceColor = sourceValue >= 0 ? palette.amber : palette.blue;
      const targetColor = targetValue >= 0 ? palette.amber : palette.blue;
      const sourceX = p[0] + 2;
      const targetX = q[0] - 2;
      const controlA: [number, number] = [p[0] + (q[0] - p[0]) * 0.42, p[1]];
      const controlB: [number, number] = [p[0] + (q[0] - p[0]) * 0.58, q[1]];
      const gradient = ctx.createLinearGradient(sourceX, p[1], targetX, q[1]);
      gradient.addColorStop(0, alpha(sourceColor, 0.07 + magnitude * 0.32));
      gradient.addColorStop(1, alpha(targetColor, 0.07 + magnitude * 0.32));
      ctx.strokeStyle = this.aesthetic === "minimal" ? gradient : alpha(palette.dim, 0.08 + magnitude * 0.2);
      ctx.lineWidth = 0.55 + magnitude * 0.9;
      ctx.beginPath();
      ctx.moveTo(sourceX, p[1]);
      ctx.bezierCurveTo(controlA[0], controlA[1], controlB[0], controlB[1], targetX, q[1]);
      ctx.stroke();

      // A restrained moving carrier makes the direction of computation
      // legible without implying that every network weight is being drawn.
      const t = (now * 0.00018 + path / sources.length * 0.72 + flowOffset * 0.19) % 1;
      const mt = 1 - t;
      const pulseX = mt ** 3 * sourceX
        + 3 * mt ** 2 * t * controlA[0]
        + 3 * mt * t ** 2 * controlB[0]
        + t ** 3 * targetX;
      const pulseY = mt ** 3 * p[1]
        + 3 * mt ** 2 * t * controlA[1]
        + 3 * mt * t ** 2 * controlB[1]
        + t ** 3 * q[1];
      const pulseColor = t < 0.5 ? sourceColor : targetColor;
      ctx.save();
      ctx.shadowBlur = 4 + magnitude * 8;
      ctx.shadowColor = pulseColor;
      ctx.fillStyle = alpha(pulseColor, 0.24 + magnitude * 0.7);
      ctx.beginPath();
      ctx.arc(pulseX, pulseY, 0.7 + magnitude * 1.15, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  private drawLayer(layer: LayerLayout, now: number, layerOffset: number): void {
    const { ctx } = this;
    const palette = paletteFor(this.aesthetic);
    ctx.fillStyle = palette.text;
    ctx.font = theme.display;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(layer.name, layer.x, 0);
    const meanMagnitude = layer.values.reduce((sum, value) => sum + Math.abs(value), 0) / Math.max(1, layer.values.length);
    const peakMagnitude = layer.values.reduce((peak, value) => Math.max(peak, Math.abs(value)), 0);
    ctx.fillStyle = palette.faint;
    ctx.font = theme.displaySm;
    ctx.fillText(`MEAN |A| ${meanMagnitude.toFixed(2)}  ·  PEAK ${peakMagnitude.toFixed(2)}`, layer.x, 11);
    const radius = Math.max(1.6, Math.min(layer.width / 8, layer.height / 16) * 0.31);
    for (let unit = 0; unit < layer.points.length; unit++) {
      const [x, y] = layer.points[unit]!;
      const value = layer.values[unit] ?? 0;
      const magnitude = Math.min(Math.abs(value), 1);
      const color = value >= 0 ? palette.amber : palette.blue;
      const breath = 0.94 + 0.06 * Math.sin(now * 0.003 + unit * 1.73 + layerOffset);
      ctx.fillStyle = alpha(color, (0.13 + magnitude * 0.82) * breath);
      if (this.aesthetic === "airspace") {
        const size = radius * (0.9 + magnitude * 0.4);
        ctx.fillRect(x - size / 2, y - size / 2, size, size);
      } else if (this.aesthetic === "terminal") {
        ctx.fillRect(x - radius * 0.8, y - radius * 0.45, radius * 1.6, radius * 0.9);
      } else {
        ctx.save();
        if (this.aesthetic === "optical" || (this.aesthetic === "minimal" && magnitude > 0.42)) {
          ctx.shadowBlur = 2 + magnitude * (this.aesthetic === "minimal" ? 5 : 8);
          ctx.shadowColor = color;
        }
        ctx.beginPath();
        ctx.arc(x, y, radius * (0.97 + magnitude * 0.07 * breath), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
  }

  draw(index: number, frame: Frame): void {
    const { ctx } = this;
    const palette = paletteFor(this.aesthetic);
    const { w, h } = fitCanvas(this.canvas, this.ctx);
    ctx.clearRect(0, 0, w, h);
    const snapshot = this.source?.(index, frame) ?? null;
    if (!snapshot) return;
    this.updateInputs(snapshot);
    this.updateAction("throttle", 0, snapshot);
    this.updateAction("gimbal", 1, snapshot);

    const top = 30;
    const bottom = 10;
    const layerHeight = Math.max(40, h - top - bottom);
    const margin = Math.max(6, Math.min(18, w * 0.012));
    const gap = Math.max(18, Math.min(54, w * 0.035));
    const layerWidth = Math.max(72, (w - margin * 2 - gap * 2) / 3);
    const l1 = this.layerLayout("LAYER 1 · 128", "", snapshot.layer1, margin, top, layerWidth, layerHeight);
    const l2 = this.layerLayout("LAYER 2 · 128", "", snapshot.layer2, margin + layerWidth + gap, top, layerWidth, layerHeight);
    const gru = this.layerLayout("GRU MEMORY · 128", "", snapshot.hidden, margin + (layerWidth + gap) * 2, top, layerWidth, layerHeight);
    const now = performance.now();
    this.drawFlow(l1, l2, now, 0);
    this.drawFlow(l2, gru, now, 1);
    this.drawLayer(l1, now, 0);
    this.drawLayer(l2, now, 1.7);
    this.drawLayer(gru, now, 3.4);

    const lastNeuronX = gru.points.reduce((right, point) => Math.max(right, point[0]), 0);
    const loopStartX = lastNeuronX + 2;
    const loopX = Math.min(w - 2, lastNeuronX + 13);
    ctx.strokeStyle = alpha(palette.blue, 0.42);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(loopStartX, gru.y + gru.height * 0.8);
    ctx.bezierCurveTo(loopX, gru.y + gru.height * 0.8, loopX, gru.y + gru.height * 0.2, loopStartX, gru.y + gru.height * 0.2);
    ctx.stroke();
    ctx.fillStyle = palette.blue;
    ctx.beginPath();
    ctx.moveTo(loopStartX, gru.y + gru.height * 0.2);
    ctx.lineTo(loopStartX + 6, gru.y + gru.height * 0.2 - 3);
    ctx.lineTo(loopStartX + 6, gru.y + gru.height * 0.2 + 3);
    ctx.closePath();
    ctx.fill();
  }
}
