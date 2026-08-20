import { Camera } from "../camera";
import {
  bodyEndpoints,
  bodyPoint,
  bodyToWorld,
  dryBodyCenter,
  enginePosition,
  plumeDirection,
  terrainHeightAt,
  terrainRayDistance,
} from "../geometry";
import { SIM } from "../sim/config";
import {
  alpha,
  paletteFor,
  phaseColor,
  theme,
  type Aesthetic,
} from "../theme";
import {
  BODY_HALF_W,
  THETA,
  VX,
  VY,
  X,
  Y,
  type MissionFixture,
  type Frame,
  type Pad,
  type Vec2,
} from "../types";
import { fitCanvas } from "./canvas";

export interface ViewportInsets {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export class WorldRenderer {
  private ctx: CanvasRenderingContext2D;
  private aesthetic: Aesthetic = "minimal";
  private highlightedRay: number | null = null;

  constructor(
    private canvas: HTMLCanvasElement,
    private fx: MissionFixture,
    private camera: Camera,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    this.ctx = ctx;
  }

  setAesthetic(aesthetic: Aesthetic): void {
    this.aesthetic = aesthetic;
  }

  setHighlightedRay(index: number | null): void { this.highlightedRay = index; }

  draw(
    frame: Frame,
    smooth: boolean,
    viewport: Partial<ViewportInsets> = {},
  ): void {
    const { w, h } = fitCanvas(this.canvas, this.ctx);
    const left = Math.max(0, viewport.left ?? 0);
    const right = Math.max(0, viewport.right ?? 0);
    const top = Math.max(0, viewport.top ?? 0);
    const bottom = Math.max(0, viewport.bottom ?? 0);
    const viewW = Math.max(260, w - left - right);
    const viewH = Math.max(200, h - top - bottom);
    const aspect = viewW / viewH;

    const com: Vec2 = [frame.state[X], frame.state[Y]];
    this.camera.update(com, aspect, smooth);
    const s = this.camera.scale(viewW);
    const px = (p: Vec2) => this.camera.toScreen(p, viewW, viewH);

    this.drawSky(w, h);
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.rect(left, top, viewW, viewH);
    this.ctx.clip();
    this.ctx.translate(left, top);
    this.drawGraticule(viewW, viewH, aspect, px);
    this.drawTerrain(viewH, px);
    this.drawPolicySensors(frame, px);
    const palette = paletteFor(this.aesthetic);
    this.drawPad(this.fx.base_pad, "BASE", palette.blue, px);
    this.drawPad(this.fx.sample_pad, "SAMPLE", palette.amber, px);
    this.drawVehicle(frame, s, px);
    this.drawReticle(frame, com, s, viewW, viewH, px);
    this.ctx.restore();
  }

  // --- layers -------------------------------------------------------------

  /** A shallow vertical gradient: enough depth that the frame is not a slab. */
  private drawSky(w: number, h: number): void {
    const { ctx } = this;
    const p = paletteFor(this.aesthetic);
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    if (this.aesthetic === "optical") {
      sky.addColorStop(0, "#03050a");
      sky.addColorStop(0.62, "#07101d");
      sky.addColorStop(1, "#0d1420");
    } else if (this.aesthetic === "terminal") {
      sky.addColorStop(0, "#030604");
      sky.addColorStop(0.72, p.ground);
      sky.addColorStop(1, "#08110a");
    } else {
      sky.addColorStop(0, this.aesthetic === "airspace" ? "#020203" : "#08080a");
      sky.addColorStop(0.72, p.ground);
      sky.addColorStop(1, this.aesthetic === "airspace" ? "#08090c" : "#0d0d10");
    }
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);

    if (this.aesthetic === "airspace") {
      const horizon = ctx.createLinearGradient(0, h - 32, 0, h);
      horizon.addColorStop(0, alpha(p.blue, 0));
      horizon.addColorStop(0.45, alpha(p.blue, 0.14));
      horizon.addColorStop(0.72, alpha(p.red, 0.17));
      horizon.addColorStop(1, alpha(p.amber, 0.2));
      ctx.fillStyle = horizon;
      ctx.fillRect(0, h - 36, w, 36);
    }
  }

  private drawGraticule(
    w: number,
    h: number,
    aspect: number,
    px: (p: Vec2) => Vec2,
  ): void {
    const { ctx } = this;
    const p = paletteFor(this.aesthetic);
    const b = this.camera.bounds(aspect);
    const step = 20;
    ctx.lineWidth = 1;
    ctx.font = theme.displaySm;
    ctx.textBaseline = "top";

    ctx.strokeStyle = alpha(p.hairline, this.aesthetic === "terminal" ? 0.82 : 0.7);
    ctx.setLineDash(this.aesthetic === "terminal" ? [1, 5] : []);
    ctx.beginPath();
    for (let x = Math.ceil(b.x0 / step) * step; x <= b.x1; x += step) {
      const p = px([x, b.y0]);
      const q = px([x, b.y1]);
      ctx.moveTo(Math.round(p[0]) + 0.5, q[1]);
      ctx.lineTo(Math.round(p[0]) + 0.5, p[1]);
    }
    for (let y = Math.ceil(b.y0 / step) * step; y <= b.y1; y += step) {
      const p = px([b.x0, y]);
      const q = px([b.x1, y]);
      ctx.moveTo(p[0], Math.round(p[1]) + 0.5);
      ctx.lineTo(q[0], Math.round(p[1]) + 0.5);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Ticked, numbered axes along the left and bottom edges.
    ctx.fillStyle = p.faint;
    ctx.textAlign = "left";
    for (let x = Math.ceil(b.x0 / step) * step; x <= b.x1; x += step) {
      const p = px([x, b.y0]);
      if (p[0] < 22 || p[0] > w - 22) continue;
      ctx.fillText(String(x), p[0] + 3, h - 13);
    }
    for (let y = Math.ceil(b.y0 / step) * step; y <= b.y1; y += step) {
      const p = px([b.x0, y]);
      if (p[1] < 14 || p[1] > h - 14) continue;
      ctx.fillText(String(y), 4, p[1] + 3);
    }
  }

  private drawTerrain(h: number, px: (p: Vec2) => Vec2): void {
    const { ctx } = this;
    const p = paletteFor(this.aesthetic);
    const v = this.fx.terrain.vertices;
    if (v.length < 2) return;

    const trace = () => {
      ctx.beginPath();
      const first = px(v[0]!);
      ctx.moveTo(first[0], first[1]);
      for (let i = 1; i < v.length; i++) {
        const p = px(v[i]!);
        ctx.lineTo(p[0], p[1]);
      }
      return first;
    };

    const first = trace();
    const last = px(v[v.length - 1]!);
    ctx.lineTo(last[0], h + 2);
    ctx.lineTo(first[0], h + 2);
    ctx.closePath();
    ctx.fillStyle = p.sunk;
    ctx.fill();

    // Hairline horizon, stroked separately so the fill never thickens it.
    trace();
    ctx.strokeStyle = p.dim;
    ctx.lineWidth = this.aesthetic === "airspace" ? 1.25 : 1;
    ctx.stroke();

    if (this.aesthetic === "optical") {
      ctx.save();
      ctx.shadowBlur = 14;
      ctx.shadowColor = alpha(p.blue, 0.65);
      trace();
      ctx.strokeStyle = alpha(p.blue, 0.28);
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.restore();
    }
  }

  private drawPad(
    pad: Pad,
    label: string,
    color: string,
    px: (p: Vec2) => Vec2,
  ): void {
    const { ctx } = this;
    const p = paletteFor(this.aesthetic);
    const a = px([pad.x - pad.half_width, pad.y]);
    const b = px([pad.x + pad.half_width, pad.y]);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();

    // Outboard tick marks read as a landing bracket at any zoom.
    ctx.lineWidth = 1;
    ctx.strokeStyle = alpha(color, 0.55);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(a[0], a[1] - 7);
    ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0], b[1] - 7);
    ctx.stroke();

    ctx.font = theme.displaySm;
    ctx.fillStyle = alpha(color, 0.8);
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(label, (a[0] + b[0]) / 2, a[1] - 11);

    if (this.aesthetic === "airspace") {
      ctx.strokeStyle = alpha(color, 0.22);
      ctx.beginPath();
      ctx.arc((a[0] + b[0]) / 2, a[1], 18, Math.PI, Math.PI * 2);
      ctx.stroke();
    } else if (this.aesthetic === "terminal") {
      ctx.fillStyle = alpha(p.ground, 0.94);
      ctx.fillRect((a[0] + b[0]) / 2 - 24, a[1] - 21, 48, 11);
      ctx.fillStyle = color;
      ctx.fillText(`[${label}]`, (a[0] + b[0]) / 2, a[1] - 11);
    }
  }

  /** Draw the exact five terrain-clearance inputs consumed by the actor. */
  private drawPolicySensors(frame: Frame, px: (p: Vec2) => Vec2): void {
    const { ctx } = this;
    const origin: Vec2 = [frame.state[X], frame.state[Y]];
    const a = px(origin);
    ctx.save();
    ctx.lineWidth = 1;
    for (let index = 0; index < SIM.rayCount; index++) {
      const mix = index / (SIM.rayCount - 1);
      const angle = SIM.rayAngleLo + mix * (SIM.rayAngleHi - SIM.rayAngleLo);
      const dx = Math.cos(angle);
      const dy = Math.sin(angle);
      const distance = terrainRayDistance(
        this.fx.terrain.vertices,
        origin[0],
        origin[1],
        dx,
        dy,
        SIM.rayMaxRange,
      );
      const b = px([origin[0] + dx * distance, origin[1] + dy * distance]);
      const color = index === 2 ? theme.amber : theme.blue;
      const highlighted = index === this.highlightedRay;
      ctx.strokeStyle = alpha(color, highlighted ? 0.72 : index === 2 ? 0.18 : 0.09);
      ctx.lineWidth = highlighted ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      ctx.fillStyle = alpha(color, highlighted ? 0.9 : 0.36);
      const marker = highlighted ? 4 : 2;
      ctx.fillRect(b[0] - marker / 2, b[1] - marker / 2, marker, marker);
    }
    ctx.restore();
  }

  private drawVehicle(frame: Frame, s: number, px: (p: Vec2) => Vec2): void {
    const { ctx } = this;
    const p = paletteFor(this.aesthetic);
    const height = this.fx.vehicle.body_height;
    const [base, tip] = bodyEndpoints(frame, height);
    const hull = Math.max(1, 0.09 * s);

    // Exhaust plume first, so the body sits over it.
    const throttle = frame.action[0];
    if (throttle > 0.01) {
      const eng = enginePosition(frame, height);
      const dir = plumeDirection(frame);
      const len = 1.0 + throttle * 4.2;
      const e0 = px(eng);
      const e1 = px([eng[0] + dir[0] * len, eng[1] + dir[1] * len]);
      const grad = ctx.createLinearGradient(e0[0], e0[1], e1[0], e1[1]);
      grad.addColorStop(0, alpha(p.text, 0.9));
      grad.addColorStop(0.36, alpha(this.aesthetic === "optical" ? p.red : p.amber, 0.68));
      grad.addColorStop(1, alpha(p.amber, 0));
      ctx.strokeStyle = grad;
      ctx.lineWidth = Math.max(1.5, 0.26 * s);
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(e0[0], e0[1]);
      ctx.lineTo(e1[0], e1[1]);
      ctx.stroke();
      ctx.lineCap = "butt";
    }

    const bp = (along: number, across: number) =>
      px(bodyPoint(frame, height, along, across));

    // Hull: a slab that tapers into a nose cone over the top 18%.
    const outline: Vec2[] = [
      bp(0, -BODY_HALF_W),
      bp(0, +BODY_HALF_W),
      bp(height * 0.82, +BODY_HALF_W),
      px(tip),
      bp(height * 0.82, -BODY_HALF_W),
    ];
    ctx.beginPath();
    outline.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    ctx.closePath();
    ctx.fillStyle = alpha(p.ground, 0.85);
    ctx.fill();
    ctx.strokeStyle = p.text;
    ctx.lineWidth = hull;
    ctx.lineJoin = "round";
    ctx.stroke();

    // Centreline, dim: reads as a body axis without competing with the hull.
    const bs = px(base);
    const ts = px(tip);
    ctx.strokeStyle = alpha(p.text, 0.3);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bs[0], bs[1]);
    ctx.lineTo(ts[0], ts[1]);
    ctx.stroke();

    this.drawPod(frame, px);

    // Combined centre of mass — the one thing the payload physically moves.
    const comPx = px([frame.state[X], frame.state[Y]]);
    ctx.strokeStyle = frame.payload_mass > 0 ? p.amber : p.dim;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(comPx[0], comPx[1], 3, 0, Math.PI * 2);
    ctx.stroke();
  }

  /** Fixed side pod. Always drawn; the fill level shows the sample loading. */
  private drawPod(frame: Frame, px: (p: Vec2) => Vec2): void {
    const { ctx } = this;
    const p = paletteFor(this.aesthetic);
    const pod = this.fx.payload;
    const theta = frame.state[THETA];
    const center = dryBodyCenter(frame);
    const at = (across: number, along: number): Vec2 => {
      const w = bodyToWorld(
        [pod.offset_body[0] + across, pod.offset_body[1] + along],
        theta,
      );
      return px([center[0] + w[0], center[1] + w[1]]);
    };

    const hw = pod.width / 2;
    const hh = pod.height / 2;
    const corners = [at(-hw, -hh), at(hw, -hh), at(hw, hh), at(-hw, hh)];

    const fill = frame.payload_mass > 0 ? 1 : frame.payload_fill;
    if (fill > 0) {
      const top = -hh + pod.height * fill;
      const inner = [
        at(-pod.width * 0.38, -hh),
        at(+pod.width * 0.38, -hh),
        at(+pod.width * 0.38, top),
        at(-pod.width * 0.38, top),
      ];
      ctx.beginPath();
      inner.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.closePath();
      ctx.fillStyle = alpha(p.amber, frame.payload_mass > 0 ? 0.75 : 0.45);
      ctx.fill();
    } else {
      // Empty pod gets a struck-through diagonal, as in the pygame harness.
      ctx.strokeStyle = p.faint;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(corners[0]![0], corners[0]![1]);
      ctx.lineTo(corners[2]![0], corners[2]![1]);
      ctx.stroke();
    }

    ctx.beginPath();
    corners.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    ctx.closePath();
    ctx.strokeStyle = fill > 0 ? p.amber : p.dim;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  private drawReticle(
    frame: Frame,
    com: Vec2,
    s: number,
    w: number,
    h: number,
    px: (p: Vec2) => Vec2,
  ): void {
    const { ctx } = this;
    const p = paletteFor(this.aesthetic);
    const c = px(com);
    const r = Math.max(16, this.fx.vehicle.body_height * s * 0.95);

    // Four measured corner marks give the velocity vector a local frame.
    ctx.strokeStyle = alpha(p.text, this.aesthetic === "optical" ? 0.42 : 0.28);
    ctx.lineWidth = 1;
    const tick = Math.max(5, r * 0.28);
    ctx.beginPath();
    ctx.moveTo(c[0] - r, c[1] - r + tick); ctx.lineTo(c[0] - r, c[1] - r); ctx.lineTo(c[0] - r + tick, c[1] - r);
    ctx.moveTo(c[0] + r - tick, c[1] - r); ctx.lineTo(c[0] + r, c[1] - r); ctx.lineTo(c[0] + r, c[1] - r + tick);
    ctx.moveTo(c[0] + r, c[1] + r - tick); ctx.lineTo(c[0] + r, c[1] + r); ctx.lineTo(c[0] + r - tick, c[1] + r);
    ctx.moveTo(c[0] - r + tick, c[1] + r); ctx.lineTo(c[0] - r, c[1] + r); ctx.lineTo(c[0] - r, c[1] + r - tick);
    ctx.stroke();

    // Velocity vector, scaled so 10 m/s draws one reticle half-width.
    const vx = frame.state[VX];
    const vy = frame.state[VY];
    const speed = Math.hypot(vx, vy);
    if (speed > 0.15) {
      const k = r / 10;
      ctx.strokeStyle = alpha(p.blue, 0.9);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(c[0], c[1]);
      ctx.lineTo(c[0] + vx * k, c[1] - vy * k);
      ctx.stroke();
    }

    const agl = com[1] - terrainHeightAt(this.fx.terrain.vertices, com[0]);
    ctx.font = theme.displaySm;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lines = [
      frame.phase,
      "AGL " + agl.toFixed(1),
      "V " + speed.toFixed(1),
      ((frame.state[THETA] * 180) / Math.PI).toFixed(1) + "°",
    ];
    const lx = Math.min(c[0] + r + 10, w - 78);
    const top = Math.min(Math.max(c[1] - 18, 14), h - 60);

    ctx.strokeStyle = alpha(p.hairlineHi, 0.9);
    ctx.beginPath();
    ctx.moveTo(c[0] + r + 2, c[1]);
    ctx.lineTo(lx - 4, c[1]);
    ctx.stroke();

    ctx.fillStyle = phaseColor(frame.phase, this.aesthetic);
    ctx.beginPath();
    ctx.arc(lx - 6, top, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = alpha(p.text, 0.7);
    let ly = top;
    for (const line of lines) {
      ctx.fillText(line, lx, ly);
      ly += 12;
    }

    if (this.aesthetic === "airspace") {
      ctx.save();
      ctx.strokeStyle = alpha(p.blue, 0.18);
      ctx.setLineDash([2, 6]);
      for (const scale of [1.75, 2.55]) {
        ctx.beginPath();
        ctx.arc(c[0], c[1], r * scale, Math.PI * 1.05, Math.PI * 1.94);
        ctx.stroke();
      }
      ctx.restore();
    } else if (this.aesthetic === "optical") {
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      ctx.strokeStyle = alpha(p.red, 0.2);
      ctx.strokeRect(c[0] - r - 2, c[1] - r, r * 2, r * 2);
      ctx.strokeStyle = alpha(p.blue, 0.24);
      ctx.strokeRect(c[0] - r + 2, c[1] - r, r * 2, r * 2);
      ctx.restore();
    } else if (this.aesthetic === "terminal") {
      ctx.strokeStyle = alpha(p.blue, 0.62);
      ctx.strokeRect(c[0] - r, c[1] - r, r * 2, r * 2);
      ctx.fillStyle = p.blue;
      ctx.fillRect(c[0] - 1, c[1] - 1, 3, 3);
    }
  }
}
