import { terrainHeightAt } from "./geometry";

import {
  FUEL,
  OMEGA,
  THETA,
  VX,
  VY,
  X,
  Y,
  type MissionFixture,
  type Frame,
} from "./types";

type Accent = "" | "is-blue" | "is-amber" | "is-red";
interface Row {
  label: string;
  value: string;
  unit?: string;
  accent?: Accent;
}

export interface FeedEvent {
  t: number;
  label: string;
}

/** Reuse the DOM nodes: this repaints every animation frame. */
class RowList {
  private rows: HTMLElement[] = [];
  constructor(private host: HTMLElement) {}

  render(rows: Row[]): void {
    while (this.rows.length < rows.length) {
      const el = document.createElement("div");
      el.className = "row";
      el.innerHTML =
        '<span class="l"></span><span class="v"></span><span class="u"></span>';
      this.host.appendChild(el);
      this.rows.push(el);
    }
    for (let i = 0; i < this.rows.length; i++) {
      const el = this.rows[i]!;
      const row = rows[i];
      if (!row) {
        el.hidden = true;
        continue;
      }
      el.hidden = false;
      el.className = "row " + (row.accent ?? "");
      el.children[0]!.textContent = row.label;
      el.children[1]!.textContent = row.value;
      el.children[2]!.textContent = row.unit ?? "";
    }
  }
}

export class Panels {
  private flight: RowList;
  private payload: RowList;
  private feedHost: HTMLElement;
  private feedLines: HTMLElement[] = [];
  private feedCount = -1;


  constructor(private fx: MissionFixture) {

    const flightHost = el("readout-flight");
    const payloadHost = el("readout-payload");
    flightHost.textContent = "";
    payloadHost.textContent = "";
    this.flight = new RowList(flightHost);
    this.payload = new RowList(payloadHost);
    this.feedHost = el("feed");
    this.feedHost.textContent = "";


    el("hdr-seed").textContent = fx.mission_seed
      .toString(16)
      .toUpperCase()
      .padStart(8, "0");
    this.setOutcome(null);
  }

  setPilot(label: string): void {
    el("hdr-pilot").textContent = label.toUpperCase();
  }

  setOutcome(outcome: string | null): void {
    const node = el("hdr-outcome");
    node.textContent = outcome ?? "IN FLIGHT";
    node.style.color = !outcome
      ? "var(--dim)"
      : outcome === "SAMPLE_RETURNED"
        ? "var(--blue)"
        : "var(--red)";
  }

  update(frame: Frame): void {
    const st = frame.state;
    const speed = Math.hypot(st[VX], st[VY]);
    const agl = st[Y] - terrainHeightAt(this.fx.terrain.vertices, st[X]);
    const thetaDeg = (st[THETA] * 180) / Math.PI;

    this.flight.render([
      { label: "PHASE", value: frame.phase, accent: accentFor(frame.phase) },
      { label: "ALT AGL", value: agl.toFixed(2), unit: "m" },
      { label: "POS X", value: st[X].toFixed(2), unit: "m" },
      { label: "VEL X", value: signed(st[VX]), unit: "m/s" },
      { label: "VEL Y", value: signed(st[VY]), unit: "m/s" },
      { label: "SPEED", value: speed.toFixed(2), unit: "m/s" },
      { label: "PITCH", value: signed(thetaDeg), unit: "°" },
      { label: "RATE", value: signed(st[OMEGA]), unit: "r/s" },
      { label: "THROTTLE", value: (frame.action[0] * 100).toFixed(0), unit: "%" },
      { label: "GIMBAL", value: signed(frame.action[1] * 10), unit: "°" },
      {
        label: "FUEL",
        value: st[FUEL].toFixed(1),
        accent: st[FUEL] < 15 ? "is-red" : "",
      },
    ]);

    const attached = frame.payload_mass > 0;
    this.payload.render([
      {
        label: "STATUS",
        value: attached ? "STOWED" : frame.payload_fill > 0 ? "LOADING" : "MANIFEST",
        accent: attached || frame.payload_fill > 0 ? "is-amber" : "",
      },
      { label: "FILL", value: bar(frame.payload_fill || (attached ? 1 : 0)) },
      {
        label: "MASS",
        value: (attached ? frame.payload_mass : this.fx.payload.mass).toFixed(3),
        unit: "kg",
      },
      {
        label: attached ? "COM OFF" : "ARM X",
        value: (attached
          ? frame.com_offset_body[0]
          : this.fx.payload.offset_body[0]
        ).toFixed(3),
        unit: "m",
      },
    ]);

  }


  /** Feed rebuilds only when the event list changes length. */
  renderFeed(events: readonly FeedEvent[], t: number): void {
    if (events.length !== this.feedCount) {
      this.feedCount = events.length;
      this.feedHost.textContent = "";
      this.feedLines = events.map((e) => {
        const node = line("▸", e.t.toFixed(1), e.label);
        this.feedHost.appendChild(node);
        return node;
      });
    }
    let latest = -1;
    events.forEach((e, i) => {
      const passed = t + 1e-9 >= e.t;
      if (passed) latest = i;
      this.feedLines[i]!.className = "line " + (passed ? "is-past" : "");
    });
    if (latest >= 0) this.feedLines[latest]!.className = "line is-latest";
  }
}

function accentFor(phase: string): Accent {
  if (phase === "SAMPLING") return "is-amber";
  if (phase === "FAILURE") return "is-red";
  return "is-blue";
}

function signed(v: number): string {
  return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
}

/** Ten-cell block meter — a gauge without needing a second colour. */
function bar(fill: number): string {
  const n = Math.round(Math.min(Math.max(fill, 0), 1) * 10);
  return "█".repeat(n) + "░".repeat(10 - n);
}

function line(mark: string, t: string, msg: string): HTMLElement {
  const node = document.createElement("div");
  node.className = "line";
  const a = document.createElement("span");
  a.textContent = mark;
  const b = document.createElement("span");
  b.className = "t";
  b.textContent = t;
  const c = document.createElement("span");
  c.className = "m";
  c.textContent = msg;
  node.append(a, b, c);
  return node;
}

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node;
}
