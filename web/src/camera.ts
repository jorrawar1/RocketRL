import type { Vec2 } from "./types";

export interface Bounds {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

interface Rect {
  cx: number;
  cy: number;
  halfW: number;
}

/**
 * Maps world metres to canvas pixels.
 *
 * Framing is driven by a content box (terrain plus the flown trajectory)
 * rather than the nominal world box: the vehicle never climbs anywhere near
 * world_h, so fitting the whole world wastes most of the frame on empty sky.
 */
export class Camera {
  private cur: Rect;
  private target: Rect;

  constructor(private content: Bounds) {
    const cx = (content.x0 + content.x1) / 2;
    const cy = (content.y0 + content.y1) / 2;
    this.cur = { cx, cy, halfW: (content.x1 - content.x0) / 2 };
    this.target = { ...this.cur };
  }

  /** Keep the entire mission map visible at every point in the flight. */
  update(_focus: Vec2, aspect: number, smooth: boolean): void {
    const contentW = this.content.x1 - this.content.x0;
    const contentH = this.content.y1 - this.content.y0;
    // Both axes must fit, so take whichever constraint is tighter. A wide
    // viewport is limited by content height; a tall one by content width.
    const fittedHalfW = Math.max(contentW / 2, (contentH / 2) * aspect) * 1.02;
    // A short viewport (for example while the policy dock is open) should not
    // make the horizontal world scale balloon far beyond the 0..220 m map.
    // A small cap keeps both pads prominent; the existing vertical margin is
    // sufficient for the trained flight envelope.
    const halfW = Math.min(fittedHalfW, (contentW / 2) * 1.1);
    this.target = {
      cx: (this.content.x0 + this.content.x1) / 2,
      cy: (this.content.y0 + this.content.y1) / 2,
      halfW,
    };
    const k = smooth ? 0.14 : 1;
    this.cur.cx += (this.target.cx - this.cur.cx) * k;
    this.cur.cy += (this.target.cy - this.cur.cy) * k;
    this.cur.halfW += (this.target.halfW - this.cur.halfW) * k;
  }

  /** Pixels per world metre for the given canvas width. */
  scale(pxWidth: number): number {
    return pxWidth / (this.cur.halfW * 2);
  }

  toScreen(p: Vec2, pxW: number, pxH: number): Vec2 {
    const s = this.scale(pxW);
    return [
      pxW / 2 + (p[0] - this.cur.cx) * s,
      pxH / 2 - (p[1] - this.cur.cy) * s,
    ];
  }

  /** Visible world bounds, used to skip off-screen graticule lines. */
  bounds(aspect: number): Bounds {
    const halfH = this.cur.halfW / Math.max(aspect, 1e-6);
    return {
      x0: this.cur.cx - this.cur.halfW,
      x1: this.cur.cx + this.cur.halfW,
      y0: this.cur.cy - halfH,
      y1: this.cur.cy + halfH,
    };
  }
}

/** Terrain plus flown trajectory, with breathing room. */
export function contentBounds(
  terrain: Vec2[],
  path: Vec2[],
  margin = { below: 7, above: 9, side: 2 },
): Bounds {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const [x, y] of [...terrain, ...path]) {
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return {
    x0: x0 - margin.side,
    x1: x1 + margin.side,
    y0: y0 - margin.below,
    y1: y1 + margin.above,
  };
}
