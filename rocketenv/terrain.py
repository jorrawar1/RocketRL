"""Terrain interface and implementations: flat ground and a polyline
height-field with a seeded procedural generator.  The env never assumes
flatness — everything goes through height_at / ray_distance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Terrain(ABC):
    @abstractmethod
    def height_at(self, x: float) -> float:
        """Ground height (world y) at horizontal position x."""

    @abstractmethod
    def ray_distance(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float
    ) -> float:
        """Distance from origin (ox, oy) along unit direction (dx, dy) to the
        first terrain intersection, clamped to max_range."""


class FlatTerrain(Terrain):
    def __init__(self, height: float = 0.0):
        self.height = height

    def height_at(self, x: float) -> float:
        return self.height

    def ray_distance(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float
    ) -> float:
        # Ray vs. the horizontal line y = height (closed form).
        if dy >= 0.0:
            # Parallel or pointing up: hits only if starting at/below ground.
            return 0.0 if oy <= self.height else max_range
        t = (self.height - oy) / dy
        if t < 0.0:
            return 0.0  # origin already below ground
        return min(t, max_range)


class PolylineTerrain(Terrain):
    """Height-field defined by vertices (xs strictly increasing, spanning the
    world).  height_at is linear interpolation; ray_distance is exact
    ray-vs-segment intersection, vectorized across segments."""

    def __init__(self, xs, ys):
        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)
        if len(self.xs) < 2 or np.any(np.diff(self.xs) <= 0):
            raise ValueError("xs must be strictly increasing with >= 2 points")
        # segment starts and direction vectors, precomputed
        self._px = self.xs[:-1]
        self._py = self.ys[:-1]
        self._sx = np.diff(self.xs)
        self._sy = np.diff(self.ys)

    def height_at(self, x: float) -> float:
        return float(np.interp(x, self.xs, self.ys))

    def ray_distance(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float
    ) -> float:
        if oy <= self.height_at(ox):
            return 0.0
        # o + t*d = p + u*s  ->  t = (p-o) x s / (d x s),  u = (p-o) x d / (d x s)
        denom = dx * self._sy - dy * self._sx
        qpx = self._px - ox
        qpy = self._py - oy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qpx * self._sy - qpy * self._sx) / denom
            u = (qpx * dy - qpy * dx) / denom
        valid = (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
        if not np.any(valid):
            return max_range
        return float(min(t[valid].min(), max_range))


def generate_terrain(rng, cfg):
    """Seeded procedural rolling hills with a flattened landing pad.

    Deterministic for a given rng state.  Returns (terrain, pad_x); the
    caller applies pad_x via reset(options={"pad_x": ...}) so reward and
    observation follow the pad.  Spawn feasibility holds by construction:
    hills top out at terrain_amp << spawn altitude.
    """
    n = int(round(cfg.world_w / cfg.terrain_res)) + 1
    xs = np.linspace(0.0, cfg.world_w, n)
    h = np.cumsum(rng.normal(0.0, 1.0, n))         # random walk...
    h = np.convolve(np.pad(h, 2, mode="edge"),      # ...box-smoothed
                    np.ones(5) / 5.0, mode="valid")
    h -= h.min()
    span = h.max()
    if span > 1e-9:
        h *= cfg.terrain_amp * rng.uniform(0.55, 1.0) / span

    pad_x = float(rng.uniform(15.0, cfg.world_w - 15.0))
    pad_y = float(np.interp(pad_x, xs, h))
    # flatten enough vertices that the whole pad plus a margin lies on one
    # level stretch regardless of grid alignment
    flatten_r = cfg.pad_half_w + cfg.terrain_res + 1.0
    h[np.abs(xs - pad_x) <= flatten_r] = pad_y
    return PolylineTerrain(xs, h), pad_x
