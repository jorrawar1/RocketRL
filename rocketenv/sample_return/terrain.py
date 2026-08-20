"""Deterministic terrain for the continuous crater sample-return mission.

The terrain is deliberately structured rather than fully procedural: a level
base leads to a raised crater rim, a smooth bowl, and a flat sample pad on the
crater floor.  A seed only changes low-amplitude roughness away from the pads,
so every generated mission keeps the same readable silhouette.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..terrain import PolylineTerrain


@dataclass(frozen=True)
class CraterTerrainSpec:
    """Parameters defining one crater mission map, in world metres."""

    world_w: float = 220.0
    world_h: float = 140.0
    base_x: float = 28.0
    sample_x: float = 145.0
    crater_center_x: float = 145.0
    crater_half_width: float = 62.0
    floor_y: float = 8.0
    rim_y: float = 45.0
    outer_y: float = 30.0
    pad_half_w: float = 5.0
    resolution: float = 2.0
    roughness: float = 0.7

    def validate(self) -> None:
        """Raise ``ValueError`` when the requested map cannot be constructed."""
        values = np.asarray(
            [
                self.world_w,
                self.world_h,
                self.base_x,
                self.sample_x,
                self.crater_center_x,
                self.crater_half_width,
                self.floor_y,
                self.rim_y,
                self.outer_y,
                self.pad_half_w,
                self.resolution,
                self.roughness,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("crater terrain parameters must be finite")
        if self.world_w <= 0.0 or self.world_h <= 0.0:
            raise ValueError("world dimensions must be positive")
        if self.resolution <= 0.0 or self.pad_half_w <= 0.0:
            raise ValueError("resolution and pad width must be positive")
        if self.crater_half_width <= self.pad_half_w:
            raise ValueError("crater must be wider than a landing pad")
        if self.roughness < 0.0:
            raise ValueError("roughness cannot be negative")

        left_rim = self.crater_center_x - self.crater_half_width
        right_rim = self.crater_center_x + self.crater_half_width
        if left_rim <= 0.0 or right_rim >= self.world_w:
            raise ValueError("crater rims must lie inside the world")
        if not (left_rim + self.pad_half_w < self.sample_x < right_rim - self.pad_half_w):
            raise ValueError("sample pad must lie fully inside the crater")
        if not (
            self.pad_half_w <= self.base_x <= self.world_w - self.pad_half_w
        ):
            raise ValueError("base pad must lie fully inside the world")
        base_left_of_crater = self.base_x + self.pad_half_w < left_rim
        base_right_of_crater = self.base_x - self.pad_half_w > right_rim
        if not (base_left_of_crater or base_right_of_crater):
            raise ValueError("base pad must lie outside the crater")
        if not (0.0 <= self.floor_y < self.outer_y <= self.rim_y < self.world_h):
            raise ValueError(
                "heights must satisfy 0 <= floor < outer <= rim < world height"
            )
        if self.floor_y - self.roughness < 0.0:
            raise ValueError("roughness can move terrain below the world")
        if self.rim_y + self.roughness > self.world_h:
            raise ValueError("roughness can move terrain above the world")


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


class CraterSampleTerrain(PolylineTerrain):
    """A seeded polyline height field with explicit base and sample pads."""

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        seed: int,
        spec: CraterTerrainSpec,
    ) -> None:
        super().__init__(xs, ys)
        self.seed = int(seed)
        self.spec = spec
        self.base_x = spec.base_x
        self.sample_x = spec.sample_x
        self.crater_center_x = spec.crater_center_x
        self.left_rim_x = spec.crater_center_x - spec.crater_half_width
        self.right_rim_x = spec.crater_center_x + spec.crater_half_width
        self.base_y = self.height_at(self.base_x)
        self.sample_y = self.height_at(self.sample_x)
        self.left_rim_y = self.height_at(self.left_rim_x)
        self.right_rim_y = self.height_at(self.right_rim_x)

    @classmethod
    def from_seed(
        cls, seed: int, spec: CraterTerrainSpec
    ) -> "CraterSampleTerrain":
        """Build the same terrain vertices for the same seed and specification."""
        spec.validate()
        left_rim = spec.crater_center_x - spec.crater_half_width
        right_rim = spec.crater_center_x + spec.crater_half_width

        regular_xs = np.arange(0.0, spec.world_w, spec.resolution)
        critical_xs = np.asarray(
            [
                0.0,
                spec.world_w,
                spec.base_x - spec.pad_half_w,
                spec.base_x,
                spec.base_x + spec.pad_half_w,
                left_rim,
                spec.sample_x - spec.pad_half_w,
                spec.sample_x,
                spec.sample_x + spec.pad_half_w,
                spec.crater_center_x,
                right_rim,
            ],
            dtype=np.float64,
        )
        xs = np.unique(np.concatenate((regular_xs, critical_xs)))

        # Outside the crater, the plateau rises smoothly toward either rim.
        # Inside it, a smoothstep bowl has horizontal tangents at the floor
        # and rims, avoiding discontinuities and artificial vertical walls.
        ys = np.full_like(xs, spec.outer_y)
        left_exterior = xs < left_rim
        left_rise_start = left_rim - 0.5 * spec.crater_half_width
        left_t = (xs[left_exterior] - left_rise_start) / (
            left_rim - left_rise_start
        )
        ys[left_exterior] += (spec.rim_y - spec.outer_y) * _smoothstep(left_t)

        right_exterior = xs > right_rim
        right_fall_end = right_rim + 0.5 * spec.crater_half_width
        right_t = (right_fall_end - xs[right_exterior]) / (
            right_fall_end - right_rim
        )
        ys[right_exterior] += (spec.rim_y - spec.outer_y) * _smoothstep(right_t)

        inside = (xs >= left_rim) & (xs <= right_rim)
        radius = np.abs(xs[inside] - spec.crater_center_x) / spec.crater_half_width
        ys[inside] = spec.floor_y + (spec.rim_y - spec.floor_y) * _smoothstep(radius)

        if spec.roughness > 0.0:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, 1.0, xs.size)
            noise = np.convolve(
                np.pad(noise, 2, mode="edge"), np.ones(5) / 5.0, mode="valid"
            )
            peak = float(np.max(np.abs(noise)))
            if peak > 0.0:
                noise *= spec.roughness / peak

            # Fade roughness to zero around both pads.  This keeps their
            # approaches readable and makes their full declared widths exact.
            clearance = np.minimum(
                np.abs(xs - spec.base_x) - spec.pad_half_w,
                np.abs(xs - spec.sample_x) - spec.pad_half_w,
            )
            fade = _smoothstep(clearance / (2.0 * spec.resolution))
            ys += noise * fade

        base_pad = np.abs(xs - spec.base_x) <= spec.pad_half_w
        sample_pad = np.abs(xs - spec.sample_x) <= spec.pad_half_w
        ys[base_pad] = spec.outer_y
        ys[sample_pad] = spec.floor_y
        ys[np.isclose(xs, left_rim) | np.isclose(xs, right_rim)] = spec.rim_y

        return cls(xs, ys, seed=seed, spec=spec)

    @property
    def base_pad_bounds(self) -> tuple[float, float]:
        return (
            self.base_x - self.spec.pad_half_w,
            self.base_x + self.spec.pad_half_w,
        )

    @property
    def sample_pad_bounds(self) -> tuple[float, float]:
        return (
            self.sample_x - self.spec.pad_half_w,
            self.sample_x + self.spec.pad_half_w,
        )

    @property
    def vertices(self) -> np.ndarray:
        """Return terrain vertices as an export-friendly ``(N, 2)`` copy."""
        return np.column_stack((self.xs, self.ys)).copy()


__all__ = ["CraterSampleTerrain", "CraterTerrainSpec"]
