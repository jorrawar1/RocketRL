"""Terrain interface. Flat ground in v1; a polyline height-field drops in later
by implementing the same two methods — the env never assumes flatness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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
