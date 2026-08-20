"""Configuration for the bounded crater sample-return mission."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from .terrain import CraterTerrainSpec
from .mission_types import PayloadSpec


def _default_flight_config() -> Config:
    """Keep the original vehicle, but give the two-leg mission room and fuel."""
    return Config(
        world_w=220.0,
        world_h=140.0,
        pad_x=145.0,
        pad_half_w=5.0,
        terrain_amp=0.0,
        terrain_res=2.0,
        fuel_0=100.0,
        burn_rate=1.65,
        max_steps=12_000,
        ray_max_range=80.0,
    )


@dataclass(frozen=True)
class SampleReturnConfig:
    """Mission-only values layered over the unchanged legacy ``Config``."""

    flight: Config = field(default_factory=_default_flight_config)
    terrain: CraterTerrainSpec = field(default_factory=CraterTerrainSpec)
    payload: PayloadSpec = field(
        default_factory=lambda: PayloadSpec(
            mass=0.35,
            offset_body_x=0.80,
            offset_body_y=0.0,
            width=0.65,
            height=0.85,
        )
    )
    sampling_steps: int = 90
    contact_arm_clearance: float = 0.9
    departure_radius: float = 7.0

    # Transparent provisional reward scaffolding.  The physical mission
    # metrics in ``info`` remain the evaluation source of truth.
    sample_landing_reward: float = 25.0
    sample_acquired_reward: float = 15.0
    sample_returned_reward: float = 150.0
    failure_penalty: float = -100.0
    fuel_penalty: float = 0.002

    def validate(self) -> None:
        self.terrain.validate()
        if self.flight.world_w != self.terrain.world_w:
            raise ValueError("flight and terrain world widths must match")
        if self.flight.world_h != self.terrain.world_h:
            raise ValueError("flight and terrain world heights must match")
        if self.flight.pad_half_w != self.terrain.pad_half_w:
            raise ValueError("flight and terrain pad widths must match")
        if self.sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        if self.contact_arm_clearance <= 0.0 or self.departure_radius <= 0.0:
            raise ValueError("contact arming distances must be positive")
        if self.payload.mass < 0.0:
            raise ValueError("payload mass cannot be negative")


DEFAULT_SAMPLE_RETURN_CONFIG = SampleReturnConfig()


__all__ = ["DEFAULT_SAMPLE_RETURN_CONFIG", "SampleReturnConfig"]
