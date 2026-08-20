"""Small domain types shared by the sample-return mission modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


class MissionPhase(IntEnum):
    """High-level phase of one continuous sample-return mission."""

    OUTBOUND = 0
    SAMPLING = 1
    RETURN = 2
    SUCCESS = 3
    FAILURE = 4


@dataclass(frozen=True)
class PayloadSpec:
    """Rigid rectangular payload, expressed relative to the dry body center."""

    mass: float
    offset_body_x: float
    offset_body_y: float
    width: float
    height: float
    sample_id: str = "regolith"

    def __post_init__(self) -> None:
        values = (
            self.mass,
            self.offset_body_x,
            self.offset_body_y,
            self.width,
            self.height,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("payload values must be finite")
        if self.mass < 0.0:
            raise ValueError("payload mass cannot be negative")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("payload dimensions must be positive")


@dataclass
class SampleReturnState:
    """Mission bookkeeping kept separate from the seven-value rocket state."""

    phase: MissionPhase
    payload_attached: bool
    sample_collected: bool
    contact_armed: bool
    sampling_steps_remaining: int
    outcome: str | None = None
