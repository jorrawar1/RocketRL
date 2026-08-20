"""Small controller interfaces shared by training and evaluation code."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class ContinuousController(Protocol):
    """Minimal policy interface used by mission rollouts."""

    def reset(self) -> None: ...

    def act(self, observation: np.ndarray) -> np.ndarray: ...


class CallableController:
    """Adapt an observation-to-action callable to ``ContinuousController``."""

    def __init__(
        self,
        action_fn: Callable[[np.ndarray], object],
        reset_fn: Callable[[], None] | None = None,
    ):
        self._action_fn = action_fn
        self._reset_fn = reset_fn

    def reset(self) -> None:
        if self._reset_fn is not None:
            self._reset_fn()

    def act(self, observation: np.ndarray) -> np.ndarray:
        action = np.asarray(self._action_fn(observation), dtype=np.float64)
        if action.shape != (2,):
            raise ValueError(
                f"controller action must have shape (2,), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("controller action must contain only finite values")
        return action


# Compatibility spelling for consumers that do not need to emphasize the
# continuous action space.
Controller = ContinuousController


__all__ = ["CallableController", "ContinuousController", "Controller"]
