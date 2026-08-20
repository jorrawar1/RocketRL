"""Thin Gymnasium adapters for sample-return curriculum training.

This module deliberately contains no learning algorithm.  It changes the rate,
starting boundary, and training-only observation views around the physical
``SampleReturnEnv``.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

from ..physics import OMEGA, THETA, VX, VY, X, Y
from .config import SampleReturnConfig
from .env import SAMPLE_RETURNED, SampleReturnEnv
from .observation import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_NAMES,
    OBSERVATION_DIM,
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
    actor_observation,
    flat_observation,
)
from .mission_types import MissionPhase
from .vehicle import body_endpoints


class TrainingTask(str, Enum):
    """Stable episode boundaries used to assemble a curriculum."""

    SAMPLE_LANDING = "sample_landing"
    RETURN_LANDING = "return_landing"
    OUTBOUND_LEG = "outbound_leg"
    RETURN_LEG = "return_leg"
    FULL_MISSION = "full_mission"


CRITIC_OBSERVATION_NAMES = (
    "position_x",
    "position_y",
    *OBSERVATION_NAMES,
    "payload_mass",
    "payload_offset_body_x",
    "payload_offset_body_y",
    "contact_armed",
    "grounded",
    "decision_fraction_remaining",
    *(f"task_{task.value}" for task in TrainingTask),
)
CRITIC_OBSERVATION_DIM = len(CRITIC_OBSERVATION_NAMES)
CRITIC_OBSERVATION_INDEX = {
    name: index for index, name in enumerate(CRITIC_OBSERVATION_NAMES)
}


DECISION_TIMEOUT = "DECISION_TIMEOUT"
TAKEOFF_RIM_CLEARANCE = "TAKEOFF_RIM_CLEARANCE"


_TASK_DECISION_LIMITS: dict[TrainingTask, int] = {
    TrainingTask.SAMPLE_LANDING: 300,
    TrainingTask.RETURN_LANDING: 300,
    TrainingTask.OUTBOUND_LEG: 1_200,
    TrainingTask.RETURN_LEG: 1_200,
    TrainingTask.FULL_MISSION: 2_400,
}


_TASK_RESET_OPTIONS: dict[TrainingTask, dict[str, Any]] = {
    TrainingTask.SAMPLE_LANDING: {
        "start_phase": "outbound",
        "spawn_mode": "airborne",
        "spawn_reference": "target",
    },
    TrainingTask.RETURN_LANDING: {
        "start_phase": "return",
        "spawn_mode": "airborne",
        "spawn_reference": "target",
    },
    TrainingTask.OUTBOUND_LEG: {
        "start_phase": "outbound",
        "spawn_mode": "ground",
    },
    TrainingTask.RETURN_LEG: {
        "start_phase": "return",
        "spawn_mode": "ground",
    },
    TrainingTask.FULL_MISSION: {
        "start_phase": "full",
        "spawn_mode": "ground",
    },
}

_SAMPLE_SEGMENT_TASKS = frozenset(
    {TrainingTask.SAMPLE_LANDING, TrainingTask.OUTBOUND_LEG}
)
_RETURN_TASKS = frozenset(
    {
        TrainingTask.RETURN_LANDING,
        TrainingTask.RETURN_LEG,
        TrainingTask.FULL_MISSION,
    }
)


def _coerce_task(task: TrainingTask | str) -> TrainingTask:
    try:
        return TrainingTask(task)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in TrainingTask)
        raise ValueError(
            f"unknown training task {task!r}; expected one of: {choices}"
        ) from exc


class SampleReturnTrainingWrapper(gym.Wrapper):
    """Expose agent-rate steps and curriculum boundaries around the mission.

    ``SampleReturnEnv.steps`` continues to count 60 Hz physics frames.  This
    wrapper's ``decision_steps`` counts calls made by the policy.  Rewards from
    all repeated/internal frames are summed without changing the underlying
    mission reward.
    """

    def __init__(
        self,
        env: SampleReturnEnv,
        *,
        task: TrainingTask | str = TrainingTask.FULL_MISSION,
        action_repeat: int = 4,
        auto_advance_sampling: bool = True,
        decision_limit: int | None = None,
    ) -> None:
        super().__init__(env)
        if isinstance(action_repeat, bool) or not isinstance(action_repeat, int):
            raise TypeError("action_repeat must be an integer")
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        if decision_limit is not None:
            if isinstance(decision_limit, bool) or not isinstance(
                decision_limit, int
            ):
                raise TypeError("decision_limit must be an integer")
            if decision_limit < 1:
                raise ValueError("decision_limit must be positive")

        self.default_task = _coerce_task(task)
        self.active_task = self.default_task
        self._decision_limit_override = decision_limit
        self.decision_limit = (
            _TASK_DECISION_LIMITS[self.active_task]
            if decision_limit is None
            else decision_limit
        )
        self.action_repeat = action_repeat
        self.auto_advance_sampling = bool(auto_advance_sampling)
        self.decision_steps = 0
        self._episode_done = False
        self._takeoff_rim_clearance: float | None = None
        self._spawn_rng = np.random.default_rng()
        self._zero_action = np.zeros(self.action_space.shape, self.action_space.dtype)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ACTOR_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.critic_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(CRITIC_OBSERVATION_DIM,),
            dtype=np.float32,
        )

    @property
    def mission_env(self) -> SampleReturnEnv:
        """The wrapped physical environment with a precise public type."""
        return self.env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        options = dict(options or {})
        if seed is not None:
            self._spawn_rng = np.random.default_rng(seed)
        self.active_task = _coerce_task(options.pop("task", self.default_task))
        self.decision_limit = (
            _TASK_DECISION_LIMITS[self.active_task]
            if self._decision_limit_override is None
            else self._decision_limit_override
        )

        takeoff_rim_clearance = options.pop("takeoff_rim_clearance", None)
        self._takeoff_rim_clearance = None
        if takeoff_rim_clearance is not None:
            if self.active_task not in {
                TrainingTask.OUTBOUND_LEG,
                TrainingTask.RETURN_LEG,
            }:
                raise ValueError(
                    "takeoff_rim_clearance requires an outbound_leg or return_leg task"
                )
            if isinstance(takeoff_rim_clearance, (bool, np.bool_)):
                raise TypeError("takeoff_rim_clearance must be a number")
            try:
                takeoff_rim_clearance = float(takeoff_rim_clearance)
            except (TypeError, ValueError) as exc:
                raise TypeError("takeoff_rim_clearance must be a number") from exc
            if not np.isfinite(takeoff_rim_clearance) or takeoff_rim_clearance < 0.0:
                raise ValueError(
                    "takeoff_rim_clearance must be finite and nonnegative"
                )
            self._takeoff_rim_clearance = takeoff_rim_clearance

        sampled_ranges = {}
        sampled_range_options = (
            ("payload_mass", "payload_mass_range", True),
            ("payload_offset_body_x", "payload_offset_body_x_range", False),
            ("payload_offset_body_y", "payload_offset_body_y_range", False),
            ("spawn_theta", "spawn_theta_range", False),
            ("spawn_omega", "spawn_omega_range", False),
        )
        for scalar_name, range_name, must_be_nonnegative in sampled_range_options:
            if scalar_name in options and range_name in options:
                raise ValueError(f"{scalar_name} and {range_name} cannot both be set")

            value_range = options.pop(range_name, None)
            if value_range is None:
                continue
            try:
                bounds = np.asarray(value_range, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{range_name} must contain two numbers") from exc
            if bounds.shape != (2,):
                raise ValueError(f"{range_name} must contain exactly two values")

            lower, upper = map(float, bounds)
            if (
                not np.isfinite(bounds).all()
                or upper < lower
                or (must_be_nonnegative and lower < 0.0)
            ):
                requirement = "finite, nonnegative, and ordered" if must_be_nonnegative else "finite and ordered"
                raise ValueError(f"{range_name} must be {requirement}")
            sampled_ranges[scalar_name] = (lower, upper)

        has_distance = "spawn_distance_from_target" in options
        has_distance_range = "spawn_distance_from_target_range" in options
        has_pad_edge_range = "spawn_pad_edge_distance_range" in options
        has_inside_probability = "spawn_inside_pad_probability" in options
        has_center_probability = "spawn_pad_center_probability" in options
        if (
            has_inside_probability or has_center_probability
        ) and not has_pad_edge_range:
            raise ValueError(
                "pad spawn probabilities require spawn_pad_edge_distance_range"
            )
        if has_pad_edge_range and (
            has_distance
            or has_distance_range
            or "spawn_x_offset" in options
        ):
            raise ValueError(
                "spawn_pad_edge_distance_range cannot be combined with "
                "another horizontal spawn option"
            )
        if has_distance and has_distance_range:
            raise ValueError(
                "spawn_distance_from_target and "
                "spawn_distance_from_target_range cannot both be set"
            )
        distance = options.pop("spawn_distance_from_target", None)
        distance_range = options.pop("spawn_distance_from_target_range", None)
        if distance_range is not None:
            try:
                bounds = np.asarray(distance_range, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "spawn_distance_from_target_range must contain two numbers"
                ) from exc
            if bounds.shape != (2,):
                raise ValueError(
                    "spawn_distance_from_target_range must contain exactly two values"
                )
            lower_distance, upper_distance = map(float, bounds)
            if (
                not np.isfinite(bounds).all()
                or lower_distance < 0.0
                or upper_distance < lower_distance
            ):
                raise ValueError(
                    "spawn_distance_from_target_range must be finite, "
                    "nonnegative, and ordered"
                )
            distance = float(
                self._spawn_rng.uniform(lower_distance, upper_distance)
            )

        pad_edge_range = options.pop("spawn_pad_edge_distance_range", None)
        inside_probability = options.pop("spawn_inside_pad_probability", 1.0 / 3.0)
        center_probability = options.pop("spawn_pad_center_probability", 0.0)
        if pad_edge_range is not None:
            if options.get("spawn_mode", "airborne") != "airborne":
                raise ValueError(
                    "spawn_pad_edge_distance_range requires spawn_mode='airborne'"
                )
            if options.get("spawn_reference", "target") != "target":
                raise ValueError(
                    "spawn_pad_edge_distance_range requires "
                    "spawn_reference='target'"
                )
            try:
                bounds = np.asarray(pad_edge_range, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "spawn_pad_edge_distance_range must contain two numbers"
                ) from exc
            if bounds.shape != (2,):
                raise ValueError(
                    "spawn_pad_edge_distance_range must contain exactly two values"
                )
            lower_edge, upper_edge = map(float, bounds)
            if (
                not np.isfinite(bounds).all()
                or lower_edge < 0.0
                or upper_edge < lower_edge
            ):
                raise ValueError(
                    "spawn_pad_edge_distance_range must be finite, "
                    "nonnegative, and ordered"
                )
            try:
                inside_probability = float(inside_probability)
                center_probability = float(center_probability)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "pad spawn probabilities must be numbers"
                ) from exc
            if (
                not np.isfinite(inside_probability)
                or not 0.0 <= inside_probability <= 1.0
                or not np.isfinite(center_probability)
                or not 0.0 <= center_probability <= 1.0
                or inside_probability + center_probability > 1.0
            ):
                raise ValueError(
                    "pad spawn probabilities must be between zero and one "
                    "and sum to at most one"
                )

            pad_half_width = float(self.mission_env.cfg.pad_half_w)
            maximum_offset = pad_half_width + upper_edge
            returning = self.active_task in {
                TrainingTask.RETURN_LANDING,
                TrainingTask.RETURN_LEG,
            }
            target_x = (
                self.mission_env.base_x if returning else self.mission_env.sample_x
            )
            if (
                target_x - maximum_offset < 0.0
                or target_x + maximum_offset > self.mission_env.cfg.world_w
            ):
                raise ValueError(
                    "symmetric pad-edge spawn range must remain inside the world"
                )

            selector = self._spawn_rng.random()
            side = -1.0 if self._spawn_rng.random() < 0.5 else 1.0
            fraction = self._spawn_rng.random()
            if selector < center_probability:
                spawn_x_offset = 0.0
            elif selector < center_probability + inside_probability:
                spawn_x_offset = (2.0 * fraction - 1.0) * pad_half_width
            else:
                edge_distance = lower_edge + fraction * (upper_edge - lower_edge)
                spawn_x_offset = side * (pad_half_width + edge_distance)

            options["spawn_mode"] = "airborne"
            options["spawn_reference"] = "target"
            options["spawn_x_offset"] = spawn_x_offset

        has_altitude = "spawn_altitude" in options
        has_altitude_range = "spawn_altitude_range" in options
        if has_altitude and has_altitude_range:
            raise ValueError(
                "spawn_altitude and spawn_altitude_range cannot both be set"
            )
        altitude_range = options.pop("spawn_altitude_range", None)
        if altitude_range is not None:
            try:
                bounds = np.asarray(altitude_range, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "spawn_altitude_range must contain two numbers"
                ) from exc
            if bounds.shape != (2,):
                raise ValueError(
                    "spawn_altitude_range must contain exactly two values"
                )
            lower_altitude, upper_altitude = map(float, bounds)
            if (
                not np.isfinite(bounds).all()
                or lower_altitude < 0.0
                or upper_altitude < lower_altitude
            ):
                raise ValueError(
                    "spawn_altitude_range must be finite, nonnegative, and ordered"
                )
            options["spawn_altitude"] = float(
                self._spawn_rng.uniform(lower_altitude, upper_altitude)
            )

        for scalar_name, (lower, upper) in sampled_ranges.items():
            options[scalar_name] = float(self._spawn_rng.uniform(lower, upper))

        if distance is not None:
            if "spawn_x_offset" in options:
                raise ValueError(
                    "spawn_distance_from_target and spawn_x_offset cannot both be set"
                )
            try:
                distance = float(distance)
            except (TypeError, ValueError) as exc:
                raise TypeError("spawn_distance_from_target must be a number") from exc
            if not np.isfinite(distance) or distance < 0.0:
                raise ValueError("spawn_distance_from_target must be finite and nonnegative")
            if options.get("spawn_mode", "airborne") != "airborne":
                raise ValueError("spawn_distance_from_target requires spawn_mode='airborne'")
            if options.get("spawn_reference", "target") != "target":
                raise ValueError(
                    "spawn_distance_from_target requires spawn_reference='target'"
                )

            returning = self.active_task in {
                TrainingTask.RETURN_LANDING,
                TrainingTask.RETURN_LEG,
            }
            options["spawn_mode"] = "airborne"
            options["spawn_reference"] = "target"
            options["spawn_x_offset"] = distance if returning else -distance

        # Task values are defaults. Explicit reset options remain available for
        # controlled spawn perturbations and payload experiments.
        env_options = dict(_TASK_RESET_OPTIONS[self.active_task])
        env_options.update(options)
        if self._takeoff_rim_clearance is not None:
            if str(env_options.get("spawn_mode", "ground")).lower() != "ground":
                raise ValueError(
                    "takeoff_rim_clearance requires spawn_mode='ground'"
                )
            expected_phase = (
                "return"
                if self.active_task == TrainingTask.RETURN_LEG
                else "outbound"
            )
            if str(env_options.get("start_phase", expected_phase)).lower() != expected_phase:
                raise ValueError(
                    "takeoff_rim_clearance start_phase must match the leg task"
                )
        full_observation, info = self.env.reset(seed=seed, options=env_options)

        self.decision_steps = 0
        self._episode_done = False
        return (
            actor_observation(full_observation),
            self._annotated_info(
                info,
                full_observation=full_observation,
                task_success=False,
                decision_timeout=False,
                physics_steps_this_decision=0,
            ),
        )

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._episode_done:
            raise RuntimeError("step() called after episode end; call reset()")

        self.decision_steps += 1
        total_reward = 0.0
        physics_steps = 0
        auto_advanced_sampling_steps = 0
        task_success = False
        takeoff_gate_success = False

        # This also handles an environment restored or manually placed in the
        # non-decision sampling phase before the policy asks for another step.
        if (
            self.auto_advance_sampling
            and self.mission_env.phase == MissionPhase.SAMPLING
        ):
            (
                sampling_reward,
                terminated,
                truncated,
                sampling_steps,
            ) = self._finish_sampling()
            total_reward += sampling_reward
            physics_steps += sampling_steps
            auto_advanced_sampling_steps += sampling_steps
        else:
            terminated = False
            truncated = False

        if physics_steps == 0:
            for _ in range(self.action_repeat):
                phase_before = self.mission_env.phase
                reward, terminated, truncated = self.mission_env._advance_frame(action)
                total_reward += float(reward)
                physics_steps += 1

                if terminated or truncated:
                    break

                if self._takeoff_rim_clearance is not None:
                    body_base, _ = body_endpoints(
                        self.mission_env.state,
                        self.mission_env.cfg,
                        self.mission_env.vehicle,
                    )
                    takeoff_height = (
                        self.mission_env.terrain.spec.rim_y
                        + self._takeoff_rim_clearance
                    )
                    controlled_ascent = (
                        0.0 <= float(self.mission_env.state[VY]) <= self.mission_env.cfg.land_vy_max
                        and abs(float(self.mission_env.state[VX]))
                        <= self.mission_env.cfg.land_vx_max
                        and abs(float(self.mission_env.state[THETA])) <= 0.10
                        and abs(float(self.mission_env.state[OMEGA])) <= 0.10
                    )
                    if (
                        self.mission_env.grounded_pad is None
                        and float(body_base[Y]) >= takeoff_height
                        and controlled_ascent
                    ):
                        total_reward += (
                            self.mission_env.mission_config.sample_landing_reward
                        )
                        task_success = True
                        takeoff_gate_success = True
                        terminated = True
                        break

                if (
                    self.active_task in _SAMPLE_SEGMENT_TASKS
                    and self.mission_env.phase == MissionPhase.SAMPLING
                ):
                    # Reaching the sample pad is a genuine terminal state for
                    # these deliberately shorter curriculum MDPs.
                    task_success = True
                    terminated = True
                    break

                if self.mission_env.phase == MissionPhase.SAMPLING:
                    if self.auto_advance_sampling:
                        (
                            sampling_reward,
                            terminated,
                            truncated,
                            sampling_steps,
                        ) = self._finish_sampling()
                        total_reward += sampling_reward
                        physics_steps += sampling_steps
                        auto_advanced_sampling_steps += sampling_steps
                        break
                    # Actions are ignored during sampling, so ordinary repeat
                    # may continue.  The phase-transition check below stops the
                    # decision before that action can reach loaded dynamics.
                    continue

                if (
                    phase_before == MissionPhase.SAMPLING
                    and self.mission_env.phase != MissionPhase.SAMPLING
                ):
                    break

        if (
            self.active_task in _RETURN_TASKS
            and terminated
            and self.mission_env.outcome == SAMPLE_RETURNED
        ):
            task_success = True

        decision_timeout = False
        if (
            not terminated
            and self.decision_steps >= self.decision_limit
        ):
            total_reward += self.mission_env.mission_config.failure_penalty
            terminated = True
            truncated = False
            decision_timeout = True

        observation, info = self.mission_env._observation_and_info()
        if decision_timeout:
            info["outcome"] = DECISION_TIMEOUT
        elif takeoff_gate_success:
            info["outcome"] = TAKEOFF_RIM_CLEARANCE

        self._episode_done = bool(terminated or truncated)
        return (
            actor_observation(observation),
            float(total_reward),
            bool(terminated),
            bool(truncated),
            self._annotated_info(
                info,
                full_observation=observation,
                task_success=task_success,
                decision_timeout=decision_timeout,
                physics_steps_this_decision=physics_steps,
                auto_advanced_sampling_steps=auto_advanced_sampling_steps,
            ),
        )

    def _finish_sampling(
        self,
    ) -> tuple[float, bool, bool, int]:
        """Advance ignored-action frames and stop at the loaded rest state."""
        reward_sum = 0.0
        physics_steps = 0
        terminated = False
        truncated = False
        while self.mission_env.phase == MissionPhase.SAMPLING:
            reward, terminated, truncated = self.mission_env._advance_frame(self._zero_action)
            reward_sum += float(reward)
            physics_steps += 1
            if terminated or truncated:
                break

        return reward_sum, bool(terminated), bool(truncated), physics_steps

    def _annotated_info(
        self,
        info: dict,
        *,
        full_observation: np.ndarray,
        task_success: bool,
        decision_timeout: bool,
        physics_steps_this_decision: int,
        auto_advanced_sampling_steps: int = 0,
    ) -> dict:
        annotated = dict(info)
        # Gymnasium batches nested info dictionaries recursively.  Omitting an
        # unavailable touchdown keeps its type consistent when only some
        # vector environments have recorded touchdown metrics.
        if annotated.get("sample_touchdown") is None:
            annotated.pop("sample_touchdown", None)
        if annotated.get("return_touchdown") is None:
            annotated.pop("return_touchdown", None)
        annotated.update(
            {
                "task": self.active_task.value,
                "task_success": bool(task_success),
                "decision_steps": self.decision_steps,
                "decision_limit": self.decision_limit,
                "decision_timeout": bool(decision_timeout),
                "physics_steps": int(info.get("steps", self.mission_env.steps)),
                "physics_steps_this_decision": int(physics_steps_this_decision),
                "auto_advanced_sampling_steps": int(
                    auto_advanced_sampling_steps
                ),
                "critic_observation": self.critic_observation(
                    full_observation
                ),
            }
        )
        return annotated

    def critic_observation(
        self,
        full_observation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the privileged simulator state available to the critic."""
        if full_observation is None:
            full_observation = flat_observation(self.mission_env)
        full = np.asarray(full_observation, dtype=np.float32)
        if full.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"expected full observation shape {(OBSERVATION_DIM,)}, "
                f"got {full.shape}"
            )

        env = self.mission_env
        cfg = env.cfg
        payload = env.payload_spec
        task_flags = np.asarray(
            [float(self.active_task == task) for task in TrainingTask],
            dtype=np.float32,
        )
        decision_fraction_remaining = max(
            0.0,
            1.0 - self.decision_steps / self.decision_limit,
        )
        return np.concatenate(
            (
                np.asarray(
                    [env.state[X] / cfg.world_w, env.state[Y] / cfg.world_h],
                    dtype=np.float32,
                ),
                full,
                np.asarray(
                    [
                        payload.mass / cfg.m,
                        payload.offset_body_x / cfg.H,
                        payload.offset_body_y / cfg.H,
                        float(env.mission_state.contact_armed),
                        float(env.grounded_pad is not None),
                        decision_fraction_remaining,
                    ],
                    dtype=np.float32,
                ),
                task_flags,
            )
        ).astype(np.float32, copy=False)


def make_training_env(
    *,
    task: TrainingTask | str = TrainingTask.FULL_MISSION,
    action_repeat: int = 4,
    auto_advance_sampling: bool = True,
    decision_limit: int | None = None,
    config: SampleReturnConfig | None = None,
) -> SampleReturnTrainingWrapper:
    """Construct one headless training adapter around a fresh mission."""
    env = SampleReturnEnv(config)
    if env.observation_space.shape != (OBSERVATION_DIM,):
        raise ValueError(
            "the training policy contract requires the default 16-dimensional "
            "observation (five terrain rays)"
        )
    return SampleReturnTrainingWrapper(
        env,
        task=task,
        action_repeat=action_repeat,
        auto_advance_sampling=auto_advance_sampling,
        decision_limit=decision_limit,
    )


def make_vector_env(
    num_envs: int,
    *,
    task: TrainingTask | str | Sequence[TrainingTask | str] = TrainingTask.FULL_MISSION,
    action_repeat: int = 4,
    auto_advance_sampling: bool = True,
    decision_limit: int | None = None,
    config: SampleReturnConfig | None = None,
    asynchronous: bool = False,
) -> SyncVectorEnv | AsyncVectorEnv:
    """Build vector environments without implicit autoresets.

    ``task`` may be one task shared by every environment or a sequence with
    exactly ``num_envs`` entries.  Disabled autoreset keeps terminal
    observations unambiguous for value bootstrapping in user-owned rollout
    code. Set ``asynchronous`` to run each environment in a worker process.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int):
        raise TypeError("num_envs must be an integer")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")

    if isinstance(task, (TrainingTask, str)):
        tasks = [_coerce_task(task)] * num_envs
    else:
        tasks = [_coerce_task(item) for item in task]
        if len(tasks) != num_envs:
            raise ValueError(
                f"task sequence length {len(tasks)} does not match num_envs {num_envs}"
            )

    env_fns = [
        partial(
            make_training_env,
            task=item,
            action_repeat=action_repeat,
            auto_advance_sampling=auto_advance_sampling,
            decision_limit=decision_limit,
            config=config,
        )
        for item in tasks
    ]
    vector_env_class = AsyncVectorEnv if asynchronous else SyncVectorEnv
    return vector_env_class(
        env_fns,
        autoreset_mode=AutoresetMode.DISABLED,
    )


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "ACTOR_OBSERVATION_NAMES",
    "CRITIC_OBSERVATION_DIM",
    "CRITIC_OBSERVATION_INDEX",
    "CRITIC_OBSERVATION_NAMES",
    "DECISION_TIMEOUT",
    "TAKEOFF_RIM_CLEARANCE",
    "SampleReturnTrainingWrapper",
    "TrainingTask",
    "make_training_env",
    "make_vector_env",
]
