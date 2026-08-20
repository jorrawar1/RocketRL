"""Gymnasium environment for one continuous crater sample-return mission."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..physics import FUEL, OMEGA, STATE_DIM, THETA, VX, VY, X, Y
from .config import DEFAULT_SAMPLE_RETURN_CONFIG, SampleReturnConfig
from .observation import flat_observation, ray_distances, structured_observation
from .physics import step_dynamics
from .reward import failed_landing_reward, flight_reward, landing_result
from .terrain import CraterSampleTerrain
from .mission_types import MissionPhase, SampleReturnState
from .vehicle import (
    VehicleModel,
    body_endpoints,
    body_to_world,
    dry_body_center,
    state_with_preserved_dry_body_center,
)

SAMPLE_RETURNED = "SAMPLE_RETURNED"
CRASHED_OUTBOUND = "CRASHED_OUTBOUND"
CRASHED_RETURN = "CRASHED_RETURN"
TIPPED_OUTBOUND = "TIPPED_OUTBOUND"
TIPPED_RETURN = "TIPPED_RETURN"
OUT_OF_FUEL = "OUT_OF_FUEL"
OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
ABORTED = "ABORTED"
TIMEOUT = "TIMEOUT"


class SampleReturnEnv(gym.Env):
    """Base -> crater sample -> base on one map and one fuel tank.

    The seven-element physical state stores the active vehicle's combined COM.
    Mission bookkeeping stays in :class:`SampleReturnState`.  The rigid pod is
    intentionally excluded from terrain collision in the MVP; normal upright
    landings keep it well clear, while dry-body base/tip contact matches the
    original simulator exactly.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: SampleReturnConfig | None = None):
        super().__init__()
        self.mission_config = config or DEFAULT_SAMPLE_RETURN_CONFIG
        self.mission_config.validate()
        self.cfg = self.mission_config.flight
        self.payload_spec = self.mission_config.payload
        self.dry_vehicle = VehicleModel.from_config(self.cfg)
        self.loaded_vehicle = VehicleModel.from_config(self.cfg, self.payload_spec)
        self.vehicle = self.dry_vehicle

        self.base_x = self.mission_config.terrain.base_x
        self.sample_x = self.mission_config.terrain.sample_x
        self.terrain = CraterSampleTerrain.from_seed(
            0, self.mission_config.terrain
        )
        self.state = np.zeros(STATE_DIM, dtype=np.float64)
        self.mission_state = SampleReturnState(
            MissionPhase.OUTBOUND, False, False, False, 0
        )
        self.grounded_pad: str | None = "BASE"
        self.departure_pad: str | None = "BASE"
        self.steps = 0
        self.mission_seed = 0
        self.events: list[dict[str, Any]] = []
        self.last_action = np.zeros(2, dtype=np.float64)
        self._sampling_total = self.mission_config.sampling_steps
        self._max_tilt = 0.0
        self._max_angular_velocity = 0.0
        self._sample_touchdown: dict[str, float] | None = None
        self._return_touchdown: dict[str, float] | None = None
        self._outbound_fuel_used: float | None = None
        self._return_fuel_used: float | None = None
        self._return_start_fuel: float | None = None
        self._episode_done = False

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11 + self.cfg.n_rays,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ views
    @property
    def phase(self) -> MissionPhase:
        return self.mission_state.phase

    @property
    def outcome(self) -> str | None:
        return self.mission_state.outcome

    @property
    def has_sample(self) -> bool:
        return self.mission_state.sample_collected

    @property
    def launch_armed(self) -> bool:
        """Compatibility name used by the original prototype renderer."""
        return self.mission_state.contact_armed

    @property
    def target(self) -> str:
        return (
            "BASE"
            if self.mission_state.sample_collected
            or self.phase in (MissionPhase.RETURN, MissionPhase.SUCCESS)
            else "SAMPLE"
        )

    @property
    def target_x(self) -> float:
        return self.base_x if self.target == "BASE" else self.sample_x

    @property
    def payload_mass(self) -> float:
        return self.payload_spec.mass if self.mission_state.payload_attached else 0.0

    @property
    def sampling_progress(self) -> float:
        if self.mission_state.payload_attached:
            return 1.0
        if self.phase != MissionPhase.SAMPLING:
            return 0.0
        return 1.0 - (
            self.mission_state.sampling_steps_remaining / self._sampling_total
        )

    def ray_distances(self) -> np.ndarray:
        return ray_distances(self)

    def structured_observation(self) -> dict[str, float | np.ndarray]:
        return structured_observation(self)

    # --------------------------------------------------------------- Gym API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self.mission_seed = int(seed)
        self.terrain = CraterSampleTerrain.from_seed(
            self.mission_seed, self.mission_config.terrain
        )

        options = dict(options or {})
        allowed = {
            "start_phase",
            "payload_mass",
            "payload_offset_body_x",
            "payload_offset_body_y",
            "spawn_mode",
            "spawn_reference",
            "spawn_altitude_reference",
            "spawn_altitude",
            "spawn_x_offset",
            "spawn_vx",
            "spawn_vy",
            "spawn_theta",
            "spawn_omega",
        }
        unknown = set(options) - allowed
        if unknown:
            raise KeyError(f"unknown reset options: {sorted(unknown)}")

        start_phase = str(options.get("start_phase", "full")).lower()
        if start_phase not in {"full", "outbound", "return"}:
            raise ValueError("start_phase must be 'full', 'outbound', or 'return'")
        spawn_mode = str(options.get("spawn_mode", "ground")).lower()
        if spawn_mode not in {"ground", "airborne"}:
            raise ValueError("spawn_mode must be 'ground' or 'airborne'")
        spawn_reference = str(options.get("spawn_reference", "departure")).lower()
        if spawn_reference not in {"departure", "target"}:
            raise ValueError("spawn_reference must be 'departure' or 'target'")
        spawn_altitude_reference = str(
            options.get("spawn_altitude_reference", "terrain")
        ).lower()
        if spawn_altitude_reference not in {"terrain", "crater_rim"}:
            raise ValueError(
                "spawn_altitude_reference must be 'terrain' or 'crater_rim'"
            )

        spawn_values = {
            "spawn_altitude": float(options.get("spawn_altitude", 18.0)),
            "spawn_x_offset": float(options.get("spawn_x_offset", 0.0)),
            "spawn_vx": float(options.get("spawn_vx", 0.0)),
            "spawn_vy": float(options.get("spawn_vy", 0.0)),
            "spawn_theta": float(options.get("spawn_theta", 0.0)),
            "spawn_omega": float(options.get("spawn_omega", 0.0)),
        }
        if not all(math.isfinite(value) for value in spawn_values.values()):
            raise ValueError("airborne spawn values must be finite")
        if spawn_values["spawn_altitude"] < 0.0:
            raise ValueError("spawn_altitude cannot be negative")
        airborne_only = {
            "spawn_reference",
            "spawn_altitude_reference",
            "spawn_altitude",
            "spawn_x_offset",
            "spawn_vx",
            "spawn_vy",
            "spawn_theta",
            "spawn_omega",
        }
        if spawn_mode == "ground" and airborne_only.intersection(options):
            raise ValueError("airborne spawn options require spawn_mode='airborne'")

        payload_updates = {
            key: float(options[key])
            for key in (
                "payload_mass",
                "payload_offset_body_x",
                "payload_offset_body_y",
            )
            if key in options
        }
        rename = {
            "payload_mass": "mass",
            "payload_offset_body_x": "offset_body_x",
            "payload_offset_body_y": "offset_body_y",
        }
        payload_kwargs = {rename[key]: value for key, value in payload_updates.items()}
        self.payload_spec = dataclasses.replace(
            self.mission_config.payload, **payload_kwargs
        )
        self.dry_vehicle = VehicleModel.from_config(self.cfg)
        self.loaded_vehicle = VehicleModel.from_config(self.cfg, self.payload_spec)

        returning = start_phase == "return"
        self.vehicle = self.loaded_vehicle if returning else self.dry_vehicle
        phase = MissionPhase.RETURN if returning else MissionPhase.OUTBOUND
        self.mission_state = SampleReturnState(
            phase=phase,
            payload_attached=returning,
            sample_collected=returning,
            contact_armed=spawn_mode == "airborne",
            sampling_steps_remaining=0,
        )
        departure_pad = "SAMPLE" if returning else "BASE"
        if spawn_mode == "airborne":
            target_pad = "BASE" if returning else "SAMPLE"
            reference_pad = (
                target_pad if spawn_reference == "target" else departure_pad
            )
            reference_x = self.base_x if reference_pad == "BASE" else self.sample_x
            body_x = reference_x + spawn_values["spawn_x_offset"]
            if not 0.0 <= body_x <= self.cfg.world_w:
                raise ValueError("airborne spawn x must remain inside the world")
            theta = spawn_values["spawn_theta"]
            altitude_reference_y = (
                self.terrain.height_at(body_x)
                if spawn_altitude_reference == "terrain"
                else self.terrain.spec.rim_y
            )
            body_center = np.array(
                [
                    body_x,
                    altitude_reference_y + self.vehicle.body_height / 2.0
                    + spawn_values["spawn_altitude"],
                ],
                dtype=np.float64,
            )
            com = body_center + body_to_world(
                self.vehicle.com_offset_body, theta
            )
            self.state = np.array(
                [
                    com[X],
                    com[Y],
                    spawn_values["spawn_vx"],
                    spawn_values["spawn_vy"],
                    theta,
                    spawn_values["spawn_omega"],
                    self.cfg.fuel_0,
                ],
                dtype=np.float64,
            )
            self.grounded_pad = None
            self.departure_pad = departure_pad
        else:
            self.state = self._rest_state(
                departure_pad, self.cfg.fuel_0, self.vehicle
            )
            self.grounded_pad = departure_pad
            self.departure_pad = departure_pad

        self.steps = 0
        self.events = []
        self.last_action = np.zeros(2, dtype=np.float64)
        self._max_tilt = abs(float(self.state[THETA]))
        self._max_angular_velocity = abs(float(self.state[OMEGA]))
        self._sample_touchdown = None
        self._return_touchdown = None
        self._outbound_fuel_used = 0.0 if returning else None
        self._return_fuel_used = None
        self._return_start_fuel = float(self.state[FUEL]) if returning else None
        self._episode_done = False
        return flat_observation(self), self._info()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward, terminated, truncated = self._advance_frame(action)
        observation, info = self._observation_and_info()
        return observation, reward, terminated, truncated, info

    def _advance_frame(self, action) -> tuple[float, bool, bool]:
        """Advance one physics frame without constructing the observation view."""
        if self._episode_done:
            raise RuntimeError("step() called after episode end; call reset()")

        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (2,):
            raise ValueError("action must have shape (2,)")
        action_array = np.clip(action_array, [0.0, -1.0], [1.0, 1.0])
        self.last_action = action_array.copy()
        self.steps += 1
        reward = -self.cfg.step_penalty
        terminated = False
        truncated = False

        if self.phase == MissionPhase.SAMPLING:
            reward = self._step_sampling()
        elif self.grounded_pad is not None:
            reward = self._step_supported(action_array)
        else:
            reward, terminated = self._step_airborne(action_array)

        self._max_tilt = max(self._max_tilt, abs(float(self.state[THETA])))
        self._max_angular_velocity = max(
            self._max_angular_velocity, abs(float(self.state[OMEGA]))
        )

        if not terminated and self.state[FUEL] <= 0.0:
            terminated = self._fail(OUT_OF_FUEL)
            reward += self.mission_config.failure_penalty
        if not terminated and self.steps >= self.cfg.max_steps:
            truncated = True
            self.mission_state.outcome = TIMEOUT
        self._episode_done = terminated or truncated
        return float(reward), terminated, truncated

    def _observation_and_info(self) -> tuple[np.ndarray, dict]:
        """Materialize the Gymnasium view of the current physical state."""
        return flat_observation(self), self._info()

    # --------------------------------------------------------------- dynamics
    def _step_supported(self, action: np.ndarray) -> float:
        throttle = float(action[0])
        phi = float(action[1]) * self.cfg.phi_max
        available_thrust = (
            throttle * self.cfg.T_max * self.cfg.thrust_multiplier
            if self.state[FUEL] > 0.0
            else 0.0
        )
        vertical_thrust = available_thrust * np.cos(float(self.state[THETA]) + phi)
        can_lift = vertical_thrust > self.vehicle.total_mass * self.cfg.g

        previous = self.state.copy()
        if can_lift:
            departure = self.grounded_pad
            self.state = step_dynamics(self.state, action, self.cfg, self.vehicle)
            self.grounded_pad = None
            self.departure_pad = departure
            self.mission_state.contact_armed = False
            self._post_event(
                "RETURN DEPARTURE" if self.phase == MissionPhase.RETURN else "BASE DEPARTURE"
            )
        else:
            # The pad reaction holds the settled pose. Firing still consumes
            # propellant, so an undersized launch command is not free.
            self.state[FUEL] = max(
                0.0,
                self.state[FUEL] - self.cfg.burn_rate * throttle * self.cfg.dt,
            )

        # Fuel exhaustion is resolved centrally after this helper returns.
        if self.state[FUEL] <= 0.0:
            return -self.cfg.step_penalty

        target_y = self.terrain.height_at(self.target_x)
        return flight_reward(
            previous,
            self.state,
            target_x=self.target_x,
            target_y=target_y,
            cfg=self.cfg,
            fuel_penalty=self.mission_config.fuel_penalty,
        )

    def _step_airborne(self, action: np.ndarray) -> tuple[float, bool]:
        previous = self.state.copy()
        target_x = self.target_x
        target_y = self.terrain.height_at(target_x)
        self.state = step_dynamics(previous, action, self.cfg, self.vehicle)
        reward = flight_reward(
            previous,
            self.state,
            target_x=target_x,
            target_y=target_y,
            cfg=self.cfg,
            fuel_penalty=self.mission_config.fuel_penalty,
        )

        self._update_contact_arming()
        if self._out_of_bounds():
            return reward + self.mission_config.failure_penalty, self._fail(OUT_OF_BOUNDS)
        if not self._body_contact():
            return reward, False

        # Before arming, a short hop that settles back on the departure pad is
        # treated as supported contact.  Once clearance/envelope is crossed,
        # every terrain touch goes through normal crash/landing resolution.
        if not self.mission_state.contact_armed and self._inside_departure_pad():
            assert self.departure_pad is not None
            departure_x = (
                self.base_x if self.departure_pad == "BASE" else self.sample_x
            )
            safe, contact_kind = landing_result(
                self.state,
                target_x=departure_x,
                cfg=self.cfg,
                vehicle=self.vehicle,
            )
            if not safe:
                outcome = self._phase_failure(contact_kind)
                return (
                    reward
                    + failed_landing_reward(
                        self.state,
                        target_x=target_x,
                        cfg=self.cfg,
                        vehicle=self.vehicle,
                        failure_penalty=self.mission_config.failure_penalty,
                    ),
                    self._fail(outcome),
                )
            fuel = float(self.state[FUEL])
            self.state = self._rest_state(self.departure_pad, fuel, self.vehicle)
            self.grounded_pad = self.departure_pad
            return reward, False

        contacted_pad = self._contacted_pad()
        if contacted_pad is None:
            outcome = self._phase_failure("CRASHED")
            return (
                reward
                + failed_landing_reward(
                    self.state,
                    target_x=target_x,
                    cfg=self.cfg,
                    vehicle=self.vehicle,
                    failure_penalty=self.mission_config.failure_penalty,
                ),
                self._fail(outcome),
            )

        pad_x = self.base_x if contacted_pad == "BASE" else self.sample_x
        safe, contact_kind = landing_result(
            self.state, target_x=pad_x, cfg=self.cfg, vehicle=self.vehicle
        )
        if not safe:
            outcome = self._phase_failure(contact_kind)
            return (
                reward
                + failed_landing_reward(
                    self.state,
                    target_x=target_x,
                    cfg=self.cfg,
                    vehicle=self.vehicle,
                    failure_penalty=self.mission_config.failure_penalty,
                ),
                self._fail(outcome),
            )
        if contacted_pad != self.target:
            return reward + self.mission_config.failure_penalty, self._fail(ABORTED)

        touchdown = {
            "vx": float(self.state[VX]),
            "vy": float(self.state[VY]),
            "theta": float(self.state[THETA]),
            "omega": float(self.state[OMEGA]),
        }
        fuel = float(self.state[FUEL])
        self.state = self._rest_state(contacted_pad, fuel, self.vehicle)
        self.grounded_pad = contacted_pad
        self.departure_pad = contacted_pad
        self.mission_state.contact_armed = False

        if self.phase == MissionPhase.OUTBOUND:
            self._sample_touchdown = touchdown
            self._outbound_fuel_used = self.cfg.fuel_0 - fuel
            self.mission_state.phase = MissionPhase.SAMPLING
            self.mission_state.sampling_steps_remaining = self._sampling_total
            self._post_event("SAMPLE PAD TOUCHDOWN")
            self._post_event("SAMPLE ACQUISITION")
            return reward + self.mission_config.sample_landing_reward, False

        self._return_touchdown = touchdown
        if self._return_start_fuel is not None:
            self._return_fuel_used = self._return_start_fuel - fuel
        self.mission_state.phase = MissionPhase.SUCCESS
        self.mission_state.outcome = SAMPLE_RETURNED
        self._post_event("BASE TOUCHDOWN")
        fuel_bonus = self.cfg.reward_fuel_bonus * (self.state[FUEL] / self.cfg.fuel_0)
        return reward + self.mission_config.sample_returned_reward + fuel_bonus, True

    def _step_sampling(self) -> float:
        self.mission_state.sampling_steps_remaining -= 1
        if self.mission_state.sampling_steps_remaining > 0:
            return -self.cfg.step_penalty

        old_vehicle = self.vehicle
        fuel = float(self.state[FUEL])
        self.state[[VX, VY, OMEGA]] = 0.0
        self.state[THETA] = 0.0
        self.vehicle = self.loaded_vehicle
        self.state = state_with_preserved_dry_body_center(
            self.state, old_vehicle, self.vehicle
        )
        # Re-assert the exact rest pose after the COM convention switches.
        body_center = dry_body_center(self.state, self.vehicle)
        body_center[X] = self.sample_x
        body_center[Y] = self.terrain.sample_y + self.vehicle.body_height / 2.0
        self.state[[X, Y]] = body_center + body_to_world(
            self.vehicle.com_offset_body, 0.0
        )
        self.state[FUEL] = fuel

        self.mission_state.phase = MissionPhase.RETURN
        self.mission_state.payload_attached = True
        self.mission_state.sample_collected = True
        self.mission_state.contact_armed = False
        self.mission_state.sampling_steps_remaining = 0
        self.grounded_pad = "SAMPLE"
        self.departure_pad = "SAMPLE"
        self._return_start_fuel = fuel
        self._post_event("SAMPLE ACQUIRED")
        self._post_event("PAYLOAD LOCKED")
        return self.mission_config.sample_acquired_reward

    # --------------------------------------------------------------- geometry
    def _rest_state(
        self, pad: str, fuel: float, vehicle: VehicleModel
    ) -> np.ndarray:
        x = self.base_x if pad == "BASE" else self.sample_x
        ground_y = self.terrain.height_at(x)
        body_center = np.array(
            [x, ground_y + vehicle.body_height / 2.0], dtype=np.float64
        )
        com = body_center + body_to_world(vehicle.com_offset_body, 0.0)
        return np.array(
            [com[X], com[Y], 0.0, 0.0, 0.0, 0.0, fuel], dtype=np.float64
        )

    def _body_contact(self) -> bool:
        base, tip = body_endpoints(self.state, self.cfg, self.vehicle)
        return (
            base[Y] <= self.terrain.height_at(float(base[X]))
            or tip[Y] <= self.terrain.height_at(float(tip[X]))
        )

    def _update_contact_arming(self) -> None:
        if self.mission_state.contact_armed or self.departure_pad is None:
            return
        base, _ = body_endpoints(self.state, self.cfg, self.vehicle)
        clearance = float(base[Y] - self.terrain.height_at(float(base[X])))
        departure_x = self.base_x if self.departure_pad == "BASE" else self.sample_x
        if (
            clearance > self.mission_config.contact_arm_clearance
            or abs(float(dry_body_center(self.state, self.vehicle)[X]) - departure_x)
            > self.mission_config.departure_radius
        ):
            self.mission_state.contact_armed = True

    def _inside_departure_pad(self) -> bool:
        if self.departure_pad is None:
            return False
        x = self.base_x if self.departure_pad == "BASE" else self.sample_x
        return abs(float(dry_body_center(self.state, self.vehicle)[X]) - x) < self.cfg.pad_half_w

    def _contacted_pad(self) -> str | None:
        body_x = float(dry_body_center(self.state, self.vehicle)[X])
        if abs(body_x - self.base_x) < self.cfg.pad_half_w:
            return "BASE"
        if abs(body_x - self.sample_x) < self.cfg.pad_half_w:
            return "SAMPLE"
        return None

    def _out_of_bounds(self) -> bool:
        x, y = map(float, self.state[[X, Y]])
        return x < 0.0 or x > self.cfg.world_w or y > self.cfg.world_h

    # --------------------------------------------------------------- outcomes
    def _phase_failure(self, kind: str) -> str:
        suffix = "RETURN" if self.phase == MissionPhase.RETURN else "OUTBOUND"
        return f"{kind}_{suffix}"

    def _fail(self, outcome: str) -> bool:
        self.mission_state.phase = MissionPhase.FAILURE
        self.mission_state.outcome = outcome
        self._post_event(outcome.replace("_", " "))
        return True

    def _post_event(self, label: str) -> None:
        self.events.append(
            {"step": self.steps, "t": self.steps * self.cfg.dt, "label": label}
        )

    def _info(self) -> dict:
        return {
            "state": self.state.copy(),
            "steps": self.steps,
            "phase": self.phase.name,
            "has_sample": self.has_sample,
            "payload_attached": self.mission_state.payload_attached,
            "payload_mass": self.payload_mass,
            "configured_payload_mass": float(self.payload_spec.mass),
            "payload_offset_body": np.array(
                [self.payload_spec.offset_body_x, self.payload_spec.offset_body_y],
                dtype=np.float64,
            ),
            "target": self.target,
            "target_x": self.target_x,
            "grounded": self.grounded_pad is not None,
            "contact_armed": self.mission_state.contact_armed,
            "sampling_progress": self.sampling_progress,
            "fuel_remaining": float(self.state[FUEL]),
            "max_tilt": self._max_tilt,
            "max_angular_velocity": self._max_angular_velocity,
            "sample_touchdown": self._sample_touchdown,
            "return_touchdown": self._return_touchdown,
            "outbound_fuel_used": self._outbound_fuel_used,
            "return_fuel_used": self._return_fuel_used,
            "outcome": self.outcome,
            "events": tuple(event["label"] for event in self.events),
        }


__all__ = [
    "ABORTED",
    "CRASHED_OUTBOUND",
    "CRASHED_RETURN",
    "OUT_OF_BOUNDS",
    "OUT_OF_FUEL",
    "SAMPLE_RETURNED",
    "SampleReturnEnv",
    "TIMEOUT",
    "TIPPED_OUTBOUND",
    "TIPPED_RETURN",
]
