"""Mission-state and Gymnasium tests for crater sample return."""

import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketenv.physics import FUEL, OMEGA, THETA, VX, VY, X, Y
from rocketenv.sample_return import (
    ABORTED,
    CRASHED_OUTBOUND,
    SAMPLE_RETURNED,
    MissionPhase,
    SampleReturnConfig,
    SampleReturnEnv,
    scripted_sample_return_action,
)
from rocketenv.sample_return.vehicle import dry_body_center
from rocketenv.sample_return.reward import failed_landing_reward, flight_reward


def _place_for_touchdown(env: SampleReturnEnv, pad: str, vy: float = -1.0) -> None:
    env.grounded_pad = None
    env.departure_pad = "BASE" if pad == "SAMPLE" else "SAMPLE"
    env.mission_state.contact_armed = True
    rest = env._rest_state(pad, float(env.state[FUEL]), env.vehicle)
    rest[Y] += 0.001
    rest[VY] = vy
    env.state = rest


def test_gymnasium_contract_and_observation_shape():
    env = SampleReturnEnv()
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=42)

    assert observation.shape == (16,)
    assert observation.dtype == np.float32
    assert env.action_space.shape == (2,)
    assert info["phase"] == "OUTBOUND"


@pytest.mark.parametrize("bad_action", [0.5, np.array([0.5])])
def test_action_shape_is_not_silently_broadcast(bad_action):
    env = SampleReturnEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(bad_action)


def test_reward_shaping_honors_config_gamma():
    from rocketenv.sample_return.reward import flight_reward, potential

    env = SampleReturnEnv()
    env.reset(seed=0)
    cfg = dataclasses.replace(env.cfg, gamma=0.9)
    previous = env.state.copy()
    state = previous.copy()
    state[X] += 1.0
    target_y = env.terrain.height_at(env.target_x)

    actual = flight_reward(
        previous,
        state,
        target_x=env.target_x,
        target_y=target_y,
        cfg=cfg,
        fuel_penalty=0.0,
    )
    expected_shaping = cfg.gamma * potential(
        state, env.target_x, target_y, cfg.shaping_k
    ) - potential(previous, env.target_x, target_y, cfg.shaping_k)
    assert actual == pytest.approx(-cfg.step_penalty + expected_shaping)


def test_reset_starts_supported_at_base_with_empty_pod():
    env = SampleReturnEnv()
    env.reset(seed=3)

    body_center = dry_body_center(env.state, env.vehicle)
    assert env.phase == MissionPhase.OUTBOUND
    assert env.grounded_pad == "BASE"
    assert not env.mission_state.contact_armed
    assert not env.has_sample
    assert env.state[FUEL] == env.cfg.fuel_0
    assert body_center[X] == env.base_x
    assert body_center[Y] == pytest.approx(env.terrain.base_y + env.cfg.L)


def test_ground_contact_is_ignored_until_clearance_arms_contact():
    env = SampleReturnEnv()
    env.reset(seed=1)

    _, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert not terminated and not truncated
    assert env.grounded_pad == "BASE"
    assert not env.mission_state.contact_armed

    for _ in range(180):
        _, _, terminated, truncated, _ = env.step(np.array([1.0, 0.0]))
        if env.mission_state.contact_armed:
            break
    assert not terminated and not truncated
    assert env.grounded_pad is None
    assert env.mission_state.contact_armed


def test_hard_fallback_before_arming_is_not_a_contact_bypass():
    env = SampleReturnEnv()
    env.reset(seed=1)
    env.grounded_pad = None
    env.departure_pad = "BASE"
    env.mission_state.contact_armed = False
    env.state = env._rest_state("BASE", 90.0, env.vehicle)
    env.state[Y] += 0.001
    env.state[VY] = -8.0

    previous = env.state.copy()
    _, reward, terminated, _, info = env.step(np.zeros(2))

    assert terminated
    assert info["outcome"] == "CRASHED_OUTBOUND"
    expected_flight_reward = flight_reward(
        previous,
        env.state,
        target_x=env.target_x,
        target_y=env.terrain.height_at(env.target_x),
        cfg=env.cfg,
        fuel_penalty=env.mission_config.fuel_penalty,
    )
    expected_failure_reward = failed_landing_reward(
        env.state,
        target_x=env.target_x,
        cfg=env.cfg,
        vehicle=env.vehicle,
        failure_penalty=env.mission_config.failure_penalty,
    )
    assert reward == pytest.approx(expected_flight_reward + expected_failure_reward)


def test_sample_landing_samples_then_attaches_without_refueling_or_teleport():
    env = SampleReturnEnv()
    env.reset(seed=2)
    fuel_before = 63.0
    env.state[FUEL] = fuel_before
    _place_for_touchdown(env, "SAMPLE")

    _, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert not terminated and not truncated
    assert env.phase == MissionPhase.SAMPLING
    assert not env.mission_state.payload_attached
    fuel_after_touchdown = float(env.state[FUEL])
    body_center_before = dry_body_center(env.state, env.vehicle).copy()

    for _ in range(env.mission_config.sampling_steps):
        env.step(np.array([1.0, 1.0]))  # ignored while sampling

    assert env.phase == MissionPhase.RETURN
    assert env.mission_state.payload_attached and env.has_sample
    assert env.state[FUEL] == pytest.approx(fuel_after_touchdown)
    assert dry_body_center(env.state, env.vehicle) == pytest.approx(body_center_before)
    assert env.target == "BASE"
    assert not env.mission_state.contact_armed


def test_safe_return_with_payload_is_physical_success():
    env = SampleReturnEnv()
    env.reset(seed=4, options={"start_phase": "return"})
    _place_for_touchdown(env, "BASE")

    _, _, terminated, truncated, info = env.step(np.zeros(2))

    assert terminated and not truncated
    assert env.phase == MissionPhase.SUCCESS
    assert info["outcome"] == SAMPLE_RETURNED
    assert info["return_touchdown"] is not None


def test_base_landing_without_sample_is_not_success():
    env = SampleReturnEnv()
    env.reset(seed=5)
    _place_for_touchdown(env, "BASE")

    _, _, terminated, _, info = env.step(np.zeros(2))

    assert terminated
    assert info["outcome"] == ABORTED


def test_non_pad_crash_is_phase_specific():
    env = SampleReturnEnv()
    env.reset(seed=6)
    env.grounded_pad = None
    env.departure_pad = "BASE"
    env.mission_state.contact_armed = True
    x = 65.0
    env.state = np.array(
        [x, env.terrain.height_at(x) + env.cfg.L + 0.001,
         0.0, -6.0, 0.0, 0.0, 80.0],
        dtype=np.float64,
    )

    _, _, terminated, _, info = env.step(np.zeros(2))

    assert terminated
    assert info["outcome"] == CRASHED_OUTBOUND


def test_failed_landing_reward_grades_near_safe_and_destructive_contacts():
    env = SampleReturnEnv()
    env.reset(seed=6)
    near_safe = env._rest_state("SAMPLE", 80.0, env.vehicle)
    near_safe[VY] = -(env.cfg.land_vy_max + 0.1)
    destructive = near_safe.copy()
    destructive[VY] = -30.0
    tipped = near_safe.copy()
    tipped[VY] = -1.0
    tipped[OMEGA] = 10.0

    near_safe_reward = failed_landing_reward(
        near_safe,
        target_x=env.sample_x,
        cfg=env.cfg,
        vehicle=env.vehicle,
        failure_penalty=env.mission_config.failure_penalty,
    )
    destructive_reward = failed_landing_reward(
        destructive,
        target_x=env.sample_x,
        cfg=env.cfg,
        vehicle=env.vehicle,
        failure_penalty=env.mission_config.failure_penalty,
    )
    tipped_reward = failed_landing_reward(
        tipped,
        target_x=env.sample_x,
        cfg=env.cfg,
        vehicle=env.vehicle,
        failure_penalty=env.mission_config.failure_penalty,
    )

    assert near_safe_reward > destructive_reward
    assert destructive_reward < env.mission_config.failure_penalty + 10.0
    assert tipped_reward > destructive_reward
    assert destructive_reward >= env.mission_config.failure_penalty
    assert near_safe_reward <= (
        env.mission_config.failure_penalty + env.cfg.partial_credit
    )

    vertical_speeds = (5.1, 10.0, 20.0, 30.0)
    vertical_rewards = []
    for speed in vertical_speeds:
        state = env._rest_state("SAMPLE", 80.0, env.vehicle)
        state[VY] = -speed
        vertical_rewards.append(
            failed_landing_reward(
                state,
                target_x=env.sample_x,
                cfg=env.cfg,
                vehicle=env.vehicle,
                failure_penalty=env.mission_config.failure_penalty,
            )
        )
    assert all(
        better > worse
        for better, worse in zip(vertical_rewards, vertical_rewards[1:])
    )


def test_failed_return_keeps_base_as_display_target():
    env = SampleReturnEnv()
    env.reset(seed=6, options={"start_phase": "return"})
    env.grounded_pad = None
    env.departure_pad = "SAMPLE"
    env.mission_state.contact_armed = True
    x = 90.0
    body_y = env.terrain.height_at(x) + env.cfg.L + 0.001
    env.state = env._rest_state("SAMPLE", 80.0, env.vehicle)
    env.state[X] = x + env.vehicle.com_offset_body[0]
    env.state[Y] = body_y
    env.state[VY] = -6.0

    _, _, terminated, _, info = env.step(np.zeros(2))

    assert terminated
    assert info["outcome"] == "CRASHED_RETURN"
    assert info["target"] == "BASE"


def test_timeout_is_truncation():
    base = SampleReturnConfig()
    short = dataclasses.replace(
        base, flight=dataclasses.replace(base.flight, max_steps=1)
    )
    env = SampleReturnEnv(short)
    env.reset(seed=0)

    _, _, terminated, truncated, info = env.step(np.zeros(2))

    assert truncated and not terminated
    assert info["outcome"] == "TIMEOUT"


def test_burning_last_fuel_while_supported_fails_cleanly():
    env = SampleReturnEnv()
    env.reset(seed=0)
    env.state[FUEL] = env.cfg.burn_rate * 0.2 * env.cfg.dt * 0.5

    _, reward, terminated, truncated, info = env.step(np.array([0.2, 0.0]))

    assert terminated and not truncated
    assert info["outcome"] == "OUT_OF_FUEL"
    assert np.isfinite(reward)


def test_return_curriculum_reset_has_loaded_vehicle():
    env = SampleReturnEnv()
    observation, _ = env.reset(
        seed=7,
        options={
            "start_phase": "return",
            "payload_mass": 0.4,
            "payload_offset_body_x": 0.7,
            "spawn_mode": "airborne",
        },
    )

    assert env.phase == MissionPhase.RETURN
    assert env.grounded_pad is None
    assert env.mission_state.contact_armed
    assert env.vehicle.total_mass == pytest.approx(env.cfg.m + 0.4)
    assert observation[-3:].tolist() == pytest.approx([1.0, 0.0, 1.0])


def test_target_relative_airborne_spawn_exposes_explicit_curriculum_state():
    env = SampleReturnEnv()
    env.reset(
        seed=8,
        options={
            "start_phase": "outbound",
            "spawn_mode": "airborne",
            "spawn_reference": "target",
            "spawn_altitude": 12.0,
            "spawn_x_offset": -2.5,
            "spawn_vx": 1.25,
            "spawn_vy": -0.75,
            "spawn_theta": 0.1,
            "spawn_omega": -0.2,
        },
    )

    body_center = dry_body_center(env.state, env.vehicle)
    expected_x = env.sample_x - 2.5
    assert body_center == pytest.approx(
        [
            expected_x,
            env.terrain.height_at(expected_x) + env.vehicle.body_height / 2.0 + 12.0,
        ]
    )
    assert env.state[[VX, VY, THETA, OMEGA]] == pytest.approx(
        [1.25, -0.75, 0.1, -0.2]
    )
    info = env._info()
    assert info["max_tilt"] == pytest.approx(0.1)
    assert info["max_angular_velocity"] == pytest.approx(0.2)
    assert env.mission_state.contact_armed
    assert env.target == "SAMPLE"


def test_loaded_target_spawn_positions_the_dry_body_not_the_shifted_com():
    env = SampleReturnEnv()
    env.reset(
        seed=9,
        options={
            "start_phase": "return",
            "spawn_mode": "airborne",
            "spawn_reference": "target",
            "spawn_altitude": 10.0,
            "spawn_theta": 0.2,
        },
    )

    body_center = dry_body_center(env.state, env.vehicle)
    assert body_center == pytest.approx(
        [
            env.base_x,
            env.terrain.base_y + env.vehicle.body_height / 2.0 + 10.0,
        ]
    )
    assert env.state[[X, Y]] != pytest.approx(body_center)


def test_airborne_spawn_can_use_the_crater_rim_as_its_altitude_reference():
    env = SampleReturnEnv()

    for offset in (-12.0, 12.0):
        env.reset(
            seed=10,
            options={
                "start_phase": "outbound",
                "spawn_mode": "airborne",
                "spawn_reference": "target",
                "spawn_altitude_reference": "crater_rim",
                "spawn_altitude": 5.0,
                "spawn_x_offset": offset,
            },
        )
        body_center = dry_body_center(env.state, env.vehicle)
        body_bottom = body_center[Y] - env.vehicle.body_height / 2.0
        assert body_bottom == pytest.approx(env.terrain.spec.rim_y + 5.0)

    with pytest.raises(ValueError, match="spawn_altitude_reference"):
        env.reset(
            seed=10,
            options={
                "spawn_mode": "airborne",
                "spawn_altitude_reference": "sky",
            },
        )


def test_airborne_spawn_options_are_rejected_for_ground_start():
    env = SampleReturnEnv()
    with pytest.raises(ValueError, match="require spawn_mode"):
        env.reset(seed=10, options={"spawn_altitude": 5.0})


def test_seed_42_scripted_controller_completes_same_tank_mission():
    env = SampleReturnEnv()
    env.reset(seed=42)
    previous_fuel = float(env.state[FUEL])

    for _ in range(env.cfg.max_steps):
        _, _, terminated, truncated, info = env.step(
            scripted_sample_return_action(env)
        )
        assert env.state[FUEL] <= previous_fuel + 1e-12
        previous_fuel = float(env.state[FUEL])
        if terminated or truncated:
            break

    assert terminated and not truncated
    assert info["outcome"] == SAMPLE_RETURNED
    assert env.state[FUEL] > 0.0
    assert "SAMPLE ACQUIRED" in info["events"]
    assert info["payload_offset_body"] == pytest.approx([0.8, 0.0])
    assert info["configured_payload_mass"] == pytest.approx(0.35)
    assert info["outbound_fuel_used"] > 0.0
    assert info["return_fuel_used"] > 0.0
