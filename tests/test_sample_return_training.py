"""Training-rate and curriculum adapters for crater sample return."""

import dataclasses

import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from rocketenv.physics import FUEL, OMEGA, THETA, VX, VY, X, Y
from rocketenv.sample_return import MissionPhase, SampleReturnConfig, SampleReturnEnv
from rocketenv.sample_return.training import (
    CRITIC_OBSERVATION_DIM,
    CRITIC_OBSERVATION_INDEX,
    CRITIC_OBSERVATION_NAMES,
    DECISION_TIMEOUT,
    TAKEOFF_RIM_CLEARANCE,
    SampleReturnTrainingWrapper,
    TrainingTask,
    make_training_env,
    make_vector_env,
)
from rocketenv.sample_return.observation import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_INDEX,
    ACTOR_OBSERVATION_NAMES,
    OBSERVATION_DIM,
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
    actor_observation,
    flat_observation,
)
from rocketenv.sample_return.vehicle import body_endpoints, dry_body_center


def _place_for_touchdown(env: SampleReturnEnv, pad: str, vy: float = -1.0) -> None:
    env.grounded_pad = None
    env.departure_pad = "BASE" if pad == "SAMPLE" else "SAMPLE"
    env.mission_state.contact_armed = True
    rest = env._rest_state(pad, float(env.state[FUEL]), env.vehicle)
    rest[Y] += 0.001
    rest[VY] = vy
    env.state = rest


def test_action_repeat_matches_manual_physics_steps():
    wrapper = make_training_env(action_repeat=4, auto_advance_sampling=False)
    manual = SampleReturnEnv()
    wrapper.reset(seed=14)
    manual.reset(
        seed=14,
        options={"start_phase": "full", "spawn_mode": "ground"},
    )
    action = np.array([1.0, 0.15], dtype=np.float32)

    observation, reward, terminated, truncated, info = wrapper.step(action)
    manual_reward = 0.0
    for _ in range(4):
        manual_observation, one_reward, manual_terminated, manual_truncated, _ = (
            manual.step(action)
        )
        manual_reward += one_reward
        if manual_terminated or manual_truncated:
            break

    np.testing.assert_allclose(wrapper.mission_env.state, manual.state)
    np.testing.assert_allclose(observation, actor_observation(manual_observation))
    assert reward == pytest.approx(manual_reward)
    assert terminated == manual_terminated
    assert truncated == manual_truncated
    assert info["decision_steps"] == 1
    assert info["physics_steps"] == 4
    assert info["physics_steps_this_decision"] == 4
    assert not info["task_success"]


def test_repeated_frames_materialize_one_final_observation(monkeypatch):
    wrapper = make_training_env(action_repeat=4, auto_advance_sampling=True)
    wrapper.reset(seed=14)
    view_calls = 0
    original_view = wrapper.mission_env._observation_and_info

    def counted_view():
        nonlocal view_calls
        view_calls += 1
        return original_view()

    monkeypatch.setattr(wrapper.mission_env, "_observation_and_info", counted_view)

    wrapper.step(np.zeros(2))
    assert view_calls == 1

    wrapper.reset(seed=14)
    _place_for_touchdown(wrapper.mission_env, "SAMPLE")
    view_calls = 0
    wrapper.step(np.array([1.0, 0.0]))
    assert view_calls == 1


def test_repeat_stops_on_timeout_and_requires_reset_afterward():
    base = SampleReturnConfig()
    short = dataclasses.replace(
        base,
        flight=dataclasses.replace(base.flight, max_steps=2),
    )
    wrapper = make_training_env(config=short, action_repeat=5)
    wrapper.reset(seed=0)

    _, _, terminated, truncated, info = wrapper.step(np.zeros(2))

    assert not terminated and truncated
    assert info["physics_steps"] == 2
    assert info["physics_steps_this_decision"] == 2
    with pytest.raises(RuntimeError, match="call reset"):
        wrapper.step(np.zeros(2))


def test_decision_deadline_is_a_terminal_failure():
    wrapper = make_training_env(
        task=TrainingTask.SAMPLE_LANDING,
        action_repeat=1,
        decision_limit=2,
    )
    wrapper.reset(seed=3)

    _, _, first_terminated, first_truncated, _ = wrapper.step(np.zeros(2))
    _, reward, terminated, truncated, info = wrapper.step(np.zeros(2))

    assert not first_terminated and not first_truncated
    assert terminated and not truncated
    assert reward < -90.0
    assert info["outcome"] == DECISION_TIMEOUT
    assert info["decision_timeout"]
    assert info["decision_limit"] == 2
    assert not info["task_success"]
    with pytest.raises(RuntimeError, match="call reset"):
        wrapper.step(np.zeros(2))


def test_decision_deadline_overrides_simultaneous_physics_timeout():
    base = SampleReturnConfig()
    short = dataclasses.replace(
        base,
        flight=dataclasses.replace(base.flight, max_steps=1),
    )
    wrapper = make_training_env(
        config=short,
        action_repeat=1,
        decision_limit=1,
    )
    wrapper.reset(seed=3)

    _, reward, terminated, truncated, info = wrapper.step(np.zeros(2))

    assert terminated and not truncated
    assert reward < -90.0
    assert info["outcome"] == DECISION_TIMEOUT
    assert info["decision_timeout"]


def test_auto_advance_sampling_matches_manual_zero_actions_and_stops_at_return():
    wrapper = make_training_env(action_repeat=4, auto_advance_sampling=True)
    manual = SampleReturnEnv()
    wrapper.reset(seed=2)
    manual.reset(
        seed=2,
        options={"start_phase": "full", "spawn_mode": "ground"},
    )
    _place_for_touchdown(wrapper.mission_env, "SAMPLE")
    _place_for_touchdown(manual, "SAMPLE")
    outbound_action = np.array([1.0, 0.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = wrapper.step(outbound_action)

    manual_observation, first_reward, manual_terminated, manual_truncated, _ = (
        manual.step(outbound_action)
    )
    manual_reward = first_reward
    manual_steps = 1
    while manual.phase == MissionPhase.SAMPLING:
        manual_observation, one_reward, manual_terminated, manual_truncated, _ = (
            manual.step(np.zeros(2))
        )
        manual_reward += one_reward
        manual_steps += 1

    assert not terminated and not truncated
    assert not manual_terminated and not manual_truncated
    assert wrapper.mission_env.phase == MissionPhase.RETURN
    assert wrapper.mission_env.grounded_pad == "SAMPLE"
    assert wrapper.mission_env.mission_state.payload_attached
    np.testing.assert_array_equal(wrapper.mission_env.last_action, np.zeros(2))
    np.testing.assert_allclose(wrapper.mission_env.state, manual.state)
    np.testing.assert_allclose(observation, actor_observation(manual_observation))
    assert reward == pytest.approx(manual_reward)
    assert info["physics_steps_this_decision"] == manual_steps
    assert info["auto_advanced_sampling_steps"] == wrapper.mission_env.mission_config.sampling_steps
    assert info["physics_steps"] == manual_steps
    assert manual_steps == 1 + wrapper.mission_env.mission_config.sampling_steps


@pytest.mark.parametrize(
    ("task", "phase", "grounded_pad", "body_x", "payload_attached"),
    [
        (TrainingTask.SAMPLE_LANDING, MissionPhase.OUTBOUND, None, "sample", False),
        (TrainingTask.RETURN_LANDING, MissionPhase.RETURN, None, "base", True),
        (TrainingTask.OUTBOUND_LEG, MissionPhase.OUTBOUND, "BASE", "base", False),
        (TrainingTask.RETURN_LEG, MissionPhase.RETURN, "SAMPLE", "sample", True),
        (TrainingTask.FULL_MISSION, MissionPhase.OUTBOUND, "BASE", "base", False),
    ],
)
def test_task_presets_create_the_expected_episode_start(
    task, phase, grounded_pad, body_x, payload_attached
):
    wrapper = make_training_env(task=task)

    _, info = wrapper.reset(seed=11)

    env = wrapper.mission_env
    expected_x = env.base_x if body_x == "base" else env.sample_x
    assert env.phase == phase
    assert env.grounded_pad == grounded_pad
    assert env.mission_state.payload_attached == payload_attached
    assert dry_body_center(env.state, env.vehicle)[X] == pytest.approx(expected_x)
    assert info["task"] == task.value
    assert not info["task_success"]
    assert info["decision_steps"] == 0
    assert info["physics_steps"] == 0
    assert info["physics_steps_this_decision"] == 0


@pytest.mark.parametrize(
    "task", [TrainingTask.SAMPLE_LANDING, TrainingTask.OUTBOUND_LEG]
)
def test_outbound_segment_tasks_end_successfully_at_sample_touchdown(task):
    wrapper = make_training_env(task=task, action_repeat=4)
    wrapper.reset(seed=4)
    _place_for_touchdown(wrapper.mission_env, "SAMPLE")

    _, _, terminated, truncated, info = wrapper.step(np.zeros(2))

    assert terminated and not truncated
    assert info["task_success"]
    assert info["task"] == task.value
    assert info["physics_steps_this_decision"] == 1
    assert wrapper.mission_env.phase == MissionPhase.SAMPLING
    assert not wrapper.mission_env.mission_state.payload_attached
    with pytest.raises(RuntimeError, match="call reset"):
        wrapper.step(np.zeros(2))


def test_reset_task_override_and_airborne_options_are_forwarded():
    wrapper = make_training_env(task=TrainingTask.FULL_MISSION)

    _, info = wrapper.reset(
        seed=8,
        options={
            "task": "return_landing",
            "payload_mass": 0.4,
            "spawn_altitude": 9.0,
            "spawn_x_offset": 1.25,
            "spawn_vx": -0.7,
            "spawn_vy": -1.3,
            "spawn_theta": 0.08,
            "spawn_omega": -0.04,
        },
    )

    env = wrapper.mission_env
    body_center = dry_body_center(env.state, env.vehicle)
    assert wrapper.active_task == TrainingTask.RETURN_LANDING
    assert info["task"] == "return_landing"
    assert env.vehicle.total_mass == pytest.approx(env.cfg.m + 0.4)
    assert body_center[X] == pytest.approx(env.base_x + 1.25)
    assert env.state[[VX, VY, THETA, OMEGA]] == pytest.approx(
        [-0.7, -1.3, 0.08, -0.04]
    )


def test_target_distance_range_is_seeded_and_direction_aware():
    outbound = make_training_env(task=TrainingTask.OUTBOUND_LEG)
    returning = make_training_env(task=TrainingTask.RETURN_LEG)
    options = {"spawn_distance_from_target_range": (10.0, 20.0)}
    expected_rng = np.random.default_rng(7)

    outbound.reset(seed=7, options=options)
    returning.reset(seed=7, options=options)
    outbound_distance = (
        outbound.mission_env.sample_x
        - dry_body_center(outbound.mission_env.state, outbound.mission_env.vehicle)[X]
    )
    return_distance = (
        dry_body_center(returning.mission_env.state, returning.mission_env.vehicle)[X]
        - returning.mission_env.base_x
    )

    assert 10.0 <= outbound_distance <= 20.0
    assert outbound_distance == pytest.approx(expected_rng.uniform(10.0, 20.0))
    assert return_distance == pytest.approx(outbound_distance)

    outbound.reset(options=options)
    next_distance = (
        outbound.mission_env.sample_x
        - dry_body_center(outbound.mission_env.state, outbound.mission_env.vehicle)[X]
    )
    assert next_distance == pytest.approx(expected_rng.uniform(10.0, 20.0))

    outbound.reset(seed=7, options=options)
    repeated_distance = (
        outbound.mission_env.sample_x
        - dry_body_center(outbound.mission_env.state, outbound.mission_env.vehicle)[X]
    )
    assert repeated_distance == pytest.approx(outbound_distance)

    with pytest.raises(ValueError, match="cannot both be set"):
        outbound.reset(
            seed=7,
            options={
                "spawn_distance_from_target": 12.0,
                "spawn_distance_from_target_range": (10.0, 20.0),
            },
        )


def test_spawn_position_region_is_seeded_and_samples_both_axes():
    wrapper = make_training_env(task=TrainingTask.OUTBOUND_LEG)
    options = {
        "spawn_distance_from_target_range": (10.0, 20.0),
        "spawn_altitude_range": (3.0, 7.0),
    }
    expected_rng = np.random.default_rng(11)
    expected_distance = expected_rng.uniform(10.0, 20.0)
    expected_altitude = expected_rng.uniform(3.0, 7.0)

    wrapper.reset(seed=11, options=options)
    env = wrapper.mission_env
    body_center = dry_body_center(env.state, env.vehicle)
    distance = env.sample_x - body_center[X]
    altitude = (
        body_center[Y]
        - env.terrain.height_at(float(body_center[X]))
        - env.vehicle.body_height / 2.0
    )

    assert distance == pytest.approx(expected_distance)
    assert altitude == pytest.approx(expected_altitude)

    with pytest.raises(ValueError, match="cannot both be set"):
        wrapper.reset(
            seed=11,
            options={
                "spawn_altitude": 5.0,
                "spawn_altitude_range": (3.0, 7.0),
            },
        )


def test_spawn_theta_range_is_seeded_and_conflicts_with_scalar_theta():
    wrapper = make_training_env(task=TrainingTask.OUTBOUND_LEG)
    options = {
        "spawn_distance_from_target": 10.0,
        "spawn_altitude": 5.0,
        "spawn_theta_range": (-0.08, 0.08),
    }
    expected_theta = np.random.default_rng(13).uniform(-0.08, 0.08)

    wrapper.reset(seed=13, options=options)
    assert wrapper.mission_env.state[THETA] == pytest.approx(expected_theta)

    with pytest.raises(ValueError, match="cannot both be set"):
        wrapper.reset(
            seed=13,
            options={**options, "spawn_theta": 0.0},
        )


def test_spawn_omega_range_is_seeded_and_conflicts_with_scalar_omega():
    wrapper = make_training_env(task=TrainingTask.OUTBOUND_LEG)
    options = {
        "spawn_distance_from_target": 10.0,
        "spawn_altitude": 5.0,
        "spawn_omega_range": (-0.05, 0.05),
    }
    expected_omega = np.random.default_rng(14).uniform(-0.05, 0.05)

    wrapper.reset(seed=14, options=options)
    assert wrapper.mission_env.state[OMEGA] == pytest.approx(expected_omega)

    with pytest.raises(ValueError, match="cannot both be set"):
        wrapper.reset(
            seed=14,
            options={**options, "spawn_omega": 0.0},
        )


@pytest.mark.parametrize(
    "task", [TrainingTask.OUTBOUND_LEG, TrainingTask.RETURN_LEG]
)
def test_takeoff_rim_clearance_is_a_successful_training_boundary(task):
    wrapper = make_training_env(task=task, action_repeat=1)
    wrapper.reset(seed=15, options={"takeoff_rim_clearance": 5.0})
    env = wrapper.mission_env
    env.grounded_pad = None
    env.mission_state.contact_armed = True
    body_base, _ = body_endpoints(env.state, env.cfg, env.vehicle)
    env.state[Y] += env.terrain.spec.rim_y + 5.0 - body_base[Y] - 0.01
    env.state[VY] = 1.0

    _, reward, terminated, truncated, info = wrapper.step(np.zeros(2))

    assert terminated and not truncated
    assert reward > 20.0
    assert info["task_success"]
    assert info["outcome"] == TAKEOFF_RIM_CLEARANCE


def test_takeoff_rim_clearance_rejects_a_ballistic_handoff():
    wrapper = make_training_env(task=TrainingTask.OUTBOUND_LEG, action_repeat=1)
    wrapper.reset(seed=16, options={"takeoff_rim_clearance": 5.0})
    env = wrapper.mission_env
    env.grounded_pad = None
    env.mission_state.contact_armed = True
    body_base, _ = body_endpoints(env.state, env.cfg, env.vehicle)
    env.state[Y] += env.terrain.spec.rim_y + 5.0 - body_base[Y] + 0.01
    env.state[VY] = 10.0

    _, reward, terminated, truncated, info = wrapper.step(np.zeros(2))

    assert not terminated and not truncated
    assert reward < 20.0
    assert not info["task_success"]


def test_takeoff_rim_clearance_requires_a_ground_leg_task():
    wrapper = make_training_env(task=TrainingTask.FULL_MISSION)
    with pytest.raises(ValueError, match="outbound_leg or return_leg"):
        wrapper.reset(options={"takeoff_rim_clearance": 5.0})

    leg = make_training_env(task=TrainingTask.OUTBOUND_LEG)
    with pytest.raises(ValueError, match="spawn_mode='ground'"):
        leg.reset(
            options={
                "takeoff_rim_clearance": 5.0,
                "spawn_mode": "airborne",
            }
        )
    with pytest.raises(ValueError, match="spawn_mode='ground'"):
        leg.reset(
            options={
                "takeoff_rim_clearance": 5.0,
                "spawn_distance_from_target": 10.0,
            }
        )
    with pytest.raises(TypeError, match="must be a number"):
        leg.reset(options={"takeoff_rim_clearance": True})


def test_payload_ranges_are_seeded_and_visible_to_the_privileged_critic():
    wrapper = make_training_env(task=TrainingTask.RETURN_LEG)
    options = {
        "spawn_pad_edge_distance_range": (0.25, 1.0),
        "spawn_inside_pad_probability": 1.0 / 6.0,
        "spawn_pad_center_probability": 1.0 / 6.0,
        "spawn_altitude_range": (3.0, 12.0),
        "payload_mass_range": (0.05, 0.20),
        "payload_offset_body_x_range": (-0.30, 0.30),
        "payload_offset_body_y_range": (-0.10, 0.10),
    }
    expected_rng = np.random.default_rng(12)
    expected_rng.random(3)
    expected_rng.uniform(3.0, 12.0)
    expected_mass = expected_rng.uniform(0.05, 0.20)
    expected_offset_x = expected_rng.uniform(-0.30, 0.30)
    expected_offset_y = expected_rng.uniform(-0.10, 0.10)

    _, info = wrapper.reset(seed=12, options=options)
    payload = wrapper.mission_env.payload_spec
    critic = info["critic_observation"]

    assert payload.mass == pytest.approx(expected_mass)
    assert payload.offset_body_x == pytest.approx(expected_offset_x)
    assert payload.offset_body_y == pytest.approx(expected_offset_y)
    assert critic[CRITIC_OBSERVATION_INDEX["payload_mass"]] == pytest.approx(
        expected_mass
    )
    assert critic[
        CRITIC_OBSERVATION_INDEX["payload_offset_body_x"]
    ] == pytest.approx(expected_offset_x / wrapper.mission_env.cfg.H)

    with pytest.raises(ValueError, match="cannot both be set"):
        wrapper.reset(
            options={"payload_mass": 0.1, "payload_mass_range": (0.05, 0.20)}
        )
    with pytest.raises(ValueError, match="nonnegative"):
        wrapper.reset(options={"payload_mass_range": (-0.1, 0.2)})


def test_symmetric_pad_edge_region_is_seeded_weighted_and_task_independent():
    outbound = make_training_env(task=TrainingTask.OUTBOUND_LEG)
    returning = make_training_env(task=TrainingTask.RETURN_LEG)
    options = {
        "spawn_pad_edge_distance_range": (0.25, 1.0),
        "spawn_inside_pad_probability": 1.0 / 6.0,
        "spawn_pad_center_probability": 1.0 / 6.0,
        "spawn_altitude_range": (3.0, 12.0),
    }

    for seed in (0, 1, 2, 3):
        expected_rng = np.random.default_rng(seed)
        selector = expected_rng.random()
        side = -1.0 if expected_rng.random() < 0.5 else 1.0
        fraction = expected_rng.random()
        if selector < 1.0 / 6.0:
            expected_offset = 0.0
        elif selector < 1.0 / 3.0:
            expected_offset = (2.0 * fraction - 1.0) * 5.0
        else:
            expected_offset = side * (5.25 + 0.75 * fraction)
        expected_altitude = expected_rng.uniform(3.0, 12.0)

        for wrapper in (outbound, returning):
            wrapper.reset(seed=seed, options=options)
            env = wrapper.mission_env
            body_center = dry_body_center(env.state, env.vehicle)
            offset = body_center[X] - env.target_x
            altitude = (
                body_center[Y]
                - env.terrain.height_at(float(body_center[X]))
                - env.vehicle.body_height / 2.0
            )
            assert offset == pytest.approx(expected_offset)
            assert altitude == pytest.approx(expected_altitude)

    with pytest.raises(ValueError, match="horizontal spawn option"):
        outbound.reset(
            options={
                **options,
                "spawn_x_offset": 1.0,
            }
        )
    with pytest.raises(ValueError, match="requires spawn_mode='airborne'"):
        outbound.reset(
            options={
                **options,
                "spawn_mode": "ground",
            }
        )


def test_vector_factory_supports_one_or_one_task_per_environment():
    vector = make_vector_env(
        2,
        task=(TrainingTask.OUTBOUND_LEG, TrainingTask.RETURN_LEG),
        action_repeat=1,
    )
    try:
        observations, info = vector.reset(seed=[21, 22])
        assert observations.shape == (2, ACTOR_OBSERVATION_DIM)
        assert info["critic_observation"].shape == (
            2,
            CRITIC_OBSERVATION_DIM,
        )
        assert info["_critic_observation"].all()
        assert info["task"].tolist() == ["outbound_leg", "return_leg"]
        assert vector.autoreset_mode == AutoresetMode.DISABLED

        observations, _, terminated, truncated, info = vector.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        assert observations.shape == (2, ACTOR_OBSERVATION_DIM)
        assert info["critic_observation"].shape == (
            2,
            CRITIC_OBSERVATION_DIM,
        )
        assert not terminated.any() and not truncated.any()
        assert info["decision_steps"].tolist() == [1, 1]
        assert info["physics_steps_this_decision"].tolist() == [1, 1]
    finally:
        vector.close()

    shared = make_vector_env(2, task="full_mission", action_repeat=1)
    try:
        _, info = shared.reset(seed=30)
        assert info["task"].tolist() == ["full_mission", "full_mission"]
    finally:
        shared.close()


def test_asynchronous_vector_env_matches_synchronous_steps():
    sync_vector = make_vector_env(
        2, task=TrainingTask.SAMPLE_LANDING,
        action_repeat=4, decision_limit=10,
    )
    async_vector = make_vector_env(
        2, task=TrainingTask.SAMPLE_LANDING,
        action_repeat=4, decision_limit=10, asynchronous=True,
    )
    try:
        seeds = [21, 22]
        sync_observations, sync_info = sync_vector.reset(seed=seeds)
        async_observations, async_info = async_vector.reset(seed=seeds)
        np.testing.assert_array_equal(sync_observations, async_observations)
        np.testing.assert_array_equal(
            sync_info["critic_observation"],
            async_info["critic_observation"],
        )

        actions = np.asarray([[0.5, -0.2], [0.7, 0.3]], dtype=np.float32)
        for _ in range(3):
            sync_step = sync_vector.step(actions)
            async_step = async_vector.step(actions)
            for sync_value, async_value in zip(sync_step[:4], async_step[:4]):
                np.testing.assert_array_equal(sync_value, async_value)
            np.testing.assert_array_equal(
                sync_step[4]["critic_observation"],
                async_step[4]["critic_observation"],
            )
    finally:
        sync_vector.close()
        async_vector.close()


def test_vector_stage_config_keeps_payload_offset_across_masked_resets():
    production = SampleReturnConfig()
    bridge = dataclasses.replace(
        production,
        payload=dataclasses.replace(
            production.payload,
            offset_body_x=0.4,
        ),
    )
    vector = make_vector_env(
        2,
        task=TrainingTask.RETURN_LANDING,
        config=bridge,
        action_repeat=1,
    )
    try:
        vector.reset(seed=[31, 32])
        assert [env.mission_env.payload_spec.offset_body_x for env in vector.envs] == [
            0.4,
            0.4,
        ]

        vector.reset(options={"reset_mask": np.ones(2, dtype=np.bool_)})
        assert [env.mission_env.payload_spec.offset_body_x for env in vector.envs] == [
            0.4,
            0.4,
        ]
        assert SampleReturnConfig().payload.offset_body_x == 0.8
    finally:
        vector.close()


def test_vector_info_handles_touchdown_in_only_one_environment():
    vector = make_vector_env(
        2,
        task=TrainingTask.SAMPLE_LANDING,
        action_repeat=1,
    )
    try:
        vector.reset(seed=[31, 32])
        _place_for_touchdown(vector.envs[1].mission_env, "SAMPLE")

        _, _, terminated, truncated, info = vector.step(
            np.zeros((2, 2), dtype=np.float32)
        )

        assert not terminated[0] and not truncated[0]
        assert terminated[1] and not truncated[1]
        assert info["_sample_touchdown"].tolist() == [False, True]
        assert info["sample_touchdown"]["_vy"].tolist() == [False, True]
    finally:
        vector.close()


def test_vector_factory_rejects_mismatched_task_count():
    with pytest.raises(ValueError, match="does not match num_envs"):
        make_vector_env(2, task=[TrainingTask.FULL_MISSION])


def test_vector_factory_keeps_terminal_observations_until_explicit_reset():
    base = SampleReturnConfig()
    short = dataclasses.replace(
        base,
        flight=dataclasses.replace(base.flight, max_steps=1),
    )
    vector = make_vector_env(2, config=short, action_repeat=1)
    try:
        vector.reset(seed=40)
        terminal_observations, _, terminated, truncated, _ = vector.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        assert not terminated.any() and truncated.all()

        first_reset, _ = vector.reset(
            options={"reset_mask": np.array([True, False], dtype=np.bool_)}
        )
        np.testing.assert_array_equal(first_reset[1], terminal_observations[1])
        vector.reset(
            options={"reset_mask": np.array([False, True], dtype=np.bool_)}
        )
        _, _, _, truncated_again, _ = vector.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        assert truncated_again.all()
    finally:
        vector.close()


def test_base_actor_and_critic_observation_contracts_are_named_and_fixed():
    assert OBSERVATION_DIM == 16
    assert len(OBSERVATION_NAMES) == OBSERVATION_DIM
    assert OBSERVATION_INDEX["target_dx"] == 0
    assert OBSERVATION_INDEX["terrain_ray_0"] == 8
    assert OBSERVATION_INDEX["terrain_ray_4"] == 12
    assert OBSERVATION_INDEX["payload_attached"] == 13
    assert OBSERVATION_INDEX["phase_return"] == 15

    assert ACTOR_OBSERVATION_DIM == OBSERVATION_DIM
    assert ACTOR_OBSERVATION_NAMES == OBSERVATION_NAMES
    assert ACTOR_OBSERVATION_INDEX == OBSERVATION_INDEX

    assert CRITIC_OBSERVATION_DIM == 29
    assert CRITIC_OBSERVATION_NAMES[:2] == ("position_x", "position_y")
    assert CRITIC_OBSERVATION_NAMES[2:18] == OBSERVATION_NAMES
    assert CRITIC_OBSERVATION_NAMES[-5:] == tuple(
        f"task_{task.value}" for task in TrainingTask
    )


def test_training_observations_include_velocity_and_privilege_the_critic():
    wrapper = make_training_env(
        task=TrainingTask.RETURN_LANDING,
        action_repeat=1,
        decision_limit=10,
    )
    observation, info = wrapper.reset(
        seed=8,
        options={
            "payload_mass": 0.4,
            "payload_offset_body_x": 0.6,
            "payload_offset_body_y": -0.2,
            "spawn_vx": -0.7,
            "spawn_vy": -1.3,
            "spawn_omega": -0.04,
        },
    )

    full = flat_observation(wrapper.mission_env)
    critic = info["critic_observation"]
    np.testing.assert_array_equal(observation, actor_observation(full))
    assert observation.dtype == np.float32
    assert wrapper.observation_space.contains(observation)
    assert observation[ACTOR_OBSERVATION_INDEX["velocity_x"]] == pytest.approx(
        -0.7 / 20.0
    )
    assert observation[ACTOR_OBSERVATION_INDEX["velocity_y"]] == pytest.approx(
        -1.3 / 20.0
    )
    assert observation[
        ACTOR_OBSERVATION_INDEX["angular_velocity"]
    ] == pytest.approx(-0.04 / 5.0)

    assert critic.dtype == np.float32
    assert wrapper.critic_observation_space.contains(critic)
    assert critic[CRITIC_OBSERVATION_INDEX["velocity_x"]] == pytest.approx(-0.7 / 20.0)
    assert critic[CRITIC_OBSERVATION_INDEX["velocity_y"]] == pytest.approx(-1.3 / 20.0)
    assert critic[CRITIC_OBSERVATION_INDEX["angular_velocity"]] == pytest.approx(
        -0.04 / 5.0
    )
    assert critic[CRITIC_OBSERVATION_INDEX["payload_mass"]] == pytest.approx(0.4)
    assert critic[CRITIC_OBSERVATION_INDEX["payload_offset_body_x"]] == pytest.approx(0.6 / 4.0)
    assert critic[CRITIC_OBSERVATION_INDEX["payload_offset_body_y"]] == pytest.approx(-0.2 / 4.0)
    assert critic[CRITIC_OBSERVATION_INDEX["decision_fraction_remaining"]] == 1.0
    assert critic[CRITIC_OBSERVATION_INDEX["task_return_landing"]] == 1.0

    _, _, terminated, truncated, step_info = wrapper.step(np.zeros(2))
    assert not terminated and not truncated
    assert step_info["critic_observation"][
        CRITIC_OBSERVATION_INDEX["decision_fraction_remaining"]
    ] == pytest.approx(0.9)
    np.testing.assert_array_equal(
        step_info["critic_observation"][2:18],
        flat_observation(wrapper.mission_env),
    )


def test_training_factory_rejects_a_different_observation_width():
    base = SampleReturnConfig()
    three_rays = dataclasses.replace(
        base,
        flight=dataclasses.replace(base.flight, n_rays=3),
    )
    with pytest.raises(ValueError, match="16-dimensional"):
        make_training_env(config=three_rays)


def test_wrapper_validates_rate_and_task():
    with pytest.raises(ValueError, match="positive"):
        make_training_env(action_repeat=0)
    with pytest.raises(TypeError, match="integer"):
        make_training_env(action_repeat=2.5)
    with pytest.raises(ValueError, match="unknown training task"):
        make_training_env(task="not_a_task")
    with pytest.raises(ValueError, match="decision_limit must be positive"):
        make_training_env(decision_limit=0)
    with pytest.raises(TypeError, match="decision_limit must be an integer"):
        make_training_env(decision_limit=2.5)

    assert isinstance(make_training_env(), SampleReturnTrainingWrapper)
