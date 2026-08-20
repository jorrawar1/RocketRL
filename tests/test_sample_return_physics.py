"""Payload-aware dynamics tests, including strict legacy parity."""

import math

import numpy as np
import pytest

from rocketenv.config import Config
from rocketenv.physics import OMEGA, VX, VY, step_dynamics as legacy_step_dynamics
from rocketenv.sample_return.physics import step_dynamics, thrust_torque
from rocketenv.sample_return.mission_types import PayloadSpec
from rocketenv.sample_return.vehicle import VehicleModel


def make_state(theta=0.0, fuel=100.0):
    return np.array([31.0, 48.0, 1.2, -2.4, theta, 0.15, fuel])


@pytest.mark.parametrize(
    ("state", "action", "cfg"),
    [
        (make_state(), np.array([0.0, 0.0]), Config()),
        (make_state(0.3), np.array([0.73, -0.42]), Config()),
        (make_state(-0.7), np.array([2.0, 3.0]), Config()),
        (make_state(0.2, fuel=0.0), np.array([1.0, 0.5]), Config()),
        (
            make_state(-0.1),
            np.array([0.4, 0.2]),
            Config(wind_x=1.3, wind_gust_x=-0.2, drag_coeff=0.01),
        ),
    ],
)
def test_no_payload_step_has_strict_legacy_parity(state, action, cfg):
    vehicle = VehicleModel.from_config(cfg)
    expected = legacy_step_dynamics(state, action, cfg)
    actual = step_dynamics(state, action, cfg, vehicle)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-16)


def test_payload_step_is_pure():
    cfg = Config()
    vehicle = VehicleModel.from_config(
        cfg, PayloadSpec(0.35, 0.8, 0.0, 0.6, 0.8)
    )
    state = make_state()
    action = np.array([0.7, 0.3])
    state_before = state.copy()
    action_before = action.copy()

    result = step_dynamics(state, action, cfg, vehicle)

    assert np.array_equal(state, state_before)
    assert np.array_equal(action, action_before)
    assert result is not state


def test_centered_payload_reduces_vertical_acceleration_without_trim_torque():
    cfg = Config()
    dry = VehicleModel.from_config(cfg)
    loaded = VehicleModel.from_config(
        cfg, PayloadSpec(0.35, 0.0, 0.0, 0.6, 0.8)
    )
    state = make_state()
    action = np.array([1.0, 0.0])

    dry_next = step_dynamics(state, action, cfg, dry)
    loaded_next = step_dynamics(state, action, cfg, loaded)

    assert loaded_next[VY] < dry_next[VY]
    assert loaded_next[OMEGA] == state[OMEGA]
    expected_dv = (
        (cfg.T_max - loaded.total_mass * cfg.g) / loaded.total_mass * cfg.dt
    )
    assert loaded_next[VY] - state[VY] == pytest.approx(expected_dv)


def test_right_mounted_payload_creates_clockwise_torque_at_zero_gimbal():
    cfg = Config()
    loaded = VehicleModel.from_config(
        cfg, PayloadSpec(0.35, 0.8, 0.0, 0.6, 0.8)
    )
    state = make_state()
    state[OMEGA] = 0.0

    next_state = step_dynamics(state, np.array([1.0, 0.0]), cfg, loaded)

    assert thrust_torque(loaded, cfg.T_max, 0.0) < 0.0
    assert next_state[OMEGA] < 0.0


def test_engine_rating_does_not_scale_with_payload_mass():
    cfg = Config()
    payload = PayloadSpec(cfg.m, 0.0, 0.0, 0.5, 0.5)
    loaded = VehicleModel.from_config(cfg, payload)
    state = make_state()
    state[1:4] = [50.0, 0.0, 0.0]

    next_state = step_dynamics(state, np.array([1.0, 0.0]), cfg, loaded)

    expected_acceleration = cfg.T_max / loaded.total_mass - cfg.g
    assert (next_state[VY] - state[VY]) / cfg.dt == pytest.approx(
        expected_acceleration
    )
    assert expected_acceleration != pytest.approx((cfg.twr - 1.0) * cfg.g)


def test_payload_does_not_change_horizontal_thrust_direction():
    cfg = Config()
    loaded = VehicleModel.from_config(
        cfg, PayloadSpec(0.35, 0.8, 0.0, 0.6, 0.8)
    )
    state = make_state(theta=math.pi / 2.0)
    state[VX] = 0.0

    next_state = step_dynamics(state, np.array([1.0, 0.0]), cfg, loaded)

    expected_vx = -cfg.T_max / loaded.total_mass * cfg.dt
    assert next_state[VX] == pytest.approx(expected_vx)
