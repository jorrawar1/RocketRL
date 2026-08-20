"""Mass-property and geometry tests for the sample-return vehicle."""

import math

import numpy as np
import pytest

from rocketenv.config import Config
from rocketenv.physics import body_endpoints as legacy_body_endpoints
from rocketenv.sample_return.mission_types import PayloadSpec
from rocketenv.sample_return.vehicle import (
    VehicleModel,
    body_endpoints,
    dry_body_center,
    engine_position,
    payload_corners,
    state_with_preserved_dry_body_center,
)


def make_state(theta=0.0):
    return np.array([12.0, 30.0, 1.0, -2.0, theta, 0.1, 70.0])


def test_no_payload_vehicle_matches_legacy_mass_and_geometry():
    cfg = Config()
    vehicle = VehicleModel.from_config(cfg)
    state = make_state(theta=0.37)

    assert vehicle.total_mass == cfg.m
    assert vehicle.total_inertia == cfg.I
    assert vehicle.com_offset_body == pytest.approx([0.0, 0.0])
    assert vehicle.engine_offset_from_com_body == pytest.approx([0.0, -cfg.L])
    actual_base, actual_tip = body_endpoints(state, cfg, vehicle)
    expected_base, expected_tip = legacy_body_endpoints(state, cfg)
    assert np.array_equal(actual_base, expected_base)
    assert np.array_equal(actual_tip, expected_tip)


def test_payload_mass_properties_match_parallel_axis_calculation():
    cfg = Config()
    payload = PayloadSpec(
        mass=0.5,
        offset_body_x=0.8,
        offset_body_y=0.2,
        width=0.6,
        height=0.9,
    )
    vehicle = VehicleModel.from_config(cfg, payload)

    payload_offset = np.array([0.8, 0.2])
    expected_com = payload.mass / (cfg.m + payload.mass) * payload_offset
    expected_inertia = (
        cfg.I
        + cfg.m * np.dot(expected_com, expected_com)
        + payload.mass * (payload.width**2 + payload.height**2) / 12.0
        + payload.mass
        * np.dot(payload_offset - expected_com, payload_offset - expected_com)
    )

    assert vehicle.total_mass == pytest.approx(1.5)
    assert vehicle.com_offset_body == pytest.approx(expected_com)
    assert vehicle.total_inertia == pytest.approx(expected_inertia)
    assert vehicle.engine_offset_from_com_body == pytest.approx(
        np.array([0.0, -cfg.L]) - expected_com
    )


def test_centered_payload_adds_local_inertia_without_shifting_com():
    cfg = Config()
    payload = PayloadSpec(0.35, 0.0, 0.0, 0.6, 0.8)
    vehicle = VehicleModel.from_config(cfg, payload)

    expected_local = payload.mass * (payload.width**2 + payload.height**2) / 12.0
    assert vehicle.com_offset_body == pytest.approx([0.0, 0.0])
    assert vehicle.total_inertia == pytest.approx(cfg.I + expected_local)


@pytest.mark.parametrize("theta", [0.0, 0.4, math.pi / 2.0, -1.2])
def test_recenter_for_attachment_preserves_visible_body_geometry(theta):
    cfg = Config()
    dry_vehicle = VehicleModel.from_config(cfg)
    loaded_vehicle = VehicleModel.from_config(
        cfg, PayloadSpec(0.35, 0.8, 0.1, 0.6, 0.8)
    )
    state = make_state(theta)
    state_before = state.copy()
    old_center = dry_body_center(state, dry_vehicle)

    loaded_state = state_with_preserved_dry_body_center(
        state, dry_vehicle, loaded_vehicle
    )

    assert np.array_equal(state, state_before)
    assert dry_body_center(loaded_state, loaded_vehicle) == pytest.approx(old_center)
    assert np.array_equal(loaded_state[2:], state[2:])


def test_payload_corners_and_engine_follow_body_attitude():
    cfg = Config()
    payload = PayloadSpec(0.35, 0.8, 0.0, 0.6, 0.8)
    vehicle = VehicleModel.from_config(cfg, payload)
    dry_vehicle = VehicleModel.from_config(cfg)
    state = state_with_preserved_dry_body_center(
        make_state(theta=math.pi / 2.0), dry_vehicle, vehicle
    )

    corners = payload_corners(state, vehicle)
    assert corners.shape == (4, 2)
    assert np.isfinite(corners).all()
    assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(payload.width)
    assert np.linalg.norm(corners[2] - corners[1]) == pytest.approx(payload.height)
    assert engine_position(state, vehicle) == pytest.approx([14.0, 30.0])


def test_vehicle_without_payload_has_no_payload_corners():
    vehicle = VehicleModel.from_config(Config())
    assert payload_corners(make_state(), vehicle).shape == (0, 2)


def test_payload_spec_rejects_invalid_physical_values():
    with pytest.raises(ValueError, match="mass"):
        PayloadSpec(-0.1, 0.8, 0.0, 0.6, 0.8)
    with pytest.raises(ValueError, match="finite"):
        PayloadSpec(0.35, math.nan, 0.0, 0.6, 0.8)
    with pytest.raises(ValueError, match="dimensions"):
        PayloadSpec(0.35, 0.8, 0.0, 0.0, 0.8)
