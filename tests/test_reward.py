"""Landing-model tests: legs absorb vertical hits, tip-over rejects
sideways/spinning arrivals, graded credit otherwise."""

import math

import numpy as np
import pytest

from rocketenv.config import Config
from rocketenv.reward import (
    CRASH, TIPPED, TOUCHDOWN, contact_reward, sticks_upright,
)

CFG = Config()


def st(x=50.0, y=2.0, vx=0.0, vy=-1.0, theta=0.0, omega=0.0, fuel=50.0):
    return np.array([x, y, vx, vy, theta, omega, fuel], dtype=np.float64)


def test_gentle_upright_touchdown():
    r, outcome = contact_reward(st(), CFG)
    assert outcome == TOUCHDOWN
    assert r == pytest.approx(CFG.reward_land + CFG.reward_fuel_bonus * 0.5)


def test_hard_vertical_landing_sticks():
    # Suicide-burn style: fast vertical, upright, no drift -> legs absorb it.
    r, outcome = contact_reward(st(vy=-4.5), CFG)
    assert outcome == TOUCHDOWN


def test_too_fast_vertical_is_crash():
    _, outcome = contact_reward(st(vy=-6.0), CFG)
    assert outcome == CRASH


def test_sideways_spinning_arrival_tips():
    # Survivable impact speeds, but drift + spin conspire to tip it over.
    s = st(vx=3.4, vy=-2.0, omega=1.5)
    assert not sticks_upright(s, CFG)
    r, outcome = contact_reward(s, CFG)
    assert outcome == TIPPED
    assert r < CFG.reward_land / 2  # tipping never pays like landing


def test_past_balance_angle_always_tips():
    theta_c = math.atan2(CFG.leg_half_w, CFG.L)
    assert not sticks_upright(st(theta=theta_c + 0.05), CFG)
    _, outcome = contact_reward(st(theta=theta_c + 0.05), CFG)
    assert outcome == TIPPED


def test_small_residual_spin_is_fine():
    # The old omega < 0.5 criterion rejected these perfectly good landings.
    assert sticks_upright(st(vx=1.4, vy=-1.2, omega=0.8), CFG)
    _, outcome = contact_reward(st(vx=1.4, vy=-1.2, omega=0.8), CFG)
    assert outcome == TOUCHDOWN


def test_off_pad_gentle_is_crash_with_partial_credit():
    r, outcome = contact_reward(st(x=50 + CFG.pad_half_w + 3, vy=-0.5), CFG)
    assert outcome == CRASH
    assert 0 < r < CFG.reward_land  # near-miss credit, not a landing


def test_fuel_bonus_scales():
    r_full, _ = contact_reward(st(fuel=CFG.fuel_0), CFG)
    r_empty, _ = contact_reward(st(fuel=0.0), CFG)
    assert r_full == pytest.approx(CFG.reward_land + CFG.reward_fuel_bonus)
    assert r_empty == pytest.approx(CFG.reward_land)
