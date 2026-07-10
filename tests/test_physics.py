"""Pure-physics tests: determinism, purity, sanity of forces and signs."""

import math

import numpy as np
import pytest

from rocketenv.config import Config
from rocketenv.physics import (
    FUEL,
    OMEGA,
    THETA,
    VX,
    VY,
    X,
    Y,
    body_endpoints,
    nose_direction,
    step_dynamics,
)

CFG = Config()


def make_state(x=50.0, y=85.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=100.0):
    return np.array([x, y, vx, vy, theta, omega, fuel], dtype=np.float64)


def test_step_is_pure():
    state = make_state()
    action = np.array([0.7, 0.3])
    state_copy = state.copy()
    action_copy = action.copy()
    out = step_dynamics(state, action, CFG)
    assert np.array_equal(state, state_copy), "input state was mutated"
    assert np.array_equal(action, action_copy), "input action was mutated"
    assert out is not state


def test_determinism_bit_identical():
    rng = np.random.default_rng(0)
    actions = rng.uniform([0, -1], [1, 1], size=(500, 2))

    def rollout():
        s = make_state()
        traj = [s]
        for a in actions:
            s = step_dynamics(s, a, CFG)
            traj.append(s)
        return np.stack(traj)

    t1, t2 = rollout(), rollout()
    assert np.array_equal(t1, t2), "same actions must give bit-identical states"


def test_free_fall_at_g():
    # No thrust: vy after n steps is exactly -g*n*dt (semi-implicit Euler).
    s = make_state()
    n = 120
    for _ in range(n):
        s = step_dynamics(s, np.array([0.0, 0.0]), CFG)
    assert s[VY] == pytest.approx(-CFG.g * n * CFG.dt, rel=1e-12)
    assert s[VX] == 0.0
    assert s[THETA] == 0.0 and s[OMEGA] == 0.0
    assert s[FUEL] == CFG.fuel_0


def test_full_throttle_upright_net_upward():
    # TWR > 1: one step of full throttle upright accelerates upward at (TWR-1)*g.
    s = make_state()
    s2 = step_dynamics(s, np.array([1.0, 0.0]), CFG)
    expected_dv = (CFG.twr - 1.0) * CFG.g * CFG.dt
    assert s2[VY] == pytest.approx(expected_dv, rel=1e-12)


def test_zero_fuel_zero_thrust():
    s = make_state(fuel=0.0)
    s2 = step_dynamics(s, np.array([1.0, 0.5]), CFG)
    assert s2[VY] == pytest.approx(-CFG.g * CFG.dt, rel=1e-12)
    assert s2[OMEGA] == 0.0
    assert s2[FUEL] == 0.0


def test_fuel_burn_rate():
    s = make_state()
    s2 = step_dynamics(s, np.array([0.5, 0.0]), CFG)
    assert s2[FUEL] == pytest.approx(CFG.fuel_0 - CFG.burn_rate * 0.5 * CFG.dt)


def test_action_clipping_in_dynamics():
    s = make_state()
    a_wild = np.array([5.0, -7.0])
    a_clipped = np.array([1.0, -1.0])
    assert np.array_equal(step_dynamics(s, a_wild, CFG), step_dynamics(s, a_clipped, CFG))


def test_torque_sign():
    # Positive gimbal command -> tau = -L*T*sin(phi) < 0 -> theta decreases
    # (rocket tips clockwise, i.e. to the right).
    s = make_state()
    s2 = step_dynamics(s, np.array([1.0, 1.0]), CFG)
    assert s2[OMEGA] < 0.0
    s3 = step_dynamics(s, np.array([1.0, -1.0]), CFG)
    assert s3[OMEGA] > 0.0


def test_nose_direction_convention():
    # theta = 0 -> nose points straight up.
    assert nose_direction(0.0) == pytest.approx((0.0, 1.0))
    # small positive (CCW) theta -> nose leans to -x.
    nx, ny = nose_direction(0.1)
    assert nx < 0.0 and ny > 0.0


def test_body_endpoints():
    s = make_state(x=50.0, y=10.0, theta=0.0)
    base, tip = body_endpoints(s, CFG)
    assert base == pytest.approx([50.0, 10.0 - CFG.H / 2])
    assert tip == pytest.approx([50.0, 10.0 + CFG.H / 2])


def test_tilted_thrust_direction():
    # theta = 90deg (nose pointing -x): full throttle pushes along -x.
    s = make_state(theta=math.pi / 2)
    s2 = step_dynamics(s, np.array([1.0, 0.0]), CFG)
    assert s2[VX] == pytest.approx(-CFG.twr * CFG.g * CFG.dt, rel=1e-12)


def test_wind_and_thrust_multiplier_params():
    cfg_wind = Config(wind_x=3.0)
    s = make_state()
    s2 = step_dynamics(s, np.array([0.0, 0.0]), cfg_wind)
    assert s2[VX] == pytest.approx(3.0 / cfg_wind.m * cfg_wind.dt)

    cfg_fail = Config(thrust_multiplier=0.5)
    s2 = step_dynamics(s, np.array([1.0, 0.0]), cfg_fail)
    expected = (0.5 * cfg_fail.twr - 1.0) * cfg_fail.g * cfg_fail.dt
    assert s2[VY] == pytest.approx(expected, rel=1e-12)
