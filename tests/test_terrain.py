"""PolylineTerrain and generator tests: interpolation, exact ray hits,
flat-equivalence, generation determinism, pad flatness."""

import math

import numpy as np
import pytest

from rocketenv import RocketEnv
from rocketenv.config import Config
from rocketenv.terrain import FlatTerrain, PolylineTerrain, generate_terrain

CFG = Config()


def test_height_interpolation():
    t = PolylineTerrain([0.0, 10.0, 20.0], [0.0, 10.0, 0.0])
    assert t.height_at(5.0) == pytest.approx(5.0)
    assert t.height_at(15.0) == pytest.approx(5.0)
    assert t.height_at(10.0) == pytest.approx(10.0)


def test_rejects_bad_vertices():
    with pytest.raises(ValueError):
        PolylineTerrain([0.0, 5.0, 5.0], [0.0, 1.0, 2.0])


def test_flat_polyline_matches_flat_terrain():
    poly = PolylineTerrain([0.0, 100.0], [0.0, 0.0])
    flat = FlatTerrain()
    for ang_deg in (-150, -120, -90, -60, -30):
        a = math.radians(ang_deg)
        d_poly = poly.ray_distance(50.0, 30.0, math.cos(a), math.sin(a), 60.0)
        d_flat = flat.ray_distance(50.0, 30.0, math.cos(a), math.sin(a), 60.0)
        assert d_poly == pytest.approx(d_flat)
    assert poly.height_at(37.3) == flat.height_at(37.3)


def test_ray_hits_hill_closed_form():
    # 45-degree slope from (0,0) to (10,10); horizontal ray at y=5 from x=20
    # traveling -x must hit the slope at x=5 -> distance 15.
    t = PolylineTerrain([0.0, 10.0, 30.0], [0.0, 10.0, 0.0])
    d = t.ray_distance(20.0, 5.0, -1.0, 0.0, 60.0)
    # hits the downslope (10,10)-(30,0) first: y=5 at x=20... origin ON the
    # downslope line: height_at(20) == 5 -> origin at ground -> 0.
    assert d == 0.0
    # from above the valley floor, straight down
    d2 = t.ray_distance(25.0, 8.0, 0.0, -1.0, 60.0)
    assert d2 == pytest.approx(8.0 - t.height_at(25.0))


def test_ray_hits_first_intersection():
    # W-shaped valley: ray angled across must return the NEAREST hit.
    t = PolylineTerrain([0, 20, 40, 60, 80], [20, 0, 15, 0, 20])
    a = math.radians(-30)
    d = t.ray_distance(30.0, 25.0, math.cos(a), math.sin(a), 100.0)
    # verify by marching: the returned point must lie on the terrain and no
    # earlier point along the ray may be below it
    hx, hy = 30 + math.cos(a) * d, 25 + math.sin(a) * d
    assert hy == pytest.approx(t.height_at(hx), abs=1e-6)
    for frac in np.linspace(0.05, 0.95, 19):
        px, py = 30 + math.cos(a) * d * frac, 25 + math.sin(a) * d * frac
        assert py > t.height_at(px)


def test_below_ground_returns_zero():
    t = PolylineTerrain([0.0, 100.0], [5.0, 5.0])
    assert t.ray_distance(50.0, 2.0, 0.0, -1.0, 60.0) == 0.0


def test_generation_deterministic():
    t1, p1 = generate_terrain(np.random.default_rng(42), CFG)
    t2, p2 = generate_terrain(np.random.default_rng(42), CFG)
    assert p1 == p2
    assert np.array_equal(t1.ys, t2.ys)
    t3, p3 = generate_terrain(np.random.default_rng(43), CFG)
    assert not np.array_equal(t1.ys, t3.ys)


def test_generated_pad_is_flat_and_feasible():
    for seed in range(30):
        terr, pad_x = generate_terrain(np.random.default_rng(seed), CFG)
        pad_y = terr.height_at(pad_x)
        # the whole pad span is level
        for x in np.linspace(pad_x - CFG.pad_half_w, pad_x + CFG.pad_half_w, 21):
            assert terr.height_at(x) == pytest.approx(pad_y, abs=1e-9)
        # hills stay well below the spawn envelope
        assert terr.ys.max() <= CFG.terrain_amp + 1e-9
        assert terr.ys.min() >= -1e-9
        assert 15.0 <= pad_x <= CFG.world_w - 15.0


def test_env_runs_on_generated_terrain():
    terr, pad_x = generate_terrain(np.random.default_rng(5), CFG)
    env = RocketEnv(terrain=terr)
    obs, _ = env.reset(seed=0, options={"pad_x": pad_x})
    assert env.cfg.pad_x == pad_x
    # free fall must terminate on hill contact with sane final altitude
    for _ in range(2000):
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0]))
        if term or trunc:
            break
    assert term
    s = info["state"]
    assert s[1] >= terr.height_at(s[0]) - 1.0  # ended at the surface, not below
    # rays reflect the local terrain, not a flat floor
    assert np.all(obs[8:] >= 0.0) and np.all(obs[8:] <= 1.0)
