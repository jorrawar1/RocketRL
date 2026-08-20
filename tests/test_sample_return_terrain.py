"""Tests for the structured crater sample-return terrain."""

import numpy as np
import pytest

from rocketenv.sample_return.terrain import CraterSampleTerrain, CraterTerrainSpec


SPEC = CraterTerrainSpec()


def test_same_seed_and_spec_produce_identical_vertices():
    first = CraterSampleTerrain.from_seed(42, SPEC)
    second = CraterSampleTerrain.from_seed(42, SPEC)
    other = CraterSampleTerrain.from_seed(43, SPEC)

    assert np.array_equal(first.vertices, second.vertices)
    assert not np.array_equal(first.ys, other.ys)


def test_base_and_sample_pads_are_flat_and_distinct():
    terrain = CraterSampleTerrain.from_seed(42, SPEC)

    for x in np.linspace(*terrain.base_pad_bounds, 31):
        assert terrain.height_at(x) == pytest.approx(terrain.base_y, abs=1e-12)
    for x in np.linspace(*terrain.sample_pad_bounds, 31):
        assert terrain.height_at(x) == pytest.approx(terrain.sample_y, abs=1e-12)

    assert terrain.base_x != terrain.sample_x
    assert terrain.base_y == pytest.approx(SPEC.outer_y)
    assert terrain.sample_y == pytest.approx(SPEC.floor_y)
    assert terrain.base_y > terrain.sample_y


def test_crater_has_named_rims_and_stays_inside_world_bounds():
    terrain = CraterSampleTerrain.from_seed(7, SPEC)

    assert terrain.left_rim_x < terrain.sample_x < terrain.right_rim_x
    assert terrain.left_rim_y == pytest.approx(SPEC.rim_y)
    assert terrain.right_rim_y == pytest.approx(SPEC.rim_y)
    assert terrain.left_rim_y > terrain.base_y > terrain.sample_y
    assert terrain.xs[0] == 0.0
    assert terrain.xs[-1] == SPEC.world_w
    assert np.all((terrain.ys >= 0.0) & (terrain.ys <= SPEC.world_h))


def test_downward_ray_hits_crater_floor_at_sample_pad():
    terrain = CraterSampleTerrain.from_seed(42, SPEC)
    origin_y = terrain.sample_y + 25.0

    distance = terrain.ray_distance(
        terrain.sample_x, origin_y, 0.0, -1.0, max_range=60.0
    )

    assert distance == pytest.approx(25.0)


def test_spec_rejects_pads_in_invalid_regions():
    with pytest.raises(ValueError, match="sample pad"):
        CraterSampleTerrain.from_seed(
            0, CraterTerrainSpec(sample_x=60.0)
        )
    with pytest.raises(ValueError, match="base pad"):
        CraterSampleTerrain.from_seed(
            0, CraterTerrainSpec(base_x=100.0)
        )
