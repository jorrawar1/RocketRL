"""Portable rollout fixture tests."""

import json

from rocketenv.sample_return.serialization import export_fixture, rollout_fixture


def test_fixture_contains_browser_contract_and_complete_mission():
    fixture = rollout_fixture(seed=42)

    assert fixture["schema_version"] == 1
    assert fixture["mission_seed"] == 42
    assert fixture["controller"] == "scripted"
    assert fixture["outcome"] == "SAMPLE_RETURNED"
    assert fixture["terrain"]["vertices"]
    assert fixture["base_pad"]["x"] != fixture["sample_pad"]["x"]
    assert fixture["payload"]["mass"] > 0.0
    assert fixture["frames"]
    assert {
        "t", "state", "action", "phase", "payload_mass",
        "payload_fill", "com_offset_body",
    } <= fixture["frames"][0].keys()


def test_fixture_rollout_is_stable_for_same_seed():
    first = rollout_fixture(seed=9)
    second = rollout_fixture(seed=9)
    assert first == second


def test_export_fixture_writes_valid_json(tmp_path):
    destination = tmp_path / "nested" / "mission.json"
    fixture = export_fixture(destination, seed=42)

    assert destination.exists()
    assert json.loads(destination.read_text(encoding="utf-8")) == fixture
