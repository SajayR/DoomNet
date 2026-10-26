"""Small harness tests; actual E1M1 feature proof requires the strict artifact CLI."""

import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from doomnet import state as S
from doomnet.runtime import MAX_DT, RuntimeModel
from scripts import validate_features as F
from scripts.validate_neural import Case, compare_frames, compare_states, controls


@pytest.fixture
def feature_probe(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    layout = S.StateLayout(66, 85, 486)
    data = {"entity_type": torch.zeros(66, dtype=torch.int64),
            "sector_floor": torch.zeros(85), "lines": torch.zeros((486, 2), dtype=torch.int64),
            "entity_pos": torch.zeros((66, 2)), "sector_special": torch.zeros(85, dtype=torch.int64)}
    data["entity_type"][[55, 7, 65]] = torch.tensor([2018, 2019, 2001])
    spawn = torch.zeros(layout.size)
    spawn[S.HEALTH] = 100
    spawn[layout.entity_alive] = 1
    spawn[layout.sector_floor][59] = 96
    spawn[layout.sector_floor][70] = 104
    frame = torch.zeros((2, 3, 3), dtype=torch.uint8)
    monkeypatch.setattr(F, "E1M1Controller", lambda data, *args: SimpleNamespace(sector_at=lambda point: 13))

    class Runner:
        device = torch.device("cpu")
        state_lower, state_upper = torch.full_like(spawn, -1e4), torch.full_like(spawn, 1e4)
        manifest = {"files": {}}
        info = {}
        restarts = 0
        _inputs = RuntimeModel._inputs

        def __init__(self):
            self.state = spawn.clone()
            self.transition = lambda state, control, dt: state.clone()
            self.executor = SimpleNamespace(transition=lambda *args: self.transition(*args), render=lambda state: frame)

        def _check_frame(self, value):
            assert value.dtype == torch.uint8 and value.shape == frame.shape

        def step(self, control, dt):
            assert control[5] == 1 and dt == 0
            self.state = spawn.clone()
            self.restarts += 1
            return frame

    runner = Runner()
    source = SimpleNamespace(transition=lambda state, control, dt: runner.transition(state, control, dt),
                             render=lambda state: frame)
    probe = F.FeatureProbe(runner, source, data, tmp_path, "test", io.StringIO())
    return probe, runner, source, data, spawn


def test_feature_samples_never_replace_returned_neural_state(feature_probe):
    probe, runner, source, _, spawn = feature_probe

    def transition(state, control, dt):
        returned = state.clone()
        returned[S.X] += 1
        return returned

    runner.transition = transition
    source.transition = lambda state, control, dt: state.clone() + 100
    probe.step(controls(forward=1))
    neural_return = runner.state
    row = probe.mark("separate_source_diagnostic")
    case, returned, frame = probe.samples[0]
    assert runner.state is neural_return
    assert torch.equal(case.state, spawn)
    assert torch.equal(returned, runner.state) and returned[S.X] == 1
    assert torch.equal(case.expected, spawn + 100)
    assert row["before"]["position"] == [0, 0, 0]
    assert row["after"]["position"] == [1, 0, 0]
    assert frame.shape == (2, 3, 3)


@pytest.mark.parametrize("entity,event,armor_type", [(55, "green_armor", 1), (7, "blue_armor", 2), (65, "shotgun_pickup", 0)])
def test_named_pickups_sample_the_actual_consumption_tick(feature_probe, entity, event, armor_type):
    probe, runner, _, _, spawn = feature_probe

    def transition(state, control, dt):
        returned = state.clone()
        returned[probe.layout.entity_alive][entity] = 0
        returned[S.ARMOR_TYPE] = armor_type
        returned[S.ARMOR] = 100 * armor_type
        returned[S.SHOTGUN_OWNED] = int(entity == 65)
        return returned

    runner.transition = transition
    probe.step()
    assert [row["name"] for row in probe.events] == ["test:" + event]
    assert probe.events[0]["before"]["entity"]["alive"] == 1
    assert probe.events[0]["after"]["entity"]["alive"] == 0
    assert torch.equal(probe.samples[0][0].state, spawn)
    probe.step()
    assert len(probe.events) == 1


def test_armor_bonus_is_not_mislabeled_as_armor_pickup(feature_probe):
    probe, runner, _, _, _ = feature_probe
    runner.state[S.ARMOR_TYPE] = 1

    def transition(state, control, dt):
        returned = state.clone()
        returned[S.ARMOR] += 1
        return returned

    runner.transition = transition
    probe.step()
    probe.step()
    assert not probe.events and runner.state[S.ARMOR] == 2


def test_neural_bounds_failure_occurs_before_carrying_bad_state(feature_probe):
    probe, runner, _, _, spawn = feature_probe
    runner.transition = lambda state, control, dt: state + 1e5
    with pytest.raises(ValueError, match="violates x"):
        probe.step()
    assert torch.equal(runner.state, spawn)
    assert probe.steps == 0 and not probe.events


@pytest.mark.parametrize("angle", [0, .7, math.pi / 2, math.pi, 4.8, 2 * math.pi - 1e-6])
def test_fast_route_controls_remain_in_contract_and_reach_target(feature_probe, angle):
    probe, runner, _, _, _ = feature_probe
    runner.state[S.ANGLE] = angle
    inputs = []

    def transition(state, control, dt):
        inputs.append(control.clone())
        returned = state.clone()
        angle = float(state[S.ANGLE])
        forward = torch.tensor([math.cos(angle), math.sin(angle)])
        right = torch.tensor([forward[1], -forward[0]])
        returned[:2] += (forward * control[0] + right * control[1]) * 300 * dt[0]
        return returned

    runner.transition = transition
    probe.goto((170, -131), radius=6, fast=True)
    assert torch.linalg.vector_norm(runner.state[:2] - torch.tensor([170, -131])) <= 6
    assert inputs and all(bool((action[:3].abs() <= 1).all()) for action in inputs)


def test_event_accuracy_applies_unchanged_budget_to_returned_state_and_frames(feature_probe, monkeypatch):
    probe, runner, source, _, spawn = feature_probe
    case = Case("feature", (), spawn, controls(), MAX_DT, spawn)
    frame = torch.zeros((2, 3, 3), dtype=torch.uint8)
    diagnostic = {"name": "feature", "states": {"source_transition": compare_states(spawn, spawn, probe.layout)},
                  "frames": {"same_state_render": compare_frames(frame, frame)}}
    monkeypatch.setattr(F, "diagnose", lambda *args: ([diagnostic], [], {}))
    returned = spawn.clone()
    returned[S.SHOTGUN_OWNED] = 1
    bad_frame = torch.full_like(frame, 9)
    rows, failures, _, accuracy = F.measure_events(runner, source, [(case, returned, bad_frame)], probe.layout, probe.directory)
    assert accuracy["budget"] == F.ACCURACY_BUDGET and accuracy["status"] == "failed"
    assert rows[0]["states"]["recurrent_transition"]["semantic_mismatches"]
    assert any("recurrent_transition shotgun_owned" in reason for reason in failures)
    assert any("returned_state_render RGB" in reason for reason in failures)
    assert any("recurrent_transition_render RGB" in reason for reason in failures)


def test_each_feature_branch_requires_an_exported_exact_restart(feature_probe, monkeypatch):
    probe, runner, source, data, spawn = feature_probe
    monkeypatch.setattr(S, "initial_state", lambda data: spawn.clone())
    monkeypatch.setattr(F, "construction_provenance", lambda source: {"sha256": "fixed-source"})

    def transition(state, control, dt):
        returned = state.clone()
        returned[S.X] += 1
        return returned

    runner.transition = transition
    reached = []

    def branch(current):
        assert torch.equal(current.cpu(), spawn)
        current.step(controls(forward=1))
        current.mark("event")
        reached.append(current.branch)

    for name in ("fist", "green_blast", "blue_hazard", "shotgun", "lift", "ledge"):
        monkeypatch.setattr(F, name, branch)

    def diagnostics(runner, source, cases, layout, directory):
        return ([{"name": case.name, "states": {"source_transition": compare_states(case.expected, case.expected, layout)},
                  "frames": {}} for case in cases], [], {})

    monkeypatch.setattr(F, "diagnose", diagnostics)
    report = F.validate_features(runner, source, data, probe.directory)
    assert runner.restarts == 6 and reached == ["fist", "green_blast", "blue_hazard", "shotgun", "lift", "ledge"]
    assert report["status"] == "passed" and not report["failures"]
    assert all(result["restart_exact"] and result["ticks"] == 1 for result in report["branches"].values())
    corpus = torch.load(probe.directory / "events.pt", weights_only=True)
    assert torch.equal(corpus["states"], spawn.expand(6, -1))
    assert torch.equal(corpus["returned"][:, S.X], torch.ones(6))


def test_original_map_source_ledge_branch_starts_at_spawn_and_keeps_returned_body_clear(tmp_path):
    """Controls-only source diagnostic; deployed proof still requires the CLI."""
    from doomnet.assets import build_assets
    from doomnet.model import ConstructedDoom

    wad = Path(__file__).resolve().parents[1] / "assets" / "doom1.wad"
    if not wad.is_file():
        pytest.skip("Fetch the original WAD with python scripts/fetch_shareware.py")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data, _ = build_assets(wad)
        source = ConstructedDoom(data, width=64, height=48).eval()
        simulation = source.simulation

        def transition(state, action, dt):
            returned = source.transition(state, action, dt)
            assert simulation.geometry.can_move(
                returned[:2][None, :], returned[S.Z:S.Z + 1],
                returned[simulation.layout.sector_floor], returned[simulation.layout.sector_ceiling],
                simulation.player_radius), "returned player body must remain geometrically clear"
            return returned

        runner = SimpleNamespace(state=S.initial_state(data), device=torch.device("cpu"), width=64, height=48,
                                 executor=SimpleNamespace(transition=transition, render=source.render))
        runner.state_lower, runner.state_upper = S.state_bounds(data)
        runner._inputs = lambda action, dt: RuntimeModel._inputs(runner, action, dt)
        runner._check_frame = lambda frame: RuntimeModel._check_frame(runner, frame)
        probe = F.FeatureProbe(runner, source, data, tmp_path, "ledge", io.StringIO())
        F.ledge(probe)
        events = {row["name"]: row for row in probe.events}
        assert set(events) == {"ledge:green_armor", "ledge:ledge_supported", "ledge:ledge_landed",
                               "ledge:ledge_movement_continues"}
        assert events["ledge:ledge_supported"]["after"]["sector"] == 24
        assert events["ledge:ledge_supported"]["after"]["position"][2] == 104
        assert events["ledge:ledge_landed"]["before"]["position"][2] == 88
        assert events["ledge:ledge_landed"]["after"]["position"][2] == -8
        assert runner.state[S.Z] == -8 and runner.state[S.ARMOR] == 100
        assert runner.state[S.HEALTH] == events["ledge:green_armor"]["after"]["health"]
        print("Original-map controls-only source ledge witness:", json.dumps(
            {"ticks": probe.steps, "events": [{"name": row["name"], "tick": row["tick"],
              "before": row["before"]["position"], "after": row["after"]["position"],
              "sector": row["after"]["sector"]} for row in probe.events]}))
    finally:
        torch.set_num_threads(previous)
