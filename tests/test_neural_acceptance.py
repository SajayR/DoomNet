"""Meaningful failures and provenance checks for strict neural acceptance."""

from types import SimpleNamespace

import pytest
import torch

from doomnet import state as S
from doomnet.runtime import CONTROL_NAMES, MAX_DT
from scripts import validate_neural
from scripts.validate_neural import (
    CaptureSession, Case, NeuralProbe, ObservedRoute, assess_accuracy, check_bounds,
    compare_frames, compare_states, controls, deployment_parity, diagnose, fields,
    live_choices, restart, validate_artifact,
)


@pytest.fixture
def layout():
    return S.StateLayout(2, 2, 2)


@pytest.fixture
def data():
    # Minimal construction data for testing capture provenance, never a route
    # completion claim or a replacement for the original-level corpus.
    return {
        "vertices": torch.tensor([[0., 0.], [0., 100.], [100., 100.], [100., 0.]]),
        "lines": torch.tensor([[0, 1], [2, 3]]),
        "line_front_sector": torch.tensor([0, 0]),
        "line_back_sector": torch.tensor([-1, -1]), "line_flags": torch.zeros(2).long(),
        "node_weight": torch.tensor([[1., 0.]]), "node_bias": torch.tensor([-50.]),
        "node_origin": torch.tensor([[50., 0.]]), "node_children": torch.tensor([[-1, -2]]),
        "subsector_sector": torch.tensor([0, 1]),
        "sector_floor": torch.tensor([0., 0.]), "sector_ceiling": torch.tensor([128., 128.]),
        "sector_special": torch.zeros(2), "player_start": torch.tensor([32., 32., 0.]),
        "entity_pos": torch.tensor([[60., 60.], [70., 70.]]),
        "entity_type": torch.tensor([2012, 2014]), "entity_angle": torch.zeros(2),
    }


def test_state_metrics_name_the_worst_scalar_and_array_element(layout):
    expected = torch.zeros(layout.size)
    actual = expected.clone()
    actual[S.X] = 2
    actual[layout.projectile_z] = torch.tensor([.25, -3.])
    report = compare_states(expected, actual, layout)
    assert report["max_abs"] == 3
    assert report["fields"]["x"]["worst"] == {"index": 0, "expected": 0., "actual": 2.}
    assert report["fields"]["projectile_z"]["mean_abs"] == 1.625
    assert report["fields"]["projectile_z"]["worst"] == {"index": 1, "expected": 0., "actual": -3.}
    assert sum(field.stop - field.start for field in fields(layout).values()) == layout.size


def test_discrete_outcomes_are_reported_even_for_tiny_absolute_error(layout):
    expected = torch.zeros(layout.size)
    actual = expected.clone()
    actual[S.HEALTH] = 1e-6
    actual[layout.entity_hp][1] = 1e-7
    actual[layout.entity_alive][1] = 1
    report = compare_states(expected, actual, layout)
    assert report["semantic_mismatches"]["player_alive"]["actual"] == [True]
    assert report["semantic_mismatches"]["entity_hp_positive"]["indices"] == [1]
    assert report["semantic_mismatches"]["entity_alive"]["expected"] == [False]


def test_integer_code_observation_keeps_the_raw_fractional_error(layout):
    expected = torch.zeros(layout.size)
    expected[S.WEAPON] = 1
    actual = expected.clone()
    actual[S.WEAPON] = 1.125
    report = compare_states(expected, actual, layout)
    assert "weapon" not in report["semantic_mismatches"]
    assert report["code_distance_to_integer"]["weapon"] == .125
    assert report["fields"]["weapon"]["max_abs"] == .125


def test_angle_wrap_is_not_mistaken_for_a_large_camera_rotation(layout):
    expected = torch.zeros(layout.size)
    expected[S.ANGLE] = 2 * torch.pi - .001
    actual = expected.clone()
    actual[S.ANGLE] = .001
    report = compare_states(expected, actual, layout)
    assert report["fields"]["angle"]["max_abs"] > 6
    assert report["angle_circular_abs"] == pytest.approx(.002, abs=1e-6)
    assert report["fields"]["angle"]["circular_max_abs"] == report["angle_circular_abs"]


def test_entity_circular_worst_identifies_the_largest_physical_rotation(layout):
    expected = torch.zeros(layout.size)
    expected[layout.entity_angle] = torch.tensor([2 * torch.pi - .001, 0.])
    actual = expected.clone()
    actual[layout.entity_angle] = torch.tensor([.001, .2])
    report = compare_states(expected, actual, layout)
    metrics = report["fields"]["entity_angle"]
    assert metrics["worst"]["index"] == 0
    assert metrics["circular_worst"]["index"] == 1
    assert metrics["circular_max_abs"] == pytest.approx(.2)
    assert metrics["circular_mean_abs"] == pytest.approx(.101, abs=1e-6)


def test_frame_metrics_use_pixels_and_do_not_overflow_uint8():
    expected = torch.zeros((2, 2, 3), dtype=torch.uint8)
    actual = expected.clone()
    actual[0, 0] = torch.tensor([255, 10, 0], dtype=torch.uint8)
    report = compare_frames(expected, actual)
    assert report["max_abs"] == 255
    assert report["mean_abs"] == pytest.approx(265 / 12)
    assert report["channel_mean_abs"] == [63.75, 2.5, 0.]
    assert report["changed_pixel_fraction"] == .25
    assert report["pixel_error_over_8_fraction"] == .25
    assert report["pixel_error_over_32_fraction"] == .25
    assert compare_frames(actual, expected) == report


@pytest.mark.parametrize("bad", [torch.zeros(2), torch.zeros((2, 2, 3)),
                                   torch.zeros((2, 2, 4), dtype=torch.uint8)])
def test_invalid_frame_contract_fails_before_accuracy_claim(bad):
    with pytest.raises(ValueError, match="uint8 HxWx3"):
        compare_frames(torch.zeros((2, 2, 3), dtype=torch.uint8), bad)


def test_bounds_failure_identifies_returned_field_and_element(layout):
    state = torch.zeros(layout.size)
    lower, upper = torch.full_like(state, -1), torch.ones_like(state)
    state[layout.projectile_z][1] = 2
    with pytest.raises(ValueError, match=r"projectile_z\[1\].*2.0 outside \[-1.0, 1.0\]"):
        check_bounds(state, lower, upper, layout)


def test_source_error_never_passes_without_a_complete_reviewed_budget(layout):
    expected = torch.zeros(layout.size)
    actual = expected.clone()
    actual[S.X] = .5
    frame_error = compare_frames(torch.zeros((1, 1, 3), dtype=torch.uint8),
                                 torch.ones((1, 1, 3), dtype=torch.uint8))
    diagnostic = {"name": "door/body-overlap", "states": {"source_forward": compare_states(expected, actual, layout)},
                  "frames": {"same_state_render": frame_error, "forward_frame": frame_error}}
    with pytest.raises(ValueError, match="complete reviewed numeric budget"):
        assess_accuracy([diagnostic], None)
    budget = {"state_max_abs": {name: 0. for name in fields(layout)},
              "frame_mean_abs": .25, "frame_error_over_8_fraction": 0.,
              "frame_error_over_32_fraction": 0.}
    report = assess_accuracy([diagnostic], budget)
    assert report["status"] == "failed"
    assert any("door/body-overlap: source_forward x max_abs" in reason for reason in report["failures"])
    assert any("same_state_render RGB mean_abs" in reason for reason in report["failures"])
    assert any("forward_frame RGB mean_abs" in reason for reason in report["failures"])
    del budget["state_max_abs"]["rng"]
    with pytest.raises(ValueError, match="every state field"):
        assess_accuracy([diagnostic], budget)


def test_reviewed_budget_covers_every_field_and_preserves_exact_discrete_states(layout):
    budget = validate_neural.ACCURACY_BUDGET
    limits = budget["state_max_abs"]
    assert set(limits) == set(fields(layout))
    for name in ("bullets", "shells", "weapon", "shotgun_owned", "kills", "won", "secrets",
                 "use_held", "armor_type", "rng", "entity_active", "entity_alive", "entity_drop",
                 "projectile_alive", "sector_motion", "sector_floor_motion", "sector_secret", "line_used"):
        assert limits[name] == 0, f"{name} must remain exact"
    assert limits["projectile_dz"] == .1
    assert limits["entity_angle"] == 1e-4
    assert limits["time"] == 1e-5
    assert budget["frame_mean_abs"] == .5
    assert budget["frame_error_over_8_fraction"] == .015
    assert budget["frame_error_over_32_fraction"] == .002


def test_reviewed_angular_budget_uses_physical_rotation_error(layout):
    expected = torch.zeros(layout.size)
    expected[S.ANGLE] = 2 * torch.pi - .001
    actual = expected.clone()
    actual[S.ANGLE] = .001
    frame = torch.zeros((1, 1, 3), dtype=torch.uint8)
    diagnostic = {"name": "angle-wrap", "states": {"source_forward": compare_states(expected, actual, layout)},
                  "frames": {"same_state_render": compare_frames(frame, frame), "forward_frame": compare_frames(frame, frame)}}
    budget = {"state_max_abs": {name: 0. for name in fields(layout)}, "frame_mean_abs": 0.,
              "frame_error_over_8_fraction": 0., "frame_error_over_32_fraction": 0.}
    budget["state_max_abs"]["angle"] = .003
    assert assess_accuracy([diagnostic], budget)["status"] == "passed"


@pytest.mark.parametrize("kind", ["same_state_render", "forward_frame"])
def test_rgb_outlier_limits_catch_sparse_large_errors_below_the_mean_limit(layout, kind):
    state = torch.zeros(layout.size)
    frame = torch.zeros((10, 10, 3), dtype=torch.uint8)
    actual = frame.clone()
    actual[0, 0] = 255
    diagnostic = {"name": "sparse-render-error", "states": {"source_forward": compare_states(state, state, layout)},
                  "frames": {"same_state_render": compare_frames(frame, frame), "forward_frame": compare_frames(frame, frame)}}
    diagnostic["frames"][kind] = compare_frames(frame, actual)
    budget = {"state_max_abs": {name: 0. for name in fields(layout)}, "frame_mean_abs": 3.,
              "frame_error_over_8_fraction": 0., "frame_error_over_32_fraction": 0.}
    report = assess_accuracy([diagnostic], budget)
    assert report["status"] == "failed"
    assert len(report["failures"]) == 2
    assert all("fraction" in reason for reason in report["failures"])


def test_empty_accuracy_corpus_cannot_claim_acceptance():
    with pytest.raises(ValueError, match="must not be empty"):
        assess_accuracy([])


def test_semantic_codes_cannot_be_relaxed_by_a_numeric_budget(layout):
    expected = torch.zeros(layout.size)
    actual = expected.clone()
    actual[S.WEAPON] = 1
    diagnostic = {"name": "wrong-weapon", "states": {"saved_deployed_forward": compare_states(expected, actual, layout)},
                  "frames": {}}
    budget = {"state_max_abs": {name: 10. for name in fields(layout)}, "frame_mean_abs": 255.,
              "frame_error_over_8_fraction": 1., "frame_error_over_32_fraction": 1.}
    report = assess_accuracy([diagnostic], budget)
    assert report["status"] == "failed"
    assert report["failures"] == ["wrong-weapon: saved_deployed_forward discrete semantics differ: weapon"]


class AddX:
    def transition(self, state, control, dt):
        result = state.clone()
        result[S.X] += control[0] * dt[0]
        return result


def test_capture_preserves_control_only_state_provenance(data):
    model, cases = AddX(), []
    session = CaptureSession(model, data, "movement", cases)
    initial = session.state.clone()
    session.step(controls(forward=1))
    first_return = session.state.clone()
    session.step(controls(forward=-1), tags=("backward",))
    assert torch.equal(cases[0].state, initial)
    assert torch.equal(cases[1].state, first_return)
    assert torch.equal(cases[0].expected, cases[1].state)
    assert torch.equal(session.state, model.transition(first_return, controls(forward=-1), torch.tensor([MAX_DT])))
    with torch.inference_mode():
        session.state[S.X] = 0
    assert torch.equal(cases[1].state, first_return), "stored diagnostic input must not alias live state"


def test_capture_retains_first_three_launches_and_contacts_per_slot(data):
    layout = S.StateLayout.from_data(data)

    class ProjectileCycle:
        def transition(self, state, control, dt):
            result = state.clone()
            result[layout.projectile_alive][0] = control[3]
            result[layout.projectile_dx][0] += control[3]
            return result

    cases = []
    session = CaptureSession(ProjectileCycle(), data, "projectile-cycle", cases)
    for _ in range(5):
        session.step(controls(fire=1))
        session.step()
    assert session.projectile_event_counts == {"projectile_launch:0": 5, "projectile_contact:0": 5}
    assert sum("projectile_launch:0" in case.tags for case in cases) == 3
    assert sum("projectile_contact:0" in case.tags for case in cases) == 3


def test_capture_records_a_launch_that_contacts_within_the_same_tick(data):
    layout = S.StateLayout.from_data(data)

    class ImmediateContact:
        def transition(self, state, control, dt):
            result = state.clone()
            result[layout.projectile_dx][0] = 1
            return result

    cases = []
    session = CaptureSession(ImmediateContact(), data, "immediate-contact", cases)
    session.step()
    assert {"projectile_launch:0", "projectile_contact:0"} <= set(cases[0].tags)
    assert session.state[layout.projectile_alive][0] == 0


def test_diagnostics_distinguish_wrong_semantics_and_full_split_output(tmp_path, layout):
    original = torch.zeros(layout.size)
    reference_frame = torch.zeros((2, 2, 3), dtype=torch.uint8)
    result = original.clone()
    result[S.WON] = 1
    result[S.X] = .25
    executor = SimpleNamespace(
        forward=lambda *args: (result.clone(), reference_frame + 7),
        transition=lambda *args: result.clone(),
        render=lambda state: reference_frame.clone(),
    )
    runner = SimpleNamespace(device=torch.device("cpu"), executor=executor, state=original,
                             state_lower=torch.zeros(layout.size), state_upper=torch.ones(layout.size))
    source = SimpleNamespace(render=lambda state: reference_frame.clone())
    case = Case("zero-time/live-actions", ("zero_dt_identity",), original,
                controls(forward=1, fire=1), 0., original.clone())
    rows, failures, worst = diagnose(runner, source, [case], layout, tmp_path)
    assert rows[0]["full_split_exact"] == {"state": True, "frame": False}
    assert rows[0]["zero_dt_identity"] is False
    assert rows[0]["states"]["source_forward"]["semantic_mismatches"]["won"]["actual"] == [True]
    assert rows[0]["frames"]["same_state_render"]["mean_abs"] == 0
    assert rows[0]["frames"]["forward_frame"]["mean_abs"] == 7
    assert all("zero-time/live-actions" in reason for reason in failures)
    assert len(failures) == 3
    assert worst["same_state_render"]["difference_display_gain"] == 4
    assert worst["forward_frame"]["mean_abs"] == 7
    assert (tmp_path / "worst-forward-frame-difference-x4.png").is_file()


def test_saved_deployed_measurements_budget_bit_differences_and_check_lifecycle(layout):
    original = torch.zeros(layout.size)
    original[S.HEALTH] = 100
    frame = torch.zeros((2, 2, 3), dtype=torch.uint8)

    def backend(error, color):
        def transition(state, control, dt):
            if control[5] > .5:
                return original.clone()
            result = state.clone()
            if dt[0] > 0 and state[S.HEALTH] > 0 and state[S.WON] < .5:
                result[S.X] += error
            return result

        def forward(state, control, dt):
            result = transition(state, control, dt)
            if dt[0] > 0 and state[S.HEALTH] > 0 and state[S.WON] < .5 and control[5] < .5:
                result[S.X] += error
            return result, frame + color

        return SimpleNamespace(forward=forward, transition=transition, render=lambda state: frame + color)

    cases = [Case(name, tags, original, controls(forward=1), MAX_DT, original)
             for name, tags in (("spawn", ()), ("launch", ("projectile_launch:0",)), ("door", ("door:4",)))]
    for tag in validate_neural.EXACT_STATE_TAGS:
        state = original.clone()
        if tag == "dead_freeze":
            state[S.HEALTH] = 0
        elif tag == "won_freeze":
            state[S.WON] = 1
        elif tag == "restart_exact":
            state[S.X] = .5
        cases.append(Case(tag, (tag,), state, controls(restart=int(tag == "restart_exact")),
                          0. if tag == "zero_dt_identity" else MAX_DT,
                          original if tag == "restart_exact" else state))
    runner = SimpleNamespace(device=torch.device("cpu"), model=backend(0, 0), executor=backend(1e-5, 1),
                             state_lower=torch.full((layout.size,), -1.), state_upper=torch.full((layout.size,), 200.))
    rows = deployment_parity(runner, SimpleNamespace(render=lambda state: frame), cases, layout)
    assert [row["name"] for row in rows] == [case.name for case in cases]
    assert not rows[0]["exact"]["saved_deployed_forward_state"]
    assert not rows[0]["exact"]["deployed_full_split_state"]
    assert not rows[0]["exact"]["saved_deployed_forward_frame"]
    assert all(all(row[f"{name}_exact_checks"].values()) for row in rows for name in ("saved", "deployed"))
    budget = {"state_max_abs": {name: 0. for name in fields(layout)}, "frame_mean_abs": 1.,
              "frame_error_over_8_fraction": 0., "frame_error_over_32_fraction": 0.}
    budget["state_max_abs"]["x"] = 3e-5
    assert assess_accuracy(rows, budget)["status"] == "passed"
    budget["state_max_abs"]["x"] = 0.
    budget["frame_mean_abs"] = 0.
    report = assess_accuracy(rows, budget)
    assert report["status"] == "failed"
    assert any("saved_deployed_forward x max_abs" in reason for reason in report["failures"])
    assert any("deployed_full_split x max_abs" in reason for reason in report["failures"])
    assert any("saved_deployed_transition_render RGB mean_abs" in reason for reason in report["failures"])


def test_neural_probe_carries_only_executor_returned_state(data):
    spawn = S.initial_state(data)
    returned = spawn.clone()
    returned[S.X] += 1
    inputs = []

    def transition(state, control, dt):
        inputs.append(state)
        return returned

    runner = SimpleNamespace(state=spawn, state_lower=S.state_bounds(data)[0], state_upper=S.state_bounds(data)[1],
                             executor=SimpleNamespace(transition=transition),
                             _inputs=lambda control, dt: (control, torch.tensor([dt])))
    probe = NeuralProbe(runner, S.StateLayout.from_data(data))
    probe.step(controls(forward=1))
    assert inputs == [spawn]
    assert runner.state is returned
    assert probe.state is returned


class ChoiceRunner:
    """Synthetic failure-test runner; these tests make no level-completion claim."""

    def __init__(self, spawn, *, move=True, render=True):
        self.spawn, self._state = spawn, spawn.clone()
        self.move, self.render, self.restarts = move, render, 0

    @property
    def state(self):
        # No setter: live-choice assessment may only observe returned state.
        return self._state

    def step(self, control, dt):
        if control[CONTROL_NAMES.index("restart")] > .5:
            self._state = self.spawn.clone()
            self.restarts += 1
        else:
            self._state = self._state.clone()
            self._state[S.Y] += control[0] * 180 * dt
            self._state[S.X] += control[1] * 180 * dt * self.move
        return torch.full((1, 1, 3), int(self.state[S.X]) if self.render else 0, dtype=torch.uint8)


def test_live_choice_check_uses_two_neural_restarts_and_observed_states(data):
    runner = ChoiceRunner(S.initial_state(data))
    report = live_choices(runner, runner.spawn)
    assert report["status"] == "passed"
    assert runner.restarts == 2
    assert report["position_separation"][0] > 20


@pytest.mark.parametrize("move,render,reason", [(False, True, "lateral motion"), (True, False, "identical neural RGB")])
def test_live_choice_check_rejects_ignored_controls_or_fixed_images(data, move, render, reason):
    runner = ChoiceRunner(S.initial_state(data), move=move, render=render)
    with pytest.raises(RuntimeError, match=reason):
        live_choices(runner, runner.spawn)


def test_restart_cannot_restore_broken_returned_activations_by_host_assignment(data):
    spawn = S.initial_state(data)
    broken = spawn.clone()
    broken[S.HEALTH] -= 1
    runner = SimpleNamespace(state=broken, step=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="every original spawn activation exactly"):
        restart(runner, spawn, "regression")
    assert runner.state is broken


def test_source_shadow_cannot_repair_the_returned_neural_route_state(data):
    initial = S.initial_state(data)

    class Runner:
        state = initial.clone()
        state[S.ANGLE] = 2 * torch.pi
        state_lower, state_upper = S.state_bounds(data)

        def step(self, control, dt):
            self.state = self.state.clone()
            self.state[S.X] += 2 * control[0] * dt
            return torch.zeros((1, 1, 3), dtype=torch.uint8)

    route = ObservedRoute(Runner(), AddX(), data)
    for _ in range(3):
        route.step(controls(forward=1), MAX_DT)
    assert route.state[S.X] > route.shadow[S.X]
    assert torch.equal(route.state, route.runner.state)
    report = route.drift()
    assert report["ticks"] == 3
    assert report["fields"]["x"]["max_abs_over_play"] > .08
    assert report["final"]["fields"]["x"]["max_abs"] > .08
    assert report["fields"]["angle"]["max_abs_over_play"] > 6
    assert report["fields"]["angle"]["circular_max_abs_over_play"] < 1e-6


@pytest.mark.parametrize("corrupt_won_freeze", [False, True])
def test_artifact_acceptance_assesses_deployment_and_restarts_every_run(tmp_path, data, monkeypatch, corrupt_won_freeze):
    spawn, layout = S.initial_state(data), S.StateLayout.from_data(data)
    frame = torch.zeros((1, 1, 3), dtype=torch.uint8)

    class Runner:
        state = spawn.clone()
        state_lower, state_upper = S.state_bounds(data)
        manifest = {"files": {}, "export": {"exact_lowered_eager_and_full_split_checks": 1}}
        info = {}
        restarts = 0

        def step(self, control, dt):
            if control[5] > .5:
                self.state = spawn.clone()
                self.restarts += 1
            else:
                self.state = self.state.clone()
                if self.state[S.WON] > .5:
                    self.state[S.HEALTH] -= int(corrupt_won_freeze)
                elif control[4] > .5:
                    self.state[S.WON] = 1
            return frame

    def source_transition(state, control, dt):
        result = state.clone()
        if result[S.WON] < .5 and control[4] > .5:
            result[S.WON] = 1
        return result

    def reproduction(factory, data):
        for name in ("wall_collision", "corner_shot", "door_occupancy", "clock_wrap"):
            assert torch.equal(factory(name).state, spawn)
        return {"status": "passed"}

    def lifecycle(factory, data):
        assert torch.equal(factory("death_restart").state, spawn)
        return {"status": "passed"}

    def choices(runner, spawn):
        for name in ("left", "right"):
            restart(runner, spawn, name)
        return {"status": "passed"}

    def route(session, *args):
        assert torch.equal(session.state, spawn)
        return {"status": "passed"}, session.step(controls(use=1), MAX_DT)

    diagnostic = {"name": "source-probe", "states": {"source_forward": compare_states(spawn, spawn, layout)}, "frames": {}}
    deployed = {"name": "deployment-probe", "states": {"saved_deployed_forward": compare_states(spawn, spawn, layout)},
                "frames": {}, "saved_exact_checks": {"restart_exact": True}, "deployed_exact_checks": {"restart_exact": True}}
    assessed = []

    def accuracy(rows):
        assessed.extend(rows)
        return assess_accuracy(rows)

    monkeypatch.setattr(validate_neural, "diagnose", lambda *args: ([diagnostic], [], {}))
    monkeypatch.setattr(validate_neural, "deployment_parity", lambda *args: [deployed])
    monkeypatch.setattr(validate_neural, "regressions", reproduction)
    monkeypatch.setattr(validate_neural, "lifecycle", lifecycle)
    monkeypatch.setattr(validate_neural, "live_choices", choices)
    monkeypatch.setattr(validate_neural, "run_acceptance", route)
    monkeypatch.setattr(validate_neural, "assess_accuracy", accuracy)
    runner = Runner()
    report = validate_artifact(runner, SimpleNamespace(transition=source_transition), [], data, {}, tmp_path)
    assert runner.restarts == 9
    assert assessed == [diagnostic, deployed]
    assert report["deployment_diagnostics"] == [deployed]
    assert report["status"] == ("failed" if corrupt_won_freeze else "passed")
    for branch in ("left", "right"):
        if corrupt_won_freeze:
            assert report["returned_state_routes"][branch]["status"] == "failed"
            assert any("won neural transition changed" in reason for reason in report["failures"])
        else:
            assert report["returned_state_routes"][branch]["won_freeze_ticks"] == 3
            assert report["returned_state_routes"][branch]["bounds_checked_ticks"] == 4
