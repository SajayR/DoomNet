"""Strict neural E1M1 acceptance and separate source-accuracy diagnostics.

``python -m scripts.validate_neural --wad assets/doom1.wad`` captures a source
corpus only. Mandatory actual-artifact acceptance is
``python -m scripts.validate_neural --wad assets/doom1.wad
--model build/full-neural-e1m1 --device cuda --threads 1``. It tests both complete
routes, controls-only regressions, and diverging live choices on one v2 runner.
Corpus states are diagnostic inputs, never replacements for returned neural
states in the gameplay routes or controls-only regression reproductions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from doomnet import state as S
from doomnet.runtime import CONTROL_NAMES, MAX_DT, NO_CONTROLS
from scripts.validate_level import AcceptanceFailure, E1M1Controller, run_acceptance


ANGULAR_FIELDS = ("angle", "entity_angle")
EXACT_STATE_TAGS = ("zero_dt_identity", "restart_exact", "dead_freeze", "won_freeze")


def construction_provenance(source):
    from doomnet.model import _tensor_manifest

    root = Path(S.__file__).resolve().parents[1]
    paths = {Path(S.__file__).resolve()}
    paths.update(Path(inspect.getsourcefile(type(module))).resolve() for module in source.modules()
                 if type(module).__module__.startswith("doomnet."))
    code = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}
    fixed = _tensor_manifest(dict(source.state_dict()))["sha256"]
    digest = hashlib.sha256(json.dumps({"code": code, "fixed_sha256": fixed}, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "code_files": code, "fixed_sha256": fixed}


def fields(layout):
    result = {name.lower(): slice(index, index + 1)
              for name, index in vars(S).items()
              if name.isupper() and isinstance(index, int) and 0 <= index < S.SCALAR_COUNT}
    result.update({name: getattr(layout, name)
                   for name in (*S.ENTITY_FIELDS, *S.SECTOR_FIELDS, "line_used")})
    assert sorted(i for field in result.values() for i in range(field.start, field.stop)) == list(range(layout.size)), "state field report must cover every activation once"
    return result


# Reviewed against the complete v2 artifact's saved-GPU render and transition
# diagnostics. Discrete flags, codes, ammunition, counts, RNG, and motion states
# remain exact; angular limits measure circular distance in radians.
ACCURACY_BUDGET = {"state_max_abs": dict.fromkeys(fields(S.StateLayout(1, 1, 1)), 0.),
                   "frame_mean_abs": .5, "frame_error_over_8_fraction": .015,
                   "frame_error_over_32_fraction": .002}
for limit, names in (
    (.01, "x y z entity_x entity_y"),
    (.05, "projectile_x projectile_y projectile_z"),
    (.1, "projectile_dx projectile_dy projectile_dz"),
    (1e-4, "angle entity_angle health armor entity_hp cooldown entity_cooldown damage_flash fire_flash pickup_flash sector_timer sector_floor_timer"),
    (1e-5, "time"),
    (.001, "sector_floor sector_ceiling"),
):
    ACCURACY_BUDGET["state_max_abs"].update(dict.fromkeys(names.split(), limit))


def _state(value, layout):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != (layout.size,):
        raise ValueError("accuracy state must be a flat float32 tensor matching StateLayout")
    value = value.detach().cpu()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("accuracy state contains nonfinite activations")
    return value


def semantics(state, layout):
    """Observe actual branch meanings, rather than treating all floats alike."""
    state_fields = fields(layout)
    result = {"player_alive": state[S.HEALTH:S.HEALTH + 1] > 0,
              "entity_hp_positive": state[layout.entity_hp] > 0}
    for name in ("won", "shotgun_owned", "use_held", "entity_active", "entity_alive",
                 "entity_drop", "projectile_alive", "sector_secret", "line_used"):
        result[name] = state[state_fields[name]] > .5
    # These activations represent integer codes/counts. Rounding here observes
    # their discrete meaning; raw error and distance to integers are reported.
    for name in ("bullets", "shells", "weapon", "kills", "secrets", "armor_type", "rng",
                 "sector_motion", "sector_floor_motion"):
        result[name] = state[state_fields[name]].round()
    return result


def circular_error(expected, actual):
    return ((actual.double() - expected.double() + math.pi) % (2 * math.pi) - math.pi).abs()


def compare_states(expected, actual, layout):
    expected, actual = _state(expected, layout), _state(actual, layout)
    error = (actual.double() - expected.double()).abs()
    measurements = {}
    for name, field in fields(layout).items():
        values = error[field]
        local = int(values.argmax())
        index = field.start + local
        measurements[name] = {"max_abs": float(values.max()), "mean_abs": float(values.mean()),
                              "p95_abs": float(torch.quantile(values, .95)),
                              "worst": {"index": local, "expected": float(expected[index]),
                                        "actual": float(actual[index])}}
        if name in ANGULAR_FIELDS:
            circular = circular_error(expected[field], actual[field])
            circular_index = int(circular.argmax())
            measurements[name].update(circular_max_abs=float(circular.max()),
                                      circular_mean_abs=float(circular.mean()),
                                      circular_p95_abs=float(torch.quantile(circular, .95)),
                                      circular_worst={"index": circular_index,
                                                      "expected": float(expected[field][circular_index]),
                                                      "actual": float(actual[field][circular_index])})
    observed, reference = semantics(actual, layout), semantics(expected, layout)
    mismatches = {}
    for name, values in observed.items():
        indices = torch.nonzero(values != reference[name]).flatten().tolist()
        if indices:
            mismatches[name] = {"count": len(indices), "indices": indices,
                                "expected": reference[name][indices].tolist(),
                                "actual": values[indices].tolist()}
    return {"fields": measurements, "max_abs": float(error.max()),
            "angle_circular_abs": measurements["angle"]["circular_max_abs"], "semantic_mismatches": mismatches,
            "code_distance_to_integer": {name: float((actual[fields(layout)[name]] - values).abs().max())
                                         for name, values in observed.items() if values.dtype != torch.bool}}


def compare_frames(expected, actual):
    for frame in (expected, actual):
        if not isinstance(frame, torch.Tensor) or frame.dtype != torch.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError("accuracy image must be a uint8 HxWx3 RGB tensor")
    if expected.shape != actual.shape:
        raise ValueError("accuracy image shapes differ")
    delta = (actual.detach().cpu().double() - expected.detach().cpu().double()).abs()
    pixel_error = delta.amax(-1)
    return {"shape": list(expected.shape), "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()), "rmse": float(delta.square().mean().sqrt()),
            "p95_abs": float(torch.quantile(delta.flatten(), .95)),
            "channel_mean_abs": delta.mean((0, 1)).tolist(),
            "changed_pixel_fraction": float((pixel_error > 0).double().mean()),
            "pixel_error_over_8_fraction": float((pixel_error > 8).double().mean()),
            "pixel_error_over_32_fraction": float((pixel_error > 32).double().mean())}


def check_bounds(state, lower, upper, layout):
    value = _state(state, layout)
    lower, upper = lower.detach().cpu(), upper.detach().cpu()
    if tuple(lower.shape) != (layout.size,) or lower.shape != upper.shape:
        raise ValueError("state bounds must match StateLayout")
    invalid = (value < lower) | (value > upper)
    if bool(invalid.any()):
        index = int(torch.nonzero(invalid)[0])
        name, field = next((name, field) for name, field in fields(layout).items()
                           if field.start <= index < field.stop)
        raise ValueError(f"returned state violates {name}[{index - field.start}] bounds: {float(value[index])} outside [{float(lower[index])}, {float(upper[index])}]")


def controls(**values):
    result = torch.tensor(NO_CONTROLS, dtype=torch.float32)
    for name, value in values.items():
        result[CONTROL_NAMES.index(name)] = value
    return result


@dataclass(frozen=True)
class Case:
    name: str
    tags: tuple[str, ...]
    state: torch.Tensor
    controls: torch.Tensor
    dt: float
    expected: torch.Tensor


class CaptureSession:
    """Source diagnostic session; records only genuinely returned states."""

    def __init__(self, model, data, name, cases):
        self.model, self.layout = model, S.StateLayout.from_data(data)
        self.state = S.initial_state(data)
        self.name, self.cases, self.index = name, cases, 0
        self.seen = set()
        self.projectile_event_counts = {}
        self.lower, self.upper = S.state_bounds(data)

    def step(self, control=NO_CONTROLS, dt=MAX_DT, tags=()):
        control = torch.as_tensor(control, dtype=torch.float32)
        before = self.state
        with torch.inference_mode():
            self.state = self.model.transition(before, control, torch.tensor([dt], dtype=torch.float32))
        check_bounds(self.state, self.lower, self.upper, self.layout)
        self.index += 1
        events = set(tags)
        restarting = bool(control[CONTROL_NAMES.index("restart")] > .5)
        if self.index == 1 or self.index % 160 == 0:
            events.add("periodic_view")
        for i, name in enumerate(CONTROL_NAMES):
            if float(control[i]) != (-1 if i == 6 else 0):
                events.add(f"control:{name}")
        for condition, name in ((self.state[S.HEALTH] < before[S.HEALTH], "damage"),
                                (self.state[S.KILLS] > before[S.KILLS], f"kill:{int(self.state[S.KILLS])}"),
                                (self.state[S.HEALTH] <= 0, "death"), (self.state[S.WON] > .5, "won"),
                                ((self.state[S.TIME] < before[S.TIME]) & (not restarting), "clock_wrap")):
            if bool(condition):
                events.add(name)
        was_alive = before[self.layout.projectile_alive] > .5
        alive = self.state[self.layout.projectile_alive] > .5
        velocity_changed = torch.stack([self.state[getattr(self.layout, name)] != before[getattr(self.layout, name)]
                                         for name in ("projectile_dx", "projectile_dy", "projectile_dz")]).any(0)
        launched = ~was_alive & (alive | velocity_changed) & (not restarting)
        contacted = (was_alive | launched) & ~alive & (not restarting)
        projectile_events = set()
        for changed, name in ((launched, "projectile_launch"), (contacted, "projectile_contact")):
            for index in torch.nonzero(changed).flatten().tolist():
                tag = f"{name}:{index}"
                count = self.projectile_event_counts.get(tag, 0) + 1
                self.projectile_event_counts[tag] = count
                if count <= 3:
                    projectile_events.add(tag)
        for field, name in ((self.layout.sector_ceiling, "door"),
                            (self.layout.sector_floor, "floor")):
            for index in torch.nonzero(self.state[field] != before[field]).flatten().tolist():
                events.add(f"{name}:{index}")
        pickup = (before[self.layout.entity_alive] > .5) & (self.state[self.layout.entity_alive] < .5) & (before[self.layout.entity_hp] == 0)
        events.update(f"pickup:{index}" for index in torch.nonzero(pickup).flatten().tolist())
        fresh = events - self.seen
        self.seen.update(events)
        selected = fresh | set(tags) | projectile_events | ({"periodic_view"} if self.index % 160 == 0 else set())
        if selected:
            self.cases.append(Case(f"{self.name}/{self.index:04d}", tuple(sorted(selected)),
                                   before.detach().cpu().clone(), control.clone(), dt,
                                   self.state.detach().cpu().clone()))
        return None


class Measurements:
    """Independent map measurements used only to assess regression outcomes."""

    def __init__(self, data):
        self.layout = S.StateLayout.from_data(data)
        self.controller = E1M1Controller(data)
        self.data = {name: value.detach().cpu().numpy() for name, value in data.items()}
        vertices, lines = self.data["vertices"], self.data["lines"]
        self.start = vertices[lines[:, 0]].astype(np.float64)
        self.delta = vertices[lines[:, 1]].astype(np.float64) - self.start
        self.front, self.back = self.data["line_front_sector"], self.data["line_back_sector"]

    def line_distances(self, point):
        projection = np.sum((point - self.start) * self.delta, axis=1) / np.sum(self.delta ** 2, axis=1)
        return np.linalg.norm(self.start + np.clip(projection, 0, 1)[:, None] * self.delta - point, axis=1)

    def wall_clearance(self, tensor):
        state = tensor.detach().cpu().numpy()
        floor, ceiling = state[self.layout.sector_floor], state[self.layout.sector_ceiling]
        back = np.maximum(self.back, 0)
        bottom = np.maximum(floor[self.front], floor[back])
        top = np.minimum(ceiling[self.front], ceiling[back])
        blocking = ((self.back < 0) | ((self.data["line_flags"] & 1) != 0)
                    | (bottom > state[S.Z] + 24) | (top < np.maximum(bottom, state[S.Z]) + 56))
        distances = self.line_distances(state[S.X:S.Y + 1])
        return {"minimum_blocking_line_distance": float(distances[blocking].min()),
                "penetrating_lines": np.flatnonzero(blocking & (distances < 16)).tolist()}

    def corner_blockers(self, before, after):
        before, after = before.detach().cpu().numpy(), after.detach().cpu().numpy()
        direction = np.array((math.cos(float(after[S.ANGLE])), math.sin(float(after[S.ANGLE]))))
        relative = self.start - before[S.X:S.Y + 1]
        denominator = direction[0] * self.delta[:, 1] - direction[1] * self.delta[:, 0]
        parallel = np.abs(denominator) < 1e-7
        divisor = np.where(parallel, 1, denominator)
        distance = (relative[:, 0] * self.delta[:, 1] - relative[:, 1] * self.delta[:, 0]) / divisor
        segment = (relative[:, 0] * direction[1] - relative[:, 1] * direction[0]) / divisor
        hit = ~parallel & (distance > 1e-4) & (segment >= 0) & (segment <= 1)
        target = np.array((before[self.layout.entity_x][5], before[self.layout.entity_y][5]))
        offset = target - before[S.X:S.Y + 1]
        side = float(direction[0] * offset[1] - direction[1] * offset[0])
        impact = max(0, float(offset @ direction) - math.sqrt(max(0, 20 ** 2 - side ** 2)))
        blockers = np.flatnonzero(hit & (self.back < 0) & (distance < impact)).tolist()
        return {"blocking_lines": blockers, "target_sideways_distance": side,
                "target_impact_distance": impact,
                "required_line_distances": {str(i): float(distance[i]) for i in (383, 381)},
                "required_lines_hit": bool(hit[383] and hit[381])}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def regressions(factory, data):
    """Every repro starts at spawn and uses controls + returned states only."""
    layout, measures = S.StateLayout.from_data(data), Measurements(data)
    report = {}
    session = factory("wall_collision")
    spawn = session.state.clone()
    for _ in range(45):
        previous = session.state
        session.step(controls(forward=-1, run=1))
        clearance = measures.wall_clearance(session.state)
        _require(not clearance["penetrating_lines"], f"player penetrated wall: {clearance}")
    _require(float((session.state[:2] - spawn[:2]).norm()) > 20, "wall collision repro did not move from spawn")
    _require(torch.equal(previous[:2], session.state[:2]), "wall collision repro failed to stop")
    report["wall_collision"] = dict(clearance, status="passed", ticks=45)

    session = factory("corner_shot")
    controller = E1M1Controller(data, "right")
    for _ in range(617):
        session.step(controller.controls(session.state))
    before = session.state.clone()
    session.step(controls(turn=-.5, fire=1, weapon=1), tags=("blocked_pistol_ray",))
    after = session.state
    observation = measures.corner_blockers(before, after)
    _require(observation["required_lines_hit"] and {383, 381} <= set(observation["blocking_lines"]), f"corner-shot precondition absent: {observation}")
    _require(before[layout.entity_hp][5] == 20 and after[layout.entity_hp][5] == 20, "blocked pistol shot damaged target entity 5")
    _require(after[S.BULLETS] == before[S.BULLETS] - 1 and after[S.FIRE_FLASH] > 0, "corner-shot repro did not actually fire a pistol")
    report["corner_shot"] = dict(observation, status="passed", target_hp=float(after[layout.entity_hp][5]))

    session = factory("door_occupancy")
    controller = E1M1Controller(data, "right")
    for _ in range(500):
        session.step(controller.controls(session.state))
        if "first_door" in controller.reached:
            break
    _require("first_door" in controller.reached, "door occupancy repro did not reach the first door")
    for _ in range(10):
        session.step(controls(forward=float(np.clip((1528 - float(session.state[S.X])) / (180 * MAX_DT), -1, 1))))
    before = session.state.clone()
    point = before[:2].detach().cpu().numpy()
    boundaries = ((measures.front == 4) | (measures.back == 4)) & (measures.front != measures.back)
    overlap = bool(np.any(measures.line_distances(point)[boundaries] < 16))
    _require(measures.controller.sector_at(point) == 3 and overlap, "door occupancy repro body must overlap sector 4 from sector 3")
    _require(before[layout.sector_ceiling][4] - before[layout.sector_floor][4] >= 60, "door occupancy repro door was not open")
    for tick in range(250):
        session.step(tags=("occupied_door_timer_expired",) if tick == 249 else ())
    after = session.state
    clearance = measures.wall_clearance(after)
    _require(after[S.HEALTH] > 0, "player died during door occupancy repro")
    _require(after[layout.sector_ceiling][4] == before[layout.sector_ceiling][4], "door closed into overlapping player body")
    _require(torch.equal(before[:2], after[:2]) and not clearance["penetrating_lines"], "occupied door displaced or obstructed player")
    report["door_occupancy"] = dict(clearance, status="passed", center_sector=3,
                                    body_overlaps=True, ceiling=float(after[layout.sector_ceiling][4]), idle_ticks=250)

    session = factory("clock_wrap")
    for tick in range(1, math.ceil(S.CLOCK_PERIOD / MAX_DT) + 3):
        before = float(session.state[S.TIME])
        session.step()
        after = float(session.state[S.TIME])
        _require(session.state[S.HEALTH] > 0, "idle clock-wrap repro died before a phase wrap")
        if after < before:
            break
    else:
        raise RuntimeError("controls-only clock did not wrap at its declared period")
    _require(0 <= after < MAX_DT * 2 and before > S.CLOCK_PERIOD - MAX_DT * 2,
             f"clock wrap returned an unexpected phase: {before} -> {after}")
    report["clock_wrap"] = {"status": "passed", "ticks": tick, "period": S.CLOCK_PERIOD,
                              "before_phase": before, "after_phase": after}
    return report


def lifecycle(factory, data):
    """Reach exposed enemies by controls, die naturally, then test restart."""
    session = factory("death_restart")
    spawn = session.state.clone()
    action = controls(forward=1, strafe=1, turn=1, fire=1, use=1, weapon=0)
    session.step(action, dt=0, tags=("zero_dt_identity",))
    _require(torch.equal(session.state, spawn), "zero-dt transition changed recurrent state")
    controller = E1M1Controller(data, "right")
    for _ in range(500):
        session.step(controller.controls(session.state))
        if "first_door" in controller.reached:
            break
    _require("first_door" in controller.reached, "death repro did not reach the first door")
    for _ in range(35):
        session.step(controls(forward=1, run=1))
    for tick in range(1, 1201):
        # Fire twice to wake nearby actors while turning away from them.
        session.step(controls(turn=1, fire=int(tick in (1, 8))))
        if session.state[S.HEALTH] <= 0:
            break
    _require(session.state[S.HEALTH] <= 0, "controls-only death repro did not kill player")
    dead = session.state.clone()
    for _ in range(3):
        session.step(action, tags=("dead_freeze",))
        _require(torch.equal(session.state, dead), "dead player transition changed recurrent state")
    session.step(controls(restart=1), dt=0, tags=("restart_exact",))
    _require(torch.equal(session.state, spawn), "restart did not restore every spawn activation exactly")
    return {"status": "passed", "exposed_combat_ticks_until_death": tick,
            "zero_dt_identity": True, "dead_freeze_ticks": 3, "restart_exact": True,
            "state_provenance": "Death reached through returned states and controls; no authored health or world state."}


def build_corpus(data, metadata, width, height, max_steps=6000):
    from doomnet.model import ConstructedDoom

    source = ConstructedDoom(data, width=width, height=height).eval()
    construction = construction_provenance(source)
    cases, route_reports, sessions = [], {}, {}

    def factory(name):
        session = CaptureSession(source, data, name, cases)
        sessions[name] = session
        return session

    for branch in ("left", "right"):
        session = factory(f"route-{branch}")
        route_reports[branch] = run_acceptance(session, data, metadata, branch, max_steps,
                                              require_render=False)[0]
        won = session.state.clone()
        for _ in range(3):
            session.step(controls(forward=1, strafe=1, turn=1, fire=1, use=1, weapon=0), tags=("won_freeze",))
            _require(torch.equal(session.state, won), "won source transition changed recurrent state")
        route_reports[branch]["won_freeze_ticks"] = 3
    behavior = regressions(factory, data)
    life = lifecycle(factory, data)
    for tag in ("projectile_launch:", "projectile_contact:", "clock_wrap"):
        _require(any(any(event.startswith(tag) for event in case.tags) for case in cases),
                 f"source corpus did not capture required {tag.rstrip(':')} event")
    return source, cases, {"routes": route_reports, "regressions": behavior, "lifecycle": life,
                           "projectile_event_counts": {name: session.projectile_event_counts for name, session in sessions.items()},
                           "construction": construction,
                           "construction_consistent_during_capture": construction == construction_provenance(source)}


def save_corpus(cases, source, directory, construction):
    with torch.inference_mode():
        views = torch.stack([source.render(case.state) for case in cases])
        expected_views = torch.stack([source.render(case.expected) for case in cases])
    torch.save({"provenance": "ConstructedDoom original spawn followed by controls-only returned states; diagnostic inputs only",
                "construction": construction,
                "names": [case.name for case in cases], "tags": [list(case.tags) for case in cases],
                "states": torch.stack([case.state for case in cases]),
                "controls": torch.stack([case.controls for case in cases]),
                "dt": torch.tensor([case.dt for case in cases]),
                "expected": torch.stack([case.expected for case in cases]),
                "views": views, "expected_views": expected_views}, directory / "corpus.pt")
    from PIL import Image
    for tag in ("zero_dt_identity", "door:4", "floor:59", "dead_freeze", "won"):
        index = next(i for i, case in enumerate(cases) if tag in case.tags)
        Image.fromarray(expected_views[index].numpy()).save(directory / f"source-{tag.replace(':', '-')}.png")


def semantic_failures(diagnostics):
    return [f"{row['name']}: {label} discrete semantics differ: {', '.join(metrics['semantic_mismatches'])}"
            for row in diagnostics for label, metrics in row["states"].items() if metrics["semantic_mismatches"]]


def exact_state_checks(case, states):
    returned = [value.detach().cpu() for value in states]
    return {tag: tag not in case.tags or all(torch.equal(value, case.expected if tag == "restart_exact" else case.state)
                                            for value in returned) for tag in EXACT_STATE_TAGS}


def assess_accuracy(diagnostics, budget=ACCURACY_BUDGET):
    if not diagnostics:
        raise ValueError("accuracy diagnostics must not be empty")
    if budget is None:
        raise ValueError("accuracy assessment requires a complete reviewed numeric budget")
    expected_fields = set(next(iter(diagnostics[0]["states"].values()))["fields"])
    if set(budget["state_max_abs"]) != expected_fields:
        raise ValueError("reviewed accuracy budget must cover every state field exactly once")
    frame_limits = {"mean_abs": budget["frame_mean_abs"],
                    "pixel_error_over_8_fraction": budget["frame_error_over_8_fraction"],
                    "pixel_error_over_32_fraction": budget["frame_error_over_32_fraction"]}
    limits = (*budget["state_max_abs"].values(), *frame_limits.values())
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in limits):
        raise ValueError("accuracy limits must be finite nonnegative numbers")
    failures = semantic_failures(diagnostics)
    for diagnostic in diagnostics:
        for label, measurement in diagnostic["states"].items():
            if set(measurement["fields"]) != expected_fields:
                raise ValueError("every state comparison must cover the same complete StateLayout")
            for name, metrics in measurement["fields"].items():
                metric = "circular_max_abs" if name in ANGULAR_FIELDS else "max_abs"
                if metrics[metric] > budget["state_max_abs"][name]:
                    failures.append(f"{diagnostic['name']}: {label} {name} {metric} {metrics[metric]} exceeds {budget['state_max_abs'][name]}")
        for label, measurement in diagnostic["frames"].items():
            for metric, limit in frame_limits.items():
                if measurement[metric] > limit:
                    failures.append(f"{diagnostic['name']}: {label} RGB {metric} {measurement[metric]} exceeds {limit}")
    return {"status": "failed" if failures else "passed", "budget": budget, "failures": failures}


def diagnose(runner, source, cases, layout, directory):
    diagnostics, failures = [], []
    worst = {kind: (-1, None) for kind in ("same_state_render", "forward_frame")}
    for case in cases:
        state, control = case.state.to(runner.device), case.controls.to(runner.device)
        dt = torch.tensor([case.dt], dtype=torch.float32, device=runner.device)
        with torch.inference_mode():
            actual, frame = runner.executor.forward(state, control, dt)
            split = runner.executor.transition(state, control, dt)
            split_frame = runner.executor.render(split)
            same_state_frame = runner.executor.render(state)
            reference_frame = source.render(case.state)
            reference_forward_frame = source.render(case.expected)
        for returned in (actual, split):
            check_bounds(returned, runner.state_lower, runner.state_upper, layout)
        parity = {"state": torch.equal(actual, split), "frame": torch.equal(frame, split_frame)}
        exact = exact_state_checks(case, (actual, split))
        row = {"name": case.name, "tags": list(case.tags), "dt": case.dt,
               "controls": case.controls.tolist(),
               "states": {"source_forward": compare_states(case.expected, actual, layout),
                          "source_transition": compare_states(case.expected, split, layout),
                          "full_split": compare_states(split, actual, layout)},
               "frames": {"same_state_render": compare_frames(reference_frame, same_state_frame),
                          "forward_frame": compare_frames(reference_forward_frame, frame),
                          "transition_render": compare_frames(reference_forward_frame, split_frame),
                          "full_split": compare_frames(split_frame, frame)},
               "full_split_exact": parity, **exact}
        diagnostics.append(row)
        failures.extend(semantic_failures([row]))
        if not all(exact.values()):
            failures.append(f"{case.name}: exact returned-state checks failed: {', '.join(tag for tag, passed in exact.items() if not passed)}")
        for kind, reference, rendered in (("same_state_render", reference_frame, same_state_frame),
                                           ("forward_frame", reference_forward_frame, frame)):
            if row["frames"][kind]["mean_abs"] > worst[kind][0]:
                worst[kind] = (row["frames"][kind]["mean_abs"], (case.name, reference, rendered.detach().cpu()))
    worst_cases = {}
    from PIL import Image
    for kind, (mean_abs, example) in worst.items():
        if example is None:
            continue
        name, reference, actual = example
        prefix = f"worst-{kind.replace('_', '-')}"
        Image.fromarray(reference.numpy()).save(directory / f"{prefix}-reference.png")
        Image.fromarray(actual.numpy()).save(directory / f"{prefix}-neural.png")
        difference = ((actual.to(torch.int16) - reference.to(torch.int16)).abs() * 4).clamp(0, 255).to(torch.uint8)
        Image.fromarray(difference.numpy()).save(directory / f"{prefix}-difference-x4.png")
        worst_cases[kind] = {"name": name, "mean_abs": mean_abs, "difference_display_gain": 4}
    return diagnostics, failures, worst_cases


def deployment_parity(runner, source, cases, layout):
    """Saved-JIT samples measure fusion differences and exact lifecycle rules."""
    selected = [cases[0],
                next(case for case in cases if any(tag.startswith("projectile_launch:") for tag in case.tags)),
                next(case for case in cases if "door:4" in case.tags)]
    selected.extend(next(case for case in cases if tag in case.tags) for tag in EXACT_STATE_TAGS)
    selected = list({case.name: case for case in selected}.values())
    rows = []
    for case in selected:
        state, control = case.state.to(runner.device), case.controls.to(runner.device)
        dt = torch.tensor([case.dt], dtype=torch.float32, device=runner.device)
        outputs = {}
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            for name, backend in (("saved", runner.model), ("deployed", runner.executor)):
                forward, frame = backend.forward(state, control, dt)
                transition = backend.transition(state, control, dt)
                rendered, split_frame = backend.render(state), backend.render(transition)
                for returned in (forward, transition):
                    check_bounds(returned, runner.state_lower, runner.state_upper, layout)
                outputs[name] = (forward, frame, transition, rendered, split_frame)
            reference_frame = source.render(case.state)
            reference_forward_frame = source.render(case.expected)
        saved, deployed = outputs["saved"], outputs["deployed"]
        row = {"name": case.name, "tags": list(case.tags),
               "states": {"source_saved_forward": compare_states(case.expected, saved[0], layout),
                          "source_saved_transition": compare_states(case.expected, saved[2], layout),
                          "saved_deployed_forward": compare_states(saved[0], deployed[0], layout),
                          "saved_deployed_transition": compare_states(saved[2], deployed[2], layout),
                          "saved_full_split": compare_states(saved[2], saved[0], layout),
                          "deployed_full_split": compare_states(deployed[2], deployed[0], layout)},
               "frames": {"source_saved_same_state_render": compare_frames(reference_frame, saved[3]),
                          "source_saved_forward": compare_frames(reference_forward_frame, saved[1]),
                          "source_saved_transition_render": compare_frames(reference_forward_frame, saved[4]),
                          "saved_deployed_same_state_render": compare_frames(saved[3], deployed[3]),
                          "saved_deployed_forward": compare_frames(saved[1], deployed[1]),
                          "saved_deployed_transition_render": compare_frames(saved[4], deployed[4]),
                          "saved_full_split": compare_frames(saved[4], saved[1]),
                          "deployed_full_split": compare_frames(deployed[4], deployed[1])},
               "saved_exact_checks": exact_state_checks(case, (saved[0], saved[2])),
               "deployed_exact_checks": exact_state_checks(case, (deployed[0], deployed[2])),
               "exact": {"saved_deployed_forward_state": torch.equal(saved[0], deployed[0]),
                         "saved_deployed_transition_state": torch.equal(saved[2], deployed[2]),
                         "saved_deployed_same_state_render": torch.equal(saved[3], deployed[3]),
                         "saved_deployed_forward_frame": torch.equal(saved[1], deployed[1]),
                         "saved_full_split_state": torch.equal(saved[0], saved[2]),
                         "saved_full_split_frame": torch.equal(saved[1], saved[4]),
                         "deployed_full_split_state": torch.equal(deployed[0], deployed[2]),
                         "deployed_full_split_frame": torch.equal(deployed[1], deployed[4])}}
        rows.append(row)
    return rows


class NeuralProbe:
    """Simulation repro adapter; only exported transitions produce its state."""

    def __init__(self, runner, layout):
        self.runner, self.layout = runner, layout
        self.steps = 0

    @property
    def state(self):
        return self.runner.state

    def step(self, control=NO_CONTROLS, dt=MAX_DT, tags=()):
        control, timestep = self.runner._inputs(control, dt)
        with torch.inference_mode():
            returned = self.runner.executor.transition(self.state, control, timestep)
        check_bounds(returned, self.runner.state_lower, self.runner.state_upper, self.layout)
        self.runner.state = returned
        self.steps += 1


class ObservedRoute:
    """Run the neural route; a separate source shadow only measures drift."""

    def __init__(self, runner, source, data):
        self.runner, self.source, self.layout = runner, source, S.StateLayout.from_data(data)
        self.field_slices = fields(self.layout)
        self.shadow = S.initial_state(data)
        self.maximum = torch.zeros(self.layout.size, dtype=torch.float64)
        self.total = torch.zeros_like(self.maximum)
        self.circular_maximum = {name: torch.zeros(self.field_slices[name].stop - self.field_slices[name].start,
                                                   dtype=torch.float64) for name in ANGULAR_FIELDS}
        self.circular_total = {name: torch.zeros_like(values) for name, values in self.circular_maximum.items()}
        self.ticks = 0

    @property
    def state(self):
        return self.runner.state

    def step(self, control, dt):
        try:
            frame = self.runner.step(control, dt)
            check_bounds(self.state, self.runner.state_lower, self.runner.state_upper, self.layout)
        except ValueError as error:
            raise RuntimeError(f"neural route output contract failed: {error}") from error
        with torch.inference_mode():
            self.shadow = self.source.transition(self.shadow, control.detach().cpu(), torch.tensor([dt], dtype=torch.float32))
        actual = self.state.detach().cpu()
        delta = (actual.double() - self.shadow.double()).abs()
        self.maximum = torch.maximum(self.maximum, delta)
        self.total += delta
        for name in ANGULAR_FIELDS:
            field = self.field_slices[name]
            circular = circular_error(self.shadow[field], actual[field])
            self.circular_maximum[name] = torch.maximum(self.circular_maximum[name], circular)
            self.circular_total[name] += circular
        self.ticks += 1
        return frame

    def drift(self):
        if not self.ticks:
            return {"ticks": 0, "status": "unmeasured", "reason": "No neural tick completed successfully."}
        measurements = {name: {"max_abs_over_play": float(self.maximum[field].max()),
                               "mean_abs_over_play": float(self.total[field].mean() / self.ticks)}
                        for name, field in self.field_slices.items()}
        for name in ANGULAR_FIELDS:
            measurements[name].update(circular_max_abs_over_play=float(self.circular_maximum[name].max()),
                                      circular_mean_abs_over_play=float(self.circular_total[name].mean() / self.ticks))
        return {"provenance": "Independent recurrent source and neural states receive identical neural-controller controls; no state repair",
                "ticks": self.ticks, "final": compare_states(self.shadow, self.state, self.layout),
                "fields": measurements}


def restart(runner, spawn, context):
    """Only an exported restart transition may restore the shared runner."""
    runner.step(controls(restart=1), dt=0)
    _require(torch.equal(runner.state.detach().cpu(), spawn),
             f"neural restart before {context} did not restore every original spawn activation exactly")


def live_choices(runner, spawn):
    """Repeat a shared live prefix, then make two divergent control choices."""
    observations, common = {}, None
    for branch, direction in (("left", -1), ("right", 1)):
        restart(runner, spawn, f"{branch} live choice")
        for _ in range(10):
            runner.step(controls(forward=1), MAX_DT)
        prefix = runner.state.detach().cpu().clone()
        if common is None:
            common = prefix
        else:
            _require(torch.equal(prefix, common), "identical live control prefix returned different neural states")
        for _ in range(5):
            frame = runner.step(controls(strafe=direction), MAX_DT)
        observations[branch] = (runner.state.detach().cpu().clone(), frame.detach().cpu().clone())
    left, left_frame = observations["left"]
    right, right_frame = observations["right"]
    separation = (left[S.X:S.Y + 1] - right[S.X:S.Y + 1]).abs()
    _require(float(separation[0]) > 20 and float(separation[1]) < 1,
             f"divergent live choices did not produce the expected lateral motion: {separation.tolist()}")
    _require(not torch.equal(left_frame, right_frame), "divergent live choices returned identical neural RGB images")
    return {"status": "passed", "identical_prefix_ticks": 10, "divergent_ticks_per_choice": 5,
            "position_separation": separation.tolist(), "frames_differ": True,
            "state_provenance": "One neural runner restarted through exported controls; all observations clone actual returned states."}


def validate_artifact(runner, source, cases, data, metadata, directory, max_steps=6000):
    layout = S.StateLayout.from_data(data)
    spawn = S.initial_state(data)
    _require(torch.equal(runner.state.detach().cpu(), spawn), "neural artifact must start at the exact original source spawn")
    print("Measuring source/deployed neural state and RGB accuracy…", flush=True)
    diagnostics, failures, worst = diagnose(runner, source, cases, layout, directory)
    print("Measuring saved/deployed neural parity and exact lifecycle states…", flush=True)
    deployment = deployment_parity(runner, source, cases, layout)
    failures.extend(semantic_failures(deployment))
    for row in deployment:
        for backend in ("saved", "deployed"):
            if not all(row[f"{backend}_exact_checks"].values()):
                failures.append(f"{row['name']}: {backend} exact returned-state checks failed")
    report = {"artifact_files": runner.manifest["files"], "execution": runner.info,
              "serialization_exact_checks": runner.manifest["export"]["exact_lowered_eager_and_full_split_checks"],
              "same_input_diagnostics": diagnostics, "deployment_diagnostics": deployment,
              "worst_render_cases": worst}

    def factory(name):
        restart(runner, spawn, name)
        return NeuralProbe(runner, layout)

    for name, reproduction in (("controls_only_regressions", regressions), ("controls_only_lifecycle", lifecycle)):
        print(f"Running neural {name.replace('_', ' ')}…", flush=True)
        try:
            report[name] = reproduction(factory, data)
        except (ValueError, RuntimeError) as error:
            failures.append(f"{name}: {error}")
            report[name] = {"status": "failed", "reason": str(error)}
    try:
        print("Running divergent neural live choices…", flush=True)
        report["divergent_live_choices"] = live_choices(runner, spawn)
    except (ValueError, RuntimeError) as error:
        failures.append(f"divergent live choices: {error}")
        report["divergent_live_choices"] = {"status": "failed", "reason": str(error)}
    route_reports, drift = {}, {}
    for branch in ("left", "right"):
        observed = ObservedRoute(runner, source, data)
        print(f"Running returned-state neural {branch} route…", flush=True)
        try:
            restart(runner, spawn, f"{branch} full route")
            route_reports[branch], frame = run_acceptance(observed, data, metadata, branch, max_steps)
            won = runner.state.clone()
            for _ in range(3):
                observed.step(controls(forward=1, strafe=1, turn=1, fire=1, use=1, weapon=0), MAX_DT)
                _require(torch.equal(runner.state, won), "won neural transition changed recurrent state")
            route_reports[branch]["won_freeze_ticks"] = 3
            route_reports[branch]["bounds_checked_ticks"] = observed.ticks
            from PIL import Image
            Image.fromarray(frame.detach().cpu().numpy()).save(directory / f"route-{branch}-final.png")
        except AcceptanceFailure as error:
            route_reports[branch] = error.report
            failures.append(f"returned-state {branch} route: {error}")
        except (ValueError, RuntimeError) as error:
            route_reports[branch] = {"status": "failed", "reason": str(error)}
            failures.append(f"returned-state {branch} route: {error}")
        finally:
            drift[branch] = observed.drift()
    report["returned_state_routes"], report["recurrent_same_controls_drift"] = route_reports, drift
    report["accuracy"] = assess_accuracy(diagnostics + deployment)
    failures.extend(report["accuracy"].get("failures", ()))
    report["failures"] = failures
    report["status"] = "failed" if failures else "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="v2 strict neural artifact; omit for source diagnostics only")
    parser.add_argument("--wad", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/neural-validation"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6000)
    args = parser.parse_args()
    if args.threads < 1 or args.max_steps < 1:
        parser.error("--threads and --max-steps must be positive")
    from doomnet.assets import build_assets
    from doomnet.runtime import RuntimeModel

    torch.set_num_threads(args.threads)
    print("Loading source WAD construction data…", flush=True)
    data, metadata = build_assets(args.wad)
    if args.model:
        print(f"Loading and auditing neural artifact {args.model}…", flush=True)
    runner = RuntimeModel(args.model, device=args.device, threads=args.threads) if args.model else None
    if runner is not None:
        artifact_source = runner.manifest["construction"]["source"]
        if (artifact_source["wad"]["md5"], artifact_source["skill"], artifact_source["level"]) != (metadata["wad"]["md5"], metadata["skill"], metadata["level"]):
            raise ValueError("test WAD/skill/level must match the neural artifact")
    width, height = (runner.width, runner.height) if runner else (64, 48)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"Capturing controls-only source corpus at {width}x{height}…", flush=True)
    source, cases, baseline = build_corpus(data, metadata, width, height, args.max_steps)
    print(f"Rendering and saving {len(cases)} source diagnostic samples…", flush=True)
    save_corpus(cases, source, args.output, baseline["construction"])
    report = {"test_kind": "strict_neural_acceptance", "source": metadata, "source_diagnostics": baseline,
              "corpus_samples": len(cases), "corpus_sha256": hashlib.sha256((args.output / "corpus.pt").read_bytes()).hexdigest(),
              "state_provenance": "Same-input source corpus is diagnostic only; actual neural routes carry only RuntimeModel returned states."}
    if args.model:
        _require(baseline["construction_consistent_during_capture"], "source construction changed during corpus capture; regenerate before neural accuracy validation")
        try:
            report.update(validate_artifact(runner, source, cases, data, metadata, args.output, args.max_steps))
        finally:
            runner.close()
    else:
        report.update(status="source_diagnostic", reason="No strict neural artifact was tested; source execution is not completion.")
    report["wall_seconds"] = time.perf_counter() - started
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(encoded)
    print(json.dumps({name: report[name] for name in ("status", "corpus_samples", "wall_seconds")}, indent=2))
    return int(report["status"] != "passed") if args.model else 0


if __name__ == "__main__":
    raise SystemExit(main())
