"""Controls-only optional E1M1 features on one strict neural artifact.

Run ``python -m scripts.validate_features --model build/full-neural-e1m1
--wad assets/doom1.wad --device cuda --threads 1``. Six exported-restart
branches carry only neural returned states. Transition runs every tick; render
runs at named feature events and restart boundaries. Source evaluations use
cloned diagnostic inputs and never supply recurrent gameplay state.
"""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from doomnet import state as S
from doomnet.runtime import MAX_DT, NO_CONTROLS, sha256_file
from scripts.validate_level import E1M1Controller, angle_difference
from scripts.validate_neural import (
    ACCURACY_BUDGET, Case, NeuralProbe, _require, assess_accuracy, compare_frames,
    compare_states, construction_provenance, controls, diagnose, restart,
)


GREEN_STAIRS = (57, 64, 78, 90, 89, 88, 87, 86, 91, 116, 352)
BLUE_PASSAGE = (224, 225, 226, 227, 228, 229, 230, 223, 222, 221, 220,
                219, 218, 217, 216, 251, 164)
SHOTGUN_PATH = ((3008, -3600), (3184, -3608), (3184, -3560), (3264, -3520),
                (3352, -3530), (3400, -3520), (3400, -3568), (3440, -3640),
                (3440, -3850), (3296, -3850))
LIFT_PATH = ((3008, -3440), (3352, -3530), (3400, -3568),
             (3488, -3868), (3552, -3868))


def file_reference(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class FeatureProbe(NeuralProbe):
    def __init__(self, runner, source, data, directory, branch, inputs):
        super().__init__(runner, S.StateLayout.from_data(data))
        self.source, self.data, self.directory = source, data, directory
        self.branch, self.inputs = branch, inputs
        self.observer = E1M1Controller(data)
        self.events, self.samples = [], []
        self.last = None
        self.lift_travel, self.minimum_lift = False, math.inf
        self.ledge_crossing = False

    def cpu(self):
        return self.state.detach().cpu()

    def observation(self, state, entity=None):
        row = {"position": state[[S.X, S.Y, S.Z]].tolist(),
               "sector": self.observer.sector_at(state[:2].numpy()),
               **{name.lower(): float(state[getattr(S, name)]) for name in
                  ("HEALTH", "ARMOR", "ARMOR_TYPE", "WEAPON", "SHOTGUN_OWNED",
                   "BULLETS", "SHELLS", "KILLS", "SECRETS", "TIME")},
               "secret_flags": state[self.layout.sector_secret][[68, 69, 70]].tolist(),
               "floor_59": float(state[self.layout.sector_floor][59]),
               "floor_70": float(state[self.layout.sector_floor][70]),
               "floor_motion_70": float(state[self.layout.sector_floor_motion][70]),
               "line_308_used": float(state[self.layout.line_used][308])}
        if entity is not None:
            row["entity"] = {"index": entity, "type": int(self.data["entity_type"][entity]),
                             "hp": float(state[self.layout.entity_hp][entity]),
                             "alive": float(state[self.layout.entity_alive][entity])}
        return row

    def mark(self, name, entity=None):
        _require(self.last is not None, "feature sample requires an actual neural transition")
        full_name = f"{self.branch}:{name}"
        _require(not any(row["name"] == full_name for row in self.events),
                 f"duplicate feature event {full_name}")
        before, control, dt, returned = self.last
        with torch.inference_mode():
            expected = self.source.transition(before, control, torch.tensor([dt]))
            frame = self.runner.executor.render(self.state)
        self.runner._check_frame(frame)
        frame = frame.detach().cpu().clone()
        image_path = self.directory / f"{full_name.replace(':', '-')}.png"
        from PIL import Image
        Image.fromarray(frame.numpy()).save(image_path)
        self.samples.append((Case(full_name, ("feature:" + name,), before, control, dt,
                                  expected.detach().cpu().clone()), returned, frame))
        row = {"name": full_name, "tick": self.steps, "dt": dt,
               "before": self.observation(before, entity),
               "after": self.observation(returned, entity), "image": file_reference(image_path)}
        self.events.append(row)
        print(f"Reached {full_name} at tick {self.steps}", flush=True)
        return row

    def step(self, control=NO_CONTROLS, dt=MAX_DT, tags=()):
        before = self.cpu().clone()
        action = torch.as_tensor(control, dtype=torch.float32).detach().cpu().clone()
        super().step(action, dt)
        returned = self.cpu().clone()
        self.last = before, action, dt, returned
        self.inputs.write(json.dumps({"branch": self.branch, "tick": self.steps,
                                      "controls": action.tolist(), "dt": dt}) + "\n")
        _require(self.steps <= 3500, f"{self.branch} exceeded 3500 neural ticks")
        _require(returned[S.HEALTH] > 0, f"{self.branch} player died at {returned[:2].tolist()}")
        _require(returned[S.WON] == 0, "optional branch must terminate before the exit")
        for entity, name in ((55, "green_armor"), (7, "blue_armor"), (65, "shotgun_pickup")):
            if returned[self.layout.entity_alive][entity] < before[self.layout.entity_alive][entity]:
                self.mark(name, entity)
        for sector in (68, 69, 70):
            if returned[self.layout.sector_secret][sector] > before[self.layout.sector_secret][sector]:
                self.mark(f"secret_{sector}")
        if before[self.layout.sector_floor][59] > -40 and returned[self.layout.sector_floor][59] == -40:
            self.mark("floor_59_lowered")
        if self.lift_travel:
            self.minimum_lift = min(self.minimum_lift, float(returned[self.layout.sector_floor][70]))
            if before[self.layout.sector_floor_motion][70] == 0 and returned[self.layout.sector_floor_motion][70] < 0:
                self.mark("lift_trigger")
            if before[self.layout.sector_floor][70] > -48 and returned[self.layout.sector_floor][70] == -48:
                self.mark("lift_bottom")
        if self.ledge_crossing and before[S.Z] > -8 and returned[S.Z] == -8:
            self.mark("ledge_landed")
        return before

    def goto(self, point, radius=5, fast=False):
        target = torch.tensor(point, dtype=torch.float32)
        for _ in range(500):
            current = self.cpu()
            delta = target - current[:2]
            distance = float(delta.norm())
            if distance <= radius:
                return
            if fast:
                angle = float(current[S.ANGLE])
                facing = torch.tensor((math.cos(angle), math.sin(angle)))
                right = torch.tensor((facing[1], -facing[0]))
                magnitude = min(1, distance / (300 * MAX_DT))
                clip = lambda value: max(-1, min(1, float(value) * magnitude))
                action = controls(forward=clip(delta @ facing / distance),
                                  strafe=clip(delta @ right / distance), run=1)
            else:
                error = angle_difference(math.atan2(float(delta[1]), float(delta[0])), float(current[S.ANGLE]))
                run = distance > 80
                speed = 300 if run else 180
                action = controls(turn=max(-1, min(1, error / (2.8 * MAX_DT))),
                                  forward=min(1, distance / (speed * MAX_DT)) if abs(error) < .12 else 0,
                                  run=int(run))
            self.step(action)
        raise RuntimeError(f"blocked waypoint {point}, actual {self.cpu()[:2].tolist()}")

    def prefix(self, controller, name):
        for _ in range(3500):
            if name in controller.reached:
                return
            self.step(controller.controls(self.state))
        raise RuntimeError(f"prefix failed at {controller.waypoint}")

    def door(self, point, angle, sector):
        self.goto(point)
        for _ in range(90):
            error = angle_difference(angle, float(self.cpu()[S.ANGLE]))
            if abs(error) < .02:
                break
            self.step(controls(turn=max(-1, min(1, error / (2.8 * MAX_DT)))))
        else:
            raise RuntimeError(f"door {sector} aiming did not converge")
        self.step(controls(use=1))
        for _ in range(120):
            current = self.cpu()
            if current[self.layout.sector_ceiling][sector] - current[self.layout.sector_floor][sector] >= 60:
                return
            self.step()
        raise RuntimeError(f"door {sector} did not open")

    def portals(self, lines):
        for line in lines:
            current = self.observer.sector_at(self.cpu()[:2].numpy())
            front, back = (int(self.data[name][line]) for name in ("line_front_sector", "line_back_sector"))
            _require(current in (front, back) and back >= 0, f"portal {line} does not border sector {current}")
            destination = back if current == front else front
            ends = self.data["vertices"][self.data["lines"][line]]
            middle, delta = ends.mean(0), ends[1] - ends[0]
            normal = torch.tensor((-float(delta[1]), float(delta[0]))) / delta.norm()
            targets = [middle + normal * 8, middle - normal * 8]
            target = next((p for p in targets if self.observer.sector_at(p.numpy()) == destination), None)
            _require(target is not None, f"portal {line} has no destination approach")
            self.goto(target, radius=6)
            _require(self.observer.sector_at(self.cpu()[:2].numpy()) == destination,
                     f"portal {line} did not cross into sector {destination}")

    def shoot(self, entity, weapon):
        for _ in range(180):
            current = self.cpu()
            delta = torch.stack((current[self.layout.entity_x][entity], current[self.layout.entity_y][entity])) - current[:2]
            error = angle_difference(math.atan2(float(delta[1]), float(delta[0])), float(current[S.ANGLE]))
            before = self.step(controls(turn=max(-1, min(1, error / (2.8 * MAX_DT))),
                                        fire=int(abs(error) < .02), weapon=weapon))
            if self.cpu()[self.layout.entity_alive][entity] < .5:
                return before
        raise RuntimeError(f"weapon {weapon} did not destroy entity {entity}")


def fist(probe):
    controller = E1M1Controller(probe.data, "right")
    probe.prefix(controller, "first_door")
    for _ in range(650):
        current = probe.cpu()
        combat = controller.combat_target(current.numpy())
        if combat is None:
            probe.goto((1720, -2496))
            continue
        distance, entity, target = combat
        delta = target - current[:2].numpy()
        error = angle_difference(math.atan2(float(delta[1]), float(delta[0])), float(current[S.ANGLE]))
        before = probe.step(controls(turn=max(-1, min(1, error / (2.8 * MAX_DT))),
                                    forward=min(1, max(0, (distance - 40) / (180 * MAX_DT))) if abs(error) < .12 else 0,
                                    fire=int(abs(error) < .02 and distance <= 60), weapon=0))
        if probe.cpu()[probe.layout.entity_hp][entity] < before[probe.layout.entity_hp][entity]:
            _require(probe.cpu()[S.WEAPON] == 0 and probe.cpu()[S.BULLETS] == before[S.BULLETS],
                     "fist hit must use no ammunition")
            probe.mark("fist_hit", entity)
            return
    raise RuntimeError("fist pursuit did not land a controls-only hit")


def green_pickup(probe):
    probe.prefix(E1M1Controller(probe.data, "left"), "west_health_bonus")
    probe.portals(GREEN_STAIRS)
    probe.goto(probe.data["entity_pos"][55])
    _require(probe.cpu()[S.ARMOR] == 100 and probe.cpu()[S.ARMOR_TYPE] == 1, "green armor not acquired")


def green_blast(probe):
    green_pickup(probe)
    probe.portals(reversed(GREEN_STAIRS))
    for point in ((768, -3472), (1280, -3472), (1286, -3340)):
        probe.goto(point)
    before = probe.shoot(24, 1)
    _require(before[S.HEALTH] > probe.cpu()[S.HEALTH] and before[S.ARMOR] > probe.cpu()[S.ARMOR],
             "barrel explosion did not damage the nearby armored player")
    probe.mark("barrel_blast", 24)


def ledge(probe):
    """Walk off the green stairs' line95 corner, then continue across the floor."""
    green_pickup(probe)
    probe.portals((352, 116))
    probe.goto((116, -3268))
    probe.goto((134, -3274), radius=3, fast=True)
    current = probe.cpu()
    _require(probe.observer.sector_at(current[:2].numpy()) == 24 and current[S.Z] == 104,
             "line95 crossing did not retain support over sector24")
    probe.mark("ledge_supported")
    probe.ledge_crossing = True
    probe.goto((184, -3288), radius=6, fast=True)
    _require(probe.cpu()[S.Z] == -8, "player did not land after clearing the stair radius")
    _require(any(row["name"] == "ledge:ledge_landed" for row in probe.events),
             "actual landing transition was not captured")
    landed = probe.cpu()[:2].clone()
    probe.goto((240, -3308), radius=6, fast=True)
    _require(float((probe.cpu()[:2] - landed).norm()) > 50, "movement remained blocked after landing")
    probe.mark("ledge_movement_continues")


def blue_hazard(probe):
    probe.prefix(E1M1Controller(probe.data, "right"), "shell_pickup")
    probe.door((2980, -3840), math.pi, 68)
    probe.goto((2928, -3840))
    _require(probe.cpu()[probe.layout.sector_secret][68] == 1, "secret 68 not discovered")
    probe.goto((2890, -3840))
    probe.portals(BLUE_PASSAGE)
    probe.goto(probe.data["entity_pos"][7])
    _require(probe.cpu()[S.ARMOR] == 200 and probe.cpu()[S.ARMOR_TYPE] == 2, "blue armor not acquired")
    for _ in range(33):
        before = probe.step()
        current = probe.cpu()
        if before[S.ARMOR] > current[S.ARMOR] and before[S.HEALTH] > current[S.HEALTH]:
            _require(probe.observer.sector_at(current[:2].numpy()) == 13 and probe.data["sector_special"][13] == 7,
                     "hazard witness must occur on actual nukage")
            _require(before[S.HEALTH] - current[S.HEALTH] == 3 and before[S.ARMOR] - current[S.ARMOR] == 2,
                     "blue armor must absorb two of the five hazard damage points")
            probe.mark("nukage_absorption")
            return
    raise RuntimeError("nukage did not damage the armored player within 33 ticks")


def lowered_prefix(probe):
    probe.prefix(E1M1Controller(probe.data, "right"), "medkit")
    probe.goto((3008, -4435))
    probe.door((3008, -4080), math.pi / 2, 76)
    probe.goto((3008, -3840))
    for _ in range(40):
        probe.step()
    current = probe.cpu()
    _require(current[probe.layout.sector_floor][59] == -40 and current[probe.layout.line_used][308] == 1,
             "real final-room crossing must lower floor 59 to -40")


def shotgun(probe):
    lowered_prefix(probe)
    for point in SHOTGUN_PATH:
        probe.goto(point, radius=6, fast=True)
    probe.goto(probe.data["entity_pos"][65])
    _require(probe.cpu()[S.SHOTGUN_OWNED] == 1 and probe.cpu()[probe.layout.sector_secret][69] == 1,
             "shotgun and secret 69 not acquired")
    for point in reversed(SHOTGUN_PATH):
        probe.goto(point, radius=6, fast=True)
    probe.goto((3008, -3840))
    probe.door((3008, -3975), -math.pi / 2, 76)
    probe.goto((3008, -4100))
    before = probe.shoot(37, 2)
    _require(before[S.SHELLS] > probe.cpu()[S.SHELLS] and probe.cpu()[S.WEAPON] == 2,
             "shotgun must spend shells and damage its target")
    probe.mark("shotgun_hit", 37)


def lift(probe):
    lowered_prefix(probe)
    for point in ((3008, -3450), (3008, -3008), (2916, -2928)):
        probe.goto(point)
    for _ in range(220):
        if probe.cpu()[probe.layout.sector_floor_motion][70] == 0:
            break
        probe.step()
    current = probe.cpu()
    _require(current[probe.layout.sector_floor_motion][70] == 0 and current[probe.layout.sector_floor][70] == 104,
             "lift must be idle at its original height before the real crossing")
    probe.lift_travel = True
    probe.goto((2916, -3000), radius=6, fast=True)
    for point in LIFT_PATH:
        probe.goto(point, radius=6, fast=True)
    current = probe.cpu().clone()
    _require(probe.minimum_lift == -48 and current[probe.layout.sector_secret][70] == 1
             and probe.observer.sector_at(current[:2].numpy()) == 70,
             "secret 70 requires entry through the genuinely lowered lift")
    for _ in range(90):
        probe.step()
        if probe.cpu()[probe.layout.sector_floor][70] == 104 and probe.cpu()[probe.layout.sector_floor_motion][70] == 0:
            break
    else:
        raise RuntimeError("entered lift did not finish rising")
    _require(probe.cpu()[S.Z] == 104 and probe.cpu()[S.HEALTH] == current[S.HEALTH]
             and probe.observer.sector_at(probe.cpu()[:2].numpy()) == 70,
             "player did not ride the lift to its original height without damage")
    probe.mark("lift_riding")


def measure_events(runner, source, samples, layout, directory):
    diagnostics, failures, worst = diagnose(runner, source, [sample[0] for sample in samples], layout, directory)
    for row, (case, returned, frame) in zip(diagnostics, samples, strict=True):
        with torch.inference_mode():
            same_state, expected_frame = source.render(returned), source.render(case.expected)
        row["states"]["recurrent_transition"] = compare_states(case.expected, returned, layout)
        row["frames"]["returned_state_render"] = compare_frames(same_state, frame)
        row["frames"]["recurrent_transition_render"] = compare_frames(expected_frame, frame)
    accuracy = assess_accuracy(diagnostics)
    return diagnostics, failures + accuracy["failures"], worst, accuracy


def validate_features(runner, source, data, directory):
    spawn = S.initial_state(data)
    _require(torch.equal(runner.state.detach().cpu(), spawn), "feature validation requires original spawn")
    construction = construction_provenance(source)
    failures, branches, samples = [], {}, []
    with (directory / "controls.jsonl").open("w") as inputs:
        for name, route in (("fist", fist), ("green_blast", green_blast),
                            ("blue_hazard", blue_hazard), ("shotgun", shotgun), ("lift", lift),
                            ("ledge", ledge)):
            print(f"Running returned-state neural {name} branch…", flush=True)
            probe = FeatureProbe(runner, source, data, directory, name, inputs)
            try:
                restart(runner, spawn, name)
                inputs.write(json.dumps({"branch": name, "tick": 0,
                                          "controls": controls(restart=1).tolist(), "dt": 0}) + "\n")
                route(probe)
                branches[name] = {"status": "passed", "restart_exact": True}
            except (ValueError, RuntimeError) as error:
                branches[name] = {"status": "failed", "reason": str(error)}
                failures.append(f"{name}: {error}")
            branches[name].update(ticks=probe.steps, events=probe.events,
                                  final=probe.observation(probe.cpu()),
                                  minimum_lift_floor=probe.minimum_lift if probe.lift_travel else None)
            samples.extend(probe.samples)
    _require(bool(samples), "no optional neural feature events were reached")
    print(f"Measuring {len(samples)} named neural feature events with the fixed acceptance budgets…", flush=True)
    layout = S.StateLayout.from_data(data)
    diagnostics, numeric_failures, worst, accuracy = measure_events(runner, source, samples, layout, directory)
    failures.extend(numeric_failures)
    consistent = construction == construction_provenance(source)
    if not consistent:
        failures.append("source construction changed during feature validation")
    corpus = directory / "events.pt"
    torch.save({"provenance": "Actual neural returned-state event samples; cloned source evaluations are diagnostic only",
                "construction": construction, "names": [sample[0].name for sample in samples],
                "states": torch.stack([sample[0].state for sample in samples]),
                "controls": torch.stack([sample[0].controls for sample in samples]),
                "dt": torch.tensor([sample[0].dt for sample in samples]),
                "expected": torch.stack([sample[0].expected for sample in samples]),
                "returned": torch.stack([sample[1] for sample in samples]),
                "frames": torch.stack([sample[2] for sample in samples])}, corpus)
    return {"status": "failed" if failures else "passed", "failures": failures,
            "branches": branches, "event_samples": file_reference(corpus),
            "controls": file_reference(directory / "controls.jsonl"),
            "construction": construction, "construction_consistent_during_run": consistent,
            "same_input_diagnostics": diagnostics, "accuracy": accuracy,
            "worst_render_cases": worst, "artifact_files": runner.manifest["files"],
            "execution": runner.info,
            "state_provenance": "One RuntimeModel; original spawn and exact exported restart per branch; only neural transitions carry recurrent state. Source inputs/results never replace it.",
            "render_cadence": "Neural transition and bounds check every gameplay tick; neural RGB at named events and full-method restart boundaries; separate full/split event diagnostics.",
            "scope": "Six authored controls-only scenarios exercise these existing E1M1 features and stair ledge support; this is not proof for every possible input or vanilla bit parity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wad", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/neural-feature-validation"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    from doomnet.assets import build_assets
    from doomnet.model import ConstructedDoom
    from doomnet.runtime import RuntimeModel

    print("Loading source WAD data and auditing strict neural artifact…", flush=True)
    data, metadata = build_assets(args.wad)
    runner = RuntimeModel(args.model, args.device, args.threads)
    source_metadata = runner.manifest["construction"]["source"]
    _require(tuple(source_metadata[key] for key in ("level", "skill")) == tuple(metadata[key] for key in ("level", "skill"))
             and source_metadata["wad"]["md5"] == metadata["wad"]["md5"], "feature WAD/level/skill must match the artifact")
    source = ConstructedDoom(data, width=runner.width, height=runner.height).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        report = validate_features(runner, source, data, args.output)
    finally:
        runner.close()
    report.update(test_kind="strict_neural_optional_features", source=metadata,
                  wall_seconds=time.perf_counter() - started,
                  manifest=file_reference(args.model / "manifest.json"),
                  harness=file_reference(Path(__file__).resolve()))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "failures", "wall_seconds")}), flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
