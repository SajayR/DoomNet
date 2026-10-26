"""Automated E1M1 gameplay test, separate from the interactive runtime.

Run ``python -m scripts.validate_level --model build/e1m1 --wad assets/doom1.wad``.
The authored waypoints and feedback policy are test inputs. Every game state and
rendered frame comes from the network; this controller never edits game state.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from doomnet.runtime import CONTROL_NAMES, MAX_DT
from doomnet.state import (
    ANGLE, BULLETS, HEALTH, KILLS, SHOTGUN_OWNED, SHELLS, TIME, WON, X, Y, Z,
    StateLayout, initial_state,
)


DT = MAX_DT


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    radius: float = 16
    use_angle: float | None = None
    door_sector: int | None = None


STARTS = {
    "left": (
        Waypoint("west_health_bonus", 736, -3520, 12),
        Waypoint("west_of_start_pillar", 768, -2992),
        Waypoint("north_of_start_pillar", 1024, -2970),
        Waypoint("corridor_approach", 1280, -2970),
    ),
    "right": (
        Waypoint("east_health_bonus", 1312, -3520, 12),
        Waypoint("east_of_start_pillar", 1280, -3472),
        Waypoint("corridor_approach", 1280, -2970),
    ),
}
FINISH = (
    Waypoint("north_corridor", 1280, -2850),
    Waypoint("corridor_bend", 1312, -2580),
    Waypoint("first_door", 1495, -2496, 8, 0, 4),
    Waypoint("first_door_passed", 1720, -2496),
    Waypoint("stair_room", 2050, -2496),
    Waypoint("south_of_stairs", 2050, -2650),
    Waypoint("stair_room_exit", 2450, -2630),
    Waypoint("east_corridor", 2600, -2630),
    Waypoint("east_corridor_bend", 2740, -2700),
    Waypoint("lift_approach", 2870, -2780),
    Waypoint("lift_crossing", 2940, -2985),
    Waypoint("brown_room", 3008, -3450),
    Waypoint("exit_corridor", 3008, -3660),
    Waypoint("shell_pickup", 3008, -3968, 12),
    Waypoint("penultimate_door", 3008, -3975, 8, -math.pi / 2, 76),
    Waypoint("penultimate_door_passed", 3008, -4100),
    Waypoint("final_room", 3008, -4435),
    Waypoint("medkit", 2880, -4435, 12),
    Waypoint("final_room_exit", 3008, -4510),
    Waypoint("exit_door", 3008, -4590, 6, -math.pi / 2, 81),
    Waypoint("exit_door_passed", 3008, -4720, 12),
    Waypoint("exit_switch", 2955, -4768, 6, math.pi),
)


class AcceptanceFailure(RuntimeError):
    def __init__(self, reason, report):
        super().__init__(reason)
        self.report = dict(report, status="failed", reason=reason)


def angle_difference(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


class E1M1Controller:
    """Test-only route follower with live aiming and independent LOS checks."""

    def __init__(self, data, branch="right"):
        if branch not in STARTS:
            raise ValueError(f"unknown E1M1 test branch: {branch}")
        self.data = {name: value.detach().cpu().numpy() for name, value in data.items()}
        self.layout = StateLayout.from_data(data)
        self.route = STARTS[branch] + FINISH
        self.branch = branch
        self.index = 0
        self.ticks = 0
        self.waypoint_ticks = 0
        self.use_started = None
        self.reached = []
        self.engaged = set()
        vertices = self.data["vertices"]
        lines = self.data["lines"]
        self.line_start = vertices[lines[:, 0]]
        self.line_delta = vertices[lines[:, 1]] - self.line_start
        self.front = self.data["line_front_sector"]
        self.back = self.data["line_back_sector"]
        self.enemies = np.flatnonzero(np.isin(self.data["entity_type"], (3001, 3004, 9)))

    @property
    def waypoint(self):
        return self.route[self.index] if self.index < len(self.route) else None

    def sector_at(self, point):
        node = len(self.data["node_children"]) - 1
        while node >= 0:
            wx, wy = self.data["node_weight"][node]
            plane = np.dot(self.data["node_weight"][node], point) + self.data["node_bias"][node]
            axis_front = (wx == 0 and wy > 0) or (wy == 0 and wx > 0)
            northwest_origin = (wx < 0 and wy < 0
                                and np.array_equal(point, self.data["node_origin"][node]))
            branch = int(plane > 0 or (plane == 0 and not (axis_front or northwest_origin)))
            node = int(self.data["node_children"][node, branch])
        return int(self.data["subsector_sector"][-1 - node])

    def visible(self, state, target):
        """Independent segment math used only to select test controls."""
        origin = state[X:Y + 1]
        direction = target - origin
        relative = self.line_start - origin
        denominator = (direction[0] * self.line_delta[:, 1]
                       - direction[1] * self.line_delta[:, 0])
        nonparallel = np.abs(denominator) > 1e-7
        divisor = np.where(nonparallel, denominator, 1)
        distance = (relative[:, 0] * self.line_delta[:, 1]
                    - relative[:, 1] * self.line_delta[:, 0]) / divisor
        segment = (relative[:, 0] * direction[1]
                   - relative[:, 1] * direction[0]) / divisor
        intersects = (nonparallel & (distance > 1e-4) & (distance < .9999)
                      & (segment >= 0) & (segment <= 1))
        floor = state[self.layout.sector_floor]
        ceiling = state[self.layout.sector_ceiling]
        back = np.maximum(self.back, 0)
        bottom = np.maximum(floor[self.front], floor[back])
        top = np.minimum(ceiling[self.front], ceiling[back])
        target_height = floor[self.sector_at(target)] + 32
        height = state[Z] + 41 + distance * (target_height - state[Z] - 41)
        blocked = (self.back < 0) | (height <= bottom) | (height >= top)
        return not np.any(intersects & blocked)

    def combat_target(self, state):
        xs, ys = state[self.layout.entity_x], state[self.layout.entity_y]
        alive = state[self.layout.entity_alive]
        candidates = []
        for index in self.enemies:
            target = np.array((xs[index], ys[index]))
            distance = float(np.linalg.norm(target - state[X:Y + 1]))
            if alive[index] > .5 and distance < 850 and self.visible(state, target):
                candidates.append((distance, int(index), target))
        return min(candidates, key=lambda item: item[0]) if candidates else None

    def controls(self, tensor):
        state = tensor.detach().cpu().numpy()
        self.ticks += 1
        self.waypoint_ticks += 1
        controls = np.zeros(8, dtype=np.float32)
        controls[6] = -1
        waypoint = self.waypoint
        if waypoint is None:
            raise RuntimeError("test route ended before the network signaled completion")
        if self.waypoint_ticks > 1400:
            raise RuntimeError(f"lost at {waypoint.name}: x={state[X]:.2f}, y={state[Y]:.2f}")

        # Deliberately different live inputs before the two feedback routes.
        if self.ticks <= 10:
            controls[0] = 1
            return torch.from_numpy(controls)
        if self.ticks <= 15:
            controls[1] = -1 if self.branch == "left" else 1
            return torch.from_numpy(controls)

        combat = self.combat_target(state)
        if combat is not None and self.use_started is None:
            distance, index, target = combat
            self.engaged.add(index)
            error = angle_difference(math.atan2(target[1] - state[Y],
                                                target[0] - state[X]), state[ANGLE])
            controls[2] = np.clip(error / (2.8 * DT), -1, 1)
            controls[3] = float(abs(error) < .08)
            if state[SHOTGUN_OWNED] > .5 and state[SHELLS] > 0:
                controls[6] = 2
            elif state[BULLETS] > 0:
                controls[6] = 1
            else:
                controls[6] = 0
                controls[0] = float(distance > 45 and abs(error) < .08)
            return torch.from_numpy(controls)

        distance = math.hypot(waypoint.x - state[X], waypoint.y - state[Y])
        if distance <= waypoint.radius:
            if waypoint.use_angle is not None:
                error = angle_difference(waypoint.use_angle, state[ANGLE])
                controls[2] = np.clip(error / (2.8 * DT), -1, 1)
                if abs(error) >= .03:
                    return torch.from_numpy(controls)
                if self.use_started is None:
                    self.use_started = self.ticks
                    controls[4] = 1
                    return torch.from_numpy(controls)
                if waypoint.door_sector is not None:
                    sector = waypoint.door_sector
                    gap = state[self.layout.sector_ceiling][sector] - state[self.layout.sector_floor][sector]
                    if gap < 60:
                        if self.ticks - self.use_started > 105:
                            raise RuntimeError(f"door {sector} did not open after use at {waypoint.name}")
                        return torch.from_numpy(controls)
            self.reached.append(waypoint.name)
            self.index += 1
            self.waypoint_ticks = 0
            self.use_started = None
            return torch.from_numpy(controls)

        desired = math.atan2(waypoint.y - state[Y], waypoint.x - state[X])
        error = angle_difference(desired, state[ANGLE])
        controls[2] = np.clip(error / (2.8 * DT), -1, 1)
        controls[0] = min(distance / (180 * DT), 1) if abs(error) < .15 else 0
        controls[7] = float(distance > 250)
        return torch.from_numpy(controls)


def run_acceptance(session, data, metadata, branch="right", max_steps=6000,
                   require_render=True):
    """Run from exact original spawn using only session.step controls.

    ``require_render=False`` is explicitly a simulation diagnostic, not an
    exported full-network acceptance result. It is useful while fixing rules.
    """
    controller = E1M1Controller(data, branch)
    layout = controller.layout
    initial = session.state.detach().cpu().clone()
    if not torch.equal(initial, initial_state(data)):
        raise ValueError("acceptance requires exactly the original fresh E1M1 spawn state")
    report = {"test_kind": "automated_feedback_controller", "level": metadata["level"],
              "skill": metadata["skill"], "wad_md5": metadata["wad"]["md5"],
              "branch": branch, "rendering": require_render,
              "spawn": initial[X:ANGLE + 1].tolist()}
    control_activity = {name: {"ticks": 0, "minimum": None, "maximum": None}
                        for name in CONTROL_NAMES}
    opened = set()
    floor_changes = set()
    frame_digests = set()
    minimum_health = float(initial[HEALTH])
    damage = 0.0
    previous_health = minimum_health
    frame = None
    started = time.perf_counter()
    failure = None
    ticks = 0
    try:
        for ticks in range(1, max_steps + 1):
            controls = controller.controls(session.state)
            for index, name in enumerate(CONTROL_NAMES):
                value = float(controls[index])
                activity = control_activity[name]
                activity["ticks"] += int(value != (-1 if index == 6 else 0))
                activity["minimum"] = value if activity["minimum"] is None else min(activity["minimum"], value)
                activity["maximum"] = value if activity["maximum"] is None else max(activity["maximum"], value)
            frame = session.step(controls, DT)
            state = session.state.detach().cpu()
            if not bool(torch.isfinite(state).all()):
                raise RuntimeError("network returned nonfinite game state")
            if require_render:
                if not isinstance(frame, torch.Tensor) or frame.dtype != torch.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise RuntimeError("network must return a uint8 RGB frame")
                frame_digests.add(hashlib.sha256(frame.detach().cpu().numpy().tobytes()).hexdigest())
            health = float(state[HEALTH])
            minimum_health = min(minimum_health, health)
            damage += max(previous_health - health, 0)
            previous_health = health
            opened.update(torch.nonzero(state[layout.sector_ceiling] > initial[layout.sector_ceiling] + 1).flatten().tolist())
            floor_changes.update(torch.nonzero((state[layout.sector_floor] - initial[layout.sector_floor]).abs() > 1).flatten().tolist())
            if health <= 0:
                raise RuntimeError(f"player died at {controller.waypoint.name}")
            if float(state[WON]) > .5:
                if controller.waypoint.name == "exit_switch":
                    controller.reached.append("exit_switch")
                break
        else:
            raise RuntimeError(f"timed out after {max_steps} ticks at {controller.waypoint.name}")
    except RuntimeError as error:
        failure = str(error)

    final = session.state.detach().cpu()
    consumed = torch.nonzero((initial[layout.entity_alive] > .5)
                             & (final[layout.entity_alive] < .5)
                             & (initial[layout.entity_hp] == 0)).flatten()
    consumed_types = data["entity_type"][consumed].tolist()
    report.update({"ticks": ticks, "simulation_seconds": float(final[TIME]),
                   "wall_seconds": time.perf_counter() - started,
                   "final_position": final[X:Y + 1].tolist(), "health_remaining": float(final[HEALTH]),
                   "minimum_health": minimum_health, "damage_observed": damage,
                   "kills": int(final[KILLS]), "pickups": len(consumed_types),
                   "pickup_types": consumed_types, "door_sectors_opened": sorted(opened),
                   "floor_sectors_changed": sorted(floor_changes),
                   "waypoints_reached": controller.reached, "enemy_indices_engaged": sorted(controller.engaged),
                   "control_activity": control_activity, "completed": bool(final[WON] > .5),
                   "state_sha256": hashlib.sha256(final.numpy().tobytes()).hexdigest(),
                   "distinct_frames": len(frame_digests)})
    if require_render and isinstance(frame, torch.Tensor):
        report["frame_shape"] = list(frame.shape)
        report["frame_sha256"] = hashlib.sha256(frame.detach().cpu().numpy().tobytes()).hexdigest()
    checks = (
        (bool(final[WON] > .5), "network did not signal level completion"),
        (int(final[KILLS]) >= 1, "route did not demonstrate a combat kill"),
        (2014 in consumed_types, "route did not collect its original health bonus"),
        ({4, 76, 81} <= opened, "route did not operate the three required doors"),
        (not require_render or len(frame_digests) >= 10, "rendered gameplay frames did not vary"),
    )
    if failure is None:
        failure = next((reason for passed, reason in checks if not passed), None)
    if failure is not None:
        raise AcceptanceFailure(failure, report)
    report["status"] = "passed"
    return report, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wad", type=Path, required=True)
    parser.add_argument("--branch", choices=STARTS, default="right")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--frame", type=Path, help="save the final network-rendered frame as PNG")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    if arguments.max_steps <= 0:
        parser.error("--max-steps must be positive")
    from doomnet.assets import build_assets
    from doomnet.runtime import RuntimeModel

    data, metadata = build_assets(arguments.wad)
    session = RuntimeModel(arguments.model, device=arguments.device, threads=arguments.threads)
    source = session.manifest["source"]
    if (source["wad"]["md5"], source["skill"], source["level"]) != (
            metadata["wad"]["md5"], metadata["skill"], metadata["level"]):
        raise ValueError("the test WAD/skill/level must match the exported model source")
    try:
        report, frame = run_acceptance(session, data, metadata, arguments.branch, arguments.max_steps)
    except AcceptanceFailure as error:
        report, frame = error.report, None
    report["artifact_files"] = session.manifest["files"]
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(encoded)
    if arguments.frame is not None and frame is not None:
        from PIL import Image
        arguments.frame.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame.detach().cpu().numpy()).save(arguments.frame)
    print(encoded, end="")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
