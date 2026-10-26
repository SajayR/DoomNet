"""Behavioral checks for the circuit, independent of licensed WAD availability."""

import math
import itertools
from pathlib import Path
import warnings

import pytest
import torch

from doomnet import state as S
from doomnet.geometry import Geometry
from doomnet.simulation import Simulation


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data():
    # A three-sector strip with a real, closed two-sided door. Its clockwise
    # boundary and BSP conventions match the WAD contract.
    return {
        "vertices": torch.tensor([[0., 0.], [0., 128.], [320., 128.], [320., 0.],
                                   [128., 128.], [128., 0.], [144., 0.], [144., 128.],
                                   [80., 128.], [80., 0.]]),
        "lines": torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [6, 7], [8, 9]]),
        "line_front_sector": torch.tensor([0, 0, 2, 0, 0, 2, 0]),
        "line_back_sector": torch.tensor([-1, -1, -1, -1, 1, 1, 0]),
        "line_flags": torch.tensor([0, 0, 0, 0, 4, 4, 4]),
        "line_special": torch.tensor([0, 0, 11, 0, 1, 1, 0]),
        "line_tag": torch.zeros(7, dtype=torch.long),
        "sector_floor": torch.tensor([0., 0., 0.]),
        "sector_ceiling": torch.tensor([128., 0., 128.]),
        "sector_special": torch.tensor([0, 0, 9]),
        "sector_tag": torch.zeros(3),
        "node_weight": torch.tensor([[1., 0.], [1., 0.]]),
        "node_bias": torch.tensor([-128., -144.]),
        "node_origin": torch.tensor([[128., 0.], [144., 0.]]),
        "node_children": torch.tensor([[-1, -2], [0, -3]]),
        "subsector_sector": torch.tensor([0, 1, 2]),
        "player_start": torch.tensor([32., 64., 0.]),
        "entity_pos": torch.tensor([[250., 96.], [250., 64.], [32., 96.],
                                     [32., 32.], [180., 24.], [250., 24.]]),
        "entity_type": torch.tensor([3004, 3001, 2012, 2007, 2001, 2035]),
        "entity_angle": torch.tensor([math.pi, math.pi, 0., 0., 0., 0.]),
        "entity_flags": torch.tensor([10, 10, 2, 2, 2, 2]),
    }


def controls(forward=0, strafe=0, turn=0, fire=0, use=0, restart=0, weapon=-1, run=0):
    return torch.tensor([forward, strafe, turn, fire, use, restart, weapon, run]).float()


def advance(model, state, control, ticks):
    for _ in range(ticks):
        state = model(state, control, torch.tensor([1 / 35]))
    return state


def frozen_transition(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return torch.jit.freeze(torch.jit.trace(
            model.eval(), (model.spawn.clone(), controls(), torch.tensor([1 / 35]))))


def place(model, state, x, y, angle=0):
    state = state.clone()
    state[S.X:S.Y + 1] = torch.tensor([x, y])
    state[S.ANGLE] = angle
    sector = model.geometry.sector_at(state[S.X:S.Y + 1][None, :])[0]
    state[S.Z] = state[model.layout.sector_floor][sector]
    return state


def remove_enemies(model, state):
    state[model.layout.entity_hp][model.enemy] = 0
    state[model.layout.entity_alive][model.enemy] = 0
    return state


def test_bsp_is_an_affine_threshold_path_network(data):
    geometry = Geometry(data)
    points = torch.tensor([[16., 64.], [128., 64.], [143., 64.], [144., 64.], [145., 64.]])
    assert geometry.sector_at(points).tolist() == [0, 0, 1, 1, 2]
    assert geometry.sector_classifier.path_weight.shape == (3, 2)
    assert set(geometry.sector_classifier.path_weight.unique().tolist()) == {-1., 0., 1.}


def test_original_e1m1_bsp_matches_native_axis_ties():
    wad_path = Path(__file__).resolve().parents[1] / "assets" / "doom1.wad"
    if not wad_path.is_file():
        pytest.skip("original shareware WAD has not been acquired")
    from doomnet.assets import build_assets
    from doomnet.wad import Wad, records

    original, _ = build_assets(wad_path)
    nodes = records(Wad(wad_path).map_lumps("E1M1")["NODES"], "<12h2H", "NODES")
    # Node origins and adjacent integer coordinates deliberately exercise
    # equality at horizontal, vertical, and general native partitions.
    points = [(node[0] + dx, node[1] + dy) for node in nodes
              for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))]
    points.extend(map(tuple, original["vertices"].long().tolist()))

    def native_sector(x, y):
        index = len(nodes) - 1
        while not index & 32768:
            nx, ny, dx, dy, *rest = nodes[index]
            if dx == 0:
                side = dy > 0 if x <= nx else dy < 0
            elif dy == 0:
                side = dx < 0 if y <= ny else dx > 0
            elif (dy ^ dx ^ (x - nx) ^ (y - ny)) < 0:
                side = (dy ^ (x - nx)) < 0
            else:
                side = (y - ny) * dx >= (x - nx) * dy
            index = rest[8 + int(side)]
        return int(original["subsector_sector"][index & 32767])

    expected = torch.tensor([native_sector(x, y) for x, y in points])
    actual = Geometry(original).sector_at(torch.tensor(points).float())
    assert torch.equal(actual, expected)


def test_zero_timestep_is_identity_and_restart_restores_every_activation(data):
    model = Simulation(data)
    initial = model.spawn.clone()
    assert torch.equal(model(initial, controls(1, 1, 1, 1, 1, weapon=0),
                             torch.zeros(1)), initial)
    dead = initial.clone()
    dead[S.HEALTH] = 0
    dead[S.WON] = 1
    dead[model.layout.entity_x] += 12
    dead[model.layout.sector_ceiling] += 4
    assert torch.equal(model(dead, controls(restart=1), torch.zeros(1)), initial)
    assert torch.equal(advance(model, dead, controls(forward=1, fire=1), 3), dead)


def test_circle_motion_stops_at_closed_door_and_outer_walls(data):
    model = Simulation(data)
    state = advance(model, model.spawn.clone(), controls(forward=1, run=1), 30)
    assert 95 < state[S.X] <= 112
    assert state[S.Z] == 0
    state = advance(model, state, controls(strafe=1, run=1), 30)
    assert 16 <= state[S.Y] < 30
    assert state[S.HEALTH] == 100


@pytest.mark.parametrize("height,passable", [(24., True), (25., False)])
def test_maximum_step_height_is_enforced(data, height, passable):
    data["sector_floor"][1] = height
    data["sector_ceiling"][1] = 128
    model = Simulation(data)
    state = advance(model, model.spawn.clone(), controls(forward=1), 20)
    assert bool(state[S.X] > 128) is passable
    assert state[S.Z] == (height if passable else 0)


def descending_ledge(data):
    data["sector_floor"] = torch.tensor([104., -8., -8.])
    data["sector_ceiling"] = torch.full((3,), 192.)
    data["line_special"] = torch.tensor([0, 0, 0, 0, 1, 1, 0])
    data["entity_angle"] = torch.zeros(6)  # The isolated movement test never awakens them.
    data["player_start"] = torch.tensor([112., 64., 0.])
    return data


@pytest.mark.parametrize("strict_lowered", [False, True])
def test_walk_off_high_ledge_retains_radius_support_then_lands_and_keeps_moving(data, strict_lowered):
    from doomnet.neural_ir import audit_neural_graph
    from doomnet.neural_lowering import lower_neural_method

    data = descending_ledge(data)
    model = Simulation(data)
    transition = model
    if strict_lowered:
        examples = (model.spawn.clone(), controls(), torch.tensor([1 / 35]))
        bounds = [S.state_bounds(data), (torch.tensor([-1., -1., -1., 0., 0., 0., -1., 0.]),
                                        torch.tensor([1., 1., 1., 1., 1., 1., 2., 1.])),
                  (torch.tensor([0.]), torch.tensor([1 / 35]))]
        network, _ = lower_neural_method(frozen_transition(model), "forward", bounds, source_modules=model)
        transition = torch.jit.freeze(torch.jit.trace(network.eval(), examples, check_trace=False))
        audit, _ = audit_neural_graph(transition, input_shapes=[x.shape for x in examples])
        assert audit["opaque_operators"] == audit["dynamic_weights"] == audit["dynamic_indices"] == 0
    state = model.spawn.clone()
    supported_over_edge = False
    for _ in range(14):
        state = transition(state, controls(forward=1), torch.tensor([1 / 35]))
        sector = int(model.geometry.sector_at(state[:2][None, :])[0])
        if state[S.X] < 144 and sector == 1:
            assert state[S.Z] == 104
            supported_over_edge = True
        elif state[S.X] >= 144:
            assert state[S.Z] == -8
        assert model.geometry.can_move(state[:2][None, :], state[S.Z:S.Z + 1],
                                       state[model.layout.sector_floor], state[model.layout.sector_ceiling],
                                       model.player_radius)
    assert supported_over_edge and state[S.X] > 180 and state[S.Z] == -8
    assert state[S.HEALTH] == 100


def test_pooled_contact_floor_and_ceiling_include_center_and_preserve_tangency(data):
    data = descending_ledge(data)
    data["sector_floor"] = torch.tensor([0., -8., 24.])
    data["sector_ceiling"] = torch.tensor([70., 192., 192.])
    geometry = Geometry(data)
    points = torch.tensor([[136., 64.], [176., 64.], [64., 64.]])
    support, ceiling = geometry.body_opening(points, data["sector_floor"], data["sector_ceiling"],
                                             torch.full((3,), 16.))
    assert support.shape == ceiling.shape == (3,)
    assert support.tolist() == [24, 24, 0]
    assert ceiling.tolist() == [70, 192, 70]
    # Each portal individually has enough headroom from its own floor, but the
    # combined floor24/ceiling70 opening cannot fit a height56 body.
    assert not geometry.can_move(points[:1], torch.tensor([0.]), data["sector_floor"],
                                  data["sector_ceiling"], torch.tensor([16.]))
    center_low = data["sector_ceiling"].clone()
    center_low[2] = 40
    assert not geometry.can_move(points[1:2], torch.tensor([24.]), data["sector_floor"],
                                  center_low, torch.tensor([16.]))
    tangent = torch.tensor([[112., 64.]])
    floor, _ = geometry.body_opening(tangent, torch.tensor([-8., 104., 104.]),
                                     data["sector_ceiling"], torch.tensor([16.]))
    assert floor.item() == -8


@pytest.mark.parametrize("special", [7, 9])
def test_center_sector_effects_require_actual_floor_contact_at_ledge(data, special):
    data = descending_ledge(data)
    data["sector_special"] = torch.tensor([0, special, special])
    model = Simulation(data)
    state = advance(model, model.spawn.clone(), controls(forward=1), 4)
    assert state[S.X] > 128 and state[S.Z] == 104
    health, secrets = state[S.HEALTH].clone(), state[S.SECRETS].clone()
    state = advance(model, state, controls(), 33)
    assert state[S.HEALTH] == health and state[S.SECRETS] == secrets == 0
    state = advance(model, state, controls(forward=1), 7)
    assert state[S.Z] == -8
    if special == 9:
        assert state[S.SECRETS] >= 1
    else:
        before = state[S.HEALTH].clone()
        state = advance(model, state, controls(), 33)
        assert before - state[S.HEALTH] == 5


def test_moving_enemy_clears_ledge_and_its_center_floor_drop_can_be_collected(data):
    data = descending_ledge(data)
    data["player_start"] = torch.tensor([180., 64., math.pi])
    data["entity_pos"][0] = torch.tensor([136., 64.])
    model = Simulation(data)
    positions = data["entity_pos"][:1]
    feet, _ = model.geometry.body_opening(positions, data["sector_floor"], data["sector_ceiling"],
                                         model.radius[:1])
    assert feet.item() == 104
    for _ in range(12):
        positions = model._slide(positions, torch.tensor([[3., 0.]]), feet,
                                 data["sector_floor"], data["sector_ceiling"], model.radius[:1],
                                 data["entity_pos"], torch.zeros(6, dtype=torch.bool), torch.tensor([0]))
        feet, _ = model.geometry.body_opening(positions, data["sector_floor"], data["sector_ceiling"],
                                             model.radius[:1])
        assert model.geometry.can_move(positions, feet, data["sector_floor"], data["sector_ceiling"], model.radius[:1])
    assert positions[0, 0] > 164 and feet.item() == -8
    # A corpse can remain supported above its dropped clip. Collection must use
    # the same center-floor height as the rendered drop, rather than corpse feet.
    state = model.spawn.clone()
    state[S.X] = 150
    state[model.layout.entity_hp][0] = 0
    state[model.layout.entity_alive][0] = 0
    state[model.layout.entity_drop][0] = 1
    before = state[S.BULLETS].clone()
    state = advance(model, state, controls(), 1)
    assert state[S.BULLETS] == before + 5 and state[model.layout.entity_drop][0] == 0


def test_manual_door_use_is_rising_edge_and_actor_prevents_closing(data):
    model = Simulation(data)
    state = place(model, remove_enemies(model, model.spawn.clone()), 96, 64)
    state = advance(model, state, controls(use=1), 75)
    assert state[model.layout.sector_ceiling][1] == 124
    assert state[model.layout.sector_motion][1] == 2
    assert state[model.layout.line_used][4] == 1
    # The door can remain open while its ordinary waiting period expires.
    state = place(model, state, 136, 64)
    state = advance(model, state, controls(), 180)
    assert state[model.layout.sector_ceiling][1] >= 122
    # Releasing and using the door from a remote position cannot trigger it.
    state = place(model, state, 16, 64)
    state = advance(model, state, controls(use=1), 200)
    assert state[model.layout.sector_ceiling][1] == 0


@pytest.mark.parametrize("special", [36, 88])
def test_walk_floor_specials_lower_and_lift_returns(data, special):
    data["line_special"][4:6] = 0
    data["line_special"][6] = special
    data["line_tag"][6] = 1
    data["sector_tag"][1] = 1
    data["sector_floor"][1] = 64
    data["sector_ceiling"][1] = 128
    model = Simulation(data)
    state = advance(model, model.spawn.clone(), controls(forward=1), 12)
    assert state[model.layout.line_used][6] == 1
    assert state[model.layout.sector_floor][1] < 64
    state = advance(model, state, controls(), 18)
    expected = 8 if special == 36 else 0
    assert state[model.layout.sector_floor][1] == expected
    state = advance(model, state, controls(), 145)
    assert state[model.layout.sector_floor][1] == (8 if special == 36 else 64)


def test_exit_requires_the_actual_switch_in_reach_and_facing(data):
    model = Simulation(data)
    state = remove_enemies(model, model.spawn.clone())
    state = place(model, state, 288, 64, math.pi)
    state = advance(model, state, controls(use=1), 1)
    assert state[S.WON] == 0
    state = advance(model, state, controls(), 1)
    state = place(model, state, 288, 64)
    state = advance(model, state, controls(use=1), 1)
    assert state[S.WON] == 1
    assert state[model.layout.line_used][2] == 1


def test_medkit_and_ammo_only_disappear_when_they_can_be_used(data):
    model = Simulation(data)
    state = place(model, model.spawn.clone(), 32, 80)
    state = advance(model, state, controls(), 1)
    assert state[model.layout.entity_alive][2] == 1
    state[S.HEALTH] = 50
    state = advance(model, state, controls(), 1)
    assert state[S.HEALTH] == 75
    assert state[model.layout.entity_alive][2] == 0
    state = place(model, state, 32, 32)
    state[S.BULLETS] = 200
    state = advance(model, state, controls(), 1)
    assert state[model.layout.entity_alive][3] == 1
    state[S.BULLETS] = 195
    state = advance(model, state, controls(), 1)
    assert state[S.BULLETS] == 200
    assert state[model.layout.entity_alive][3] == 0


def test_weapon_ammo_cooldown_wall_occlusion_and_shotgun_pellets(data):
    model = Simulation(data)
    state = advance(model, model.spawn.clone(), controls(fire=1), 1)
    assert state[S.BULLETS] == 49
    assert state[model.layout.entity_hp][0] == 20  # the closed door blocks aim
    state = advance(model, state, controls(fire=1), 1)
    assert state[S.BULLETS] == 49
    state = place(model, model.spawn.clone(), 180, 96)
    state[S.WEAPON] = 2
    state[S.SHOTGUN_OWNED] = 1
    state[S.SHELLS] = 2
    state = advance(model, state, controls(fire=1), 1)
    assert state[S.SHELLS] == 1
    assert state[model.layout.entity_hp][0] == 0
    assert state[model.layout.entity_alive][0] == 0
    assert state[model.layout.entity_drop][0] == 1
    assert state[S.KILLS] == 1


def corner_wall(data):
    start = len(data["vertices"])
    data["vertices"] = torch.cat((data["vertices"], torch.tensor([[220., 64.], [220., 90.]])))
    data["lines"] = torch.cat((data["lines"], torch.tensor([[start, start + 1]])))
    for key, value in (("line_front_sector", 2), ("line_back_sector", -1),
                       ("line_flags", 0), ("line_special", 0), ("line_tag", 0)):
        data[key] = torch.cat((data[key], data[key].new_tensor([value])))


@pytest.mark.parametrize("weapon", [0, 1])
def test_frozen_shot_ray_cannot_use_clear_center_sight_at_wall_corner(data, weapon):
    corner_wall(data)
    model = Simulation(data)
    frozen = frozen_transition(model)
    state = place(model, remove_enemies(model, model.spawn.clone()), 180, 96, -.18)
    state[model.layout.entity_hp][0] = 20
    state[model.layout.entity_alive][0] = 1
    center_clear = model.geometry.visible(
        state[:2][None, :], torch.tensor([[250., 96.]]), torch.tensor([41.]),
        torch.tensor([28.]), state[model.layout.sector_floor], state[model.layout.sector_ceiling])
    assert center_clear.item()
    blocked = advance(frozen, state, controls(fire=1, weapon=weapon), 1)
    assert blocked[model.layout.entity_hp][0] == 20
    assert blocked[S.FIRE_FLASH] > 0
    state[S.ANGLE] = 0  # the center ray passes above the finite wall segment
    clear = advance(frozen, state, controls(fire=1, weapon=weapon), 1)
    assert clear[model.layout.entity_hp][0] < 20


def test_frozen_shotgun_checks_each_pellet_ray_independently(data):
    corner_wall(data)
    data["entity_pos"][1] = torch.tensor([250., 96.])
    model = Simulation(data)
    state = place(model, remove_enemies(model, model.spawn.clone()), 180, 96, -.12)
    state[model.layout.entity_hp][1] = 60
    state[model.layout.entity_alive][1] = 1
    state[S.SHOTGUN_OWNED] = 1
    state[S.SHELLS] = 2
    partial = advance(frozen_transition(model), state, controls(fire=1, weapon=2), 1)
    # At x=220 the -.21,-.18,-.15 rays strike the wall below its y=90 end.
    # The other four rays pass its end and deal 5+10+15+5 damage with seed1.
    assert partial[model.layout.entity_hp][1] == 25
    assert partial[S.SHELLS] == 1
    data["line_back_sector"][-1] = 2  # turn the wall into an open internal line
    clear_model = Simulation(data)
    complete = advance(frozen_transition(clear_model), state, controls(fire=1, weapon=2), 1)
    assert complete[model.layout.entity_hp][1] == 0


def test_frozen_actual_impact_ray_preserves_vertical_portal_autoaim(data):
    data["sector_floor"] = torch.tensor([0., 40., 64.])
    data["sector_ceiling"] = torch.tensor([128., 70., 128.])
    data["line_special"][4:6] = 0
    data["entity_pos"][0] = torch.tensor([210., 96.])
    model = Simulation(data)
    state = place(model, model.spawn.clone(), 96, 96)
    state[model.layout.entity_hp][1] = 0
    state[model.layout.entity_alive][1] = 0
    # The center sight ray falls below the far portal floor. A steeper ray to
    # the body's front impact traverses both openings and hits its torso.
    center_clear = model.geometry.visible(
        state[:2][None, :], torch.tensor([[210., 96.]]), torch.tensor([41.]),
        torch.tensor([92.]), state[model.layout.sector_floor], state[model.layout.sector_ceiling])
    assert not center_clear.item()
    result = advance(frozen_transition(model), state, controls(fire=1, weapon=1), 1)
    assert result[model.layout.entity_hp][0] == 15


def test_frozen_nearest_body_blocks_more_distant_target(data):
    data["entity_pos"][5] = torch.tensor([215., 96.])
    model = Simulation(data)
    state = place(model, model.spawn.clone(), 180, 96)
    state[model.layout.entity_hp][1] = 0
    state[model.layout.entity_alive][1] = 0
    result = advance(frozen_transition(model), state, controls(fire=1, weapon=1), 1)
    assert result[model.layout.entity_hp][5] == 15
    assert result[model.layout.entity_hp][0] == 20


@pytest.mark.parametrize("actor,x,occupied", [
    ("player", 120., True), ("player", 136., True),
    ("player", 112., False), ("player", 111., False),
    ("enemy", 120., True), ("enemy", 108., False),
    ("dead_enemy", 120., False),
])
def test_frozen_door_uses_body_overlap_and_closes_when_clear(data, actor, x, occupied):
    data["entity_pos"][0] = torch.tensor([x, 64.])
    data["entity_angle"][0] = 0  # keep the idle guard facing away from the player
    model = Simulation(data)
    state = remove_enemies(model, model.spawn.clone())
    state[model.layout.sector_ceiling][1] = 124
    state[model.layout.sector_motion][1] = 2
    state[model.layout.sector_timer][1] = 0
    if actor == "player":
        state = place(model, state, x, 64)
    elif actor == "enemy":
        state[model.layout.entity_hp][0] = 20
        state[model.layout.entity_alive][0] = 1
    frozen = frozen_transition(model)
    state = advance(frozen, state, controls(), 1)
    assert state[model.layout.sector_ceiling][1] == (124 if occupied else 122)
    state = advance(frozen, state, controls(), 75)
    assert state[model.layout.sector_ceiling][1] == (124 if occupied else 0)
    if actor == "player":
        assert model.geometry.can_move(state[:2][None, :], state[S.Z:S.Z + 1],
                                       state[model.layout.sector_floor], state[model.layout.sector_ceiling],
                                       torch.tensor([16.])).item()
    elif actor == "enemy":
        position = torch.stack((state[model.layout.entity_x][0],
                                state[model.layout.entity_y][0]))[None, :]
        assert model.geometry.can_move(position, torch.tensor([0.]),
                                       state[model.layout.sector_floor], state[model.layout.sector_ceiling],
                                       torch.tensor([20.])).item()


@pytest.fixture(scope="module")
def original_transition():
    wad_path = Path(__file__).resolve().parents[1] / "assets" / "doom1.wad"
    if not wad_path.is_file():
        pytest.skip("original shareware WAD has not been acquired")
    from doomnet.assets import build_assets
    data, _ = build_assets(wad_path)
    model = Simulation(data)
    return data, model, frozen_transition(model)


def test_frozen_original_route_corner_shot_stops_at_linedefs_383_and_381(original_transition):
    from scripts.validate_level import E1M1Controller
    data, model, frozen = original_transition
    state = model.spawn.clone()
    controller = E1M1Controller(data, "right")
    for _ in range(617):
        state = advance(frozen, state, controller.controls(state), 1)
    result = advance(frozen, state, controls(turn=-.5, fire=1, weapon=1), 1)
    direction = torch.stack((result[S.ANGLE].cos(), result[S.ANGLE].sin()))[None, :]
    distance, hit = model.geometry.ray_intersections(state[:2][None, :], direction)
    assert model.geometry.one_sided[383] and model.geometry.one_sided[381]
    assert hit[0, 383] and hit[0, 381]
    assert 160 < distance[0, 383] < distance[0, 381] < 200
    assert state[model.layout.entity_hp][5] == 20
    assert result[model.layout.entity_hp][5] == 20
    assert result[S.BULLETS] == state[S.BULLETS] - 1


def test_frozen_original_first_door_wait_with_body_eight_units_outside(original_transition):
    from scripts.validate_level import E1M1Controller
    data, model, frozen = original_transition
    state = model.spawn.clone()
    controller = E1M1Controller(data, "right")
    for _tick in range(500):
        state = advance(frozen, state, controller.controls(state), 1)
        if "first_door" in controller.reached:
            break
    assert "first_door" in controller.reached
    # Continue using controls only, stopping eight map units before line151.
    for _tick in range(10):
        distance = 1528 - float(state[S.X])
        state = advance(frozen, state, controls(forward=min(1, distance / (180 / 35))), 1)
    assert abs(float(state[S.X]) - 1528) < .001
    assert model.geometry.sector_at(state[:2][None, :]).item() == 3
    assert state[model.layout.sector_ceiling][4] == 68
    state = advance(frozen, state, controls(), 250)
    assert state[S.HEALTH] > 0
    assert state[model.layout.sector_ceiling][4] == 68
    assert model.geometry.can_move(state[:2][None, :], state[S.Z:S.Z + 1],
                                   state[model.layout.sector_floor], state[model.layout.sector_ceiling],
                                   torch.tensor([16.])).item()


def test_fist_and_dropped_ammo(data):
    model = Simulation(data)
    state = place(model, model.spawn.clone(), 216, 96)
    state[model.layout.entity_hp][0] = 2
    state[S.WEAPON] = 0
    state = advance(model, state, controls(fire=1), 1)
    assert state[model.layout.entity_hp][0] == 0
    assert state[S.BULLETS] == 50
    state = place(model, state, 240, 96)
    state = advance(model, state, controls(), 1)
    assert state[S.BULLETS] == 55
    assert state[model.layout.entity_drop][0] == 0


def test_enemies_activate_chase_and_imp_projectile_causes_damage(data):
    model = Simulation(data)
    state = advance(model, model.spawn.clone(), controls(), 10)
    assert state[model.layout.entity_active][:2].sum() == 0
    state = place(model, state, 160, 64)
    state[model.layout.entity_hp][0] = 0
    state[model.layout.entity_alive][0] = 0
    state[model.layout.entity_active][1] = 1
    state[model.layout.entity_cooldown][1] = 0
    state = advance(model, state, controls(), 1)
    assert state[model.layout.projectile_alive][1] == 1
    assert state[model.layout.projectile_dx][1] < 0
    initial_x = state[model.layout.entity_x][1].clone()
    state = advance(model, state, controls(), 15)
    assert state[S.HEALTH] < 100
    assert state[model.layout.entity_x][1] < initial_x
    assert state[model.layout.projectile_alive][1] == 0


def test_frozen_projectile_wall_contact_and_inactive_positions_do_not_drift(data):
    model = Simulation(data)
    frozen = frozen_transition(model)
    layout = model.layout
    base = remove_enemies(model, model.spawn.clone())
    lower, upper = S.state_bounds(data)
    for start_x, survives in ((304., True), (304.125, False), (310., False), (313., False)):
        state = base.clone()
        state[layout.projectile_x][1] = start_x
        state[layout.projectile_y][1] = 64
        state[layout.projectile_z][1] = 40
        state[layout.projectile_dx][1] = S.PROJECTILE_SPEED
        state[layout.projectile_dz][1] = -35
        state[layout.projectile_alive][1] = 1
        assert torch.equal(frozen(state, controls(), torch.zeros(1)), state)
        result = advance(frozen, state, controls(), 1)
        assert result[layout.projectile_x][1] == start_x + 10
        assert bool(result[layout.projectile_alive][1]) is survives
        assert ((result >= lower) & (result <= upper)).all()
        if not survives:
            fields = (layout.projectile_x, layout.projectile_y, layout.projectile_z)
            position = torch.stack([result[field] for field in fields])
            later = advance(frozen, result, controls(), 200)
            assert torch.equal(torch.stack([later[field] for field in fields]), position)
            assert later[layout.projectile_dx][1] == S.PROJECTILE_SPEED
            assert later[layout.projectile_dz][1] == -35


def test_frozen_projectile_radius_contacts_a_finite_wall_endpoint(data):
    corner_wall(data)
    model = Simulation(data)
    frozen = frozen_transition(model)
    for y, survives in ((95., False), (96., True)):
        state = remove_enemies(model, model.spawn.clone())
        layout = model.layout
        state[layout.projectile_x][1] = 210
        state[layout.projectile_y][1] = y
        state[layout.projectile_z][1] = 40
        state[layout.projectile_dx][1] = S.PROJECTILE_SPEED
        state[layout.projectile_alive][1] = 1
        # Both center rays pass above the wall; only the first circle touches
        # its endpoint at (220,90). Exact radius equality remains clear.
        assert model.geometry.visible(
            torch.tensor([[210., y]]), torch.tensor([[220., y]]), torch.tensor([40.]),
            torch.tensor([40.]), data["sector_floor"], data["sector_ceiling"]).item()
        result = advance(frozen, state, controls(), 1)
        assert bool(result[layout.projectile_alive][1]) is survives


def test_frozen_projectile_floor_contact_and_original_ceiling_height_tie(data):
    model = Simulation(data)
    frozen = frozen_transition(model)
    layout = model.layout
    for z, velocity, survives in ((10., -350., False), (110., 350., True),
                                  (110.125, 350., False), (50., 0., True)):
        state = remove_enemies(model, model.spawn.clone())
        state[layout.projectile_x][1] = 60
        state[layout.projectile_y][1] = 64
        state[layout.projectile_z][1] = z
        state[layout.projectile_dz][1] = velocity
        state[layout.projectile_alive][1] = 1
        result = advance(frozen, state, controls(), 1)
        assert result[layout.projectile_z][1] == z + velocity / 35
        assert bool(result[layout.projectile_alive][1]) is survives
        assert result[S.HEALTH] == 100


def test_projectile_endpoint_circle_checks_both_portal_heights(data):
    data["sector_floor"][1] = 16
    data["sector_ceiling"][1] = 64
    geometry = Geometry(data)
    # These centers stay in sector0 while their radius overlaps the portal.
    # Missile flags ignore the player/monster-only impassable line flag.
    data["line_flags"][4] |= 1
    blocked_geometry = Geometry(data)
    heights = torch.tensor([16., 16.125, 56., 56.125])
    origins = torch.tensor([[120., 64.]]).expand(4, -1)
    targets = torch.tensor([[124., 64.]]).expand(4, -1)
    expected = torch.tensor([False, True, True, False])
    assert torch.equal(geometry.projectile_clear(
        origins, targets, heights, heights, data["sector_floor"], data["sector_ceiling"]), expected)
    assert torch.equal(blocked_geometry.projectile_clear(
        origins, targets, heights, heights, data["sector_floor"], data["sector_ceiling"]), expected)


def test_projectile_endpoint_clearance_is_symmetric_at_an_outer_wall(data):
    geometry = Geometry(data)
    points = torch.tensor([[-6.125, 64.], [-6., 64.], [-5.875, 64.],
                           [5.875, 64.], [6., 64.], [6.125, 64.]])
    heights = torch.full((6,), 40.)
    expected = torch.tensor([True, True, False, False, True, True])
    # Zero-length sweeps isolate the body-clearance decision on either side.
    assert torch.equal(geometry.projectile_clear(
        points, points, heights, heights, data["sector_floor"], data["sector_ceiling"]), expected)


def test_barrel_explosion_and_armor_absorption(data):
    model = Simulation(data)
    state = remove_enemies(model, model.spawn.clone())
    state = place(model, state, 180, 64, math.atan2(-40, 70))
    state[model.layout.entity_hp][5] = 5
    state[S.ARMOR] = 100
    state[S.ARMOR_TYPE] = 1
    state = advance(model, state, controls(fire=1), 1)
    assert state[model.layout.entity_alive][5] == 0
    assert state[S.ARMOR] < 100
    assert state[S.HEALTH] < 100
    assert state[S.ARMOR] + state[S.HEALTH] > 120


def test_spent_blue_armor_does_not_change_later_armor_bonus(data):
    data["sector_special"][0] = 7
    data["entity_type"][3] = 2015
    model = Simulation(data)
    state = place(model, model.spawn.clone(), 32, 32)
    state[S.ARMOR] = 1
    state[S.ARMOR_TYPE] = 2
    state[S.TIME] = .9
    state = advance(model, state, controls(), 1)
    assert state[S.HEALTH] == 96
    assert state[S.ARMOR] == 1
    assert state[S.ARMOR_TYPE] == 1


def test_shotgun_guy_hitscan_and_dropped_weapon(data):
    data["entity_type"][0] = 9
    model = Simulation(data)
    state = place(model, model.spawn.clone(), 208, 96)
    state[model.layout.entity_hp][1] = 0
    state[model.layout.entity_alive][1] = 0
    state[model.layout.entity_active][0] = 1
    state = advance(model, state, controls(), 1)
    assert state[S.HEALTH] == 91
    state[model.layout.entity_hp][0] = 1
    state = advance(model, state, controls(fire=1), 1)
    assert state[model.layout.entity_drop][0] == 1
    state = place(model, state, 240, 96)
    state = advance(model, state, controls(), 1)
    assert state[S.SHOTGUN_OWNED] == 1
    assert state[S.WEAPON] == 2
    assert state[S.SHELLS] == 4


def test_nukage_and_secret_sector_are_recurrent_rules(data):
    data["sector_special"][0] = 7
    model = Simulation(data)
    state = advance(model, model.spawn.clone(), controls(), 33)
    assert state[S.HEALTH] == 95
    state = place(model, state, 180, 112)
    state = advance(model, state, controls(), 3)
    assert state[S.SECRETS] == 1
    assert state[model.layout.sector_secret][2] == 1


def test_frozen_phase_wrap_preserves_the_boundary_hazard_tick(data):
    data["sector_special"][0] = 7
    model = Simulation(data)
    state = remove_enemies(model, model.spawn.clone())
    state[S.TIME] = 95.99
    frozen = frozen_transition(model)
    unwrapped = state[S.TIME] + 1 / 35
    state = advance(frozen, state, controls(), 1)
    assert state[S.TIME] == torch.remainder(unwrapped, S.CLOCK_PERIOD)
    assert 0 < state[S.TIME] < 1 / 35
    assert state[S.HEALTH] == 95
    state = advance(frozen, state, controls(), 1)
    assert state[S.HEALTH] == 95


@pytest.mark.parametrize("skill", range(5))
def test_original_spawn_has_finite_bounds_for_every_recurrent_field(skill):
    wad_path = Path(__file__).resolve().parents[1] / "assets" / "doom1.wad"
    if not wad_path.is_file():
        pytest.skip("original shareware WAD has not been acquired")
    from doomnet.assets import build_assets
    data, _ = build_assets(wad_path, skill=skill)
    lower, upper = S.state_bounds(data)
    state = S.initial_state(data)
    layout = S.StateLayout.from_data(data)
    assert lower.shape == upper.shape == state.shape == (layout.size,)
    assert lower.dtype == upper.dtype == torch.float32
    assert torch.isfinite(lower).all() and torch.isfinite(upper).all()
    assert ((state >= lower) & (state <= upper)).all()
    assert upper[S.TIME] == S.CLOCK_PERIOD
    assert upper[S.KILLS] == torch.isin(data["entity_type"], torch.tensor([9, 3001, 3004])).sum()
    assert upper[S.RNG] == 32748
    assert (upper[layout.projectile_dx] / 35 < 2 * S.PROJECTILE_RADIUS).all()


@pytest.mark.parametrize("branch", ["right", "left"])
def test_frozen_original_routes_remain_within_bounds_and_exit_before_45_seconds(original_transition, branch):
    from scripts.validate_level import E1M1Controller
    data, model, frozen = original_transition
    lower, upper = S.state_bounds(data)
    state = model.spawn.clone()
    controller = E1M1Controller(data, branch)
    # Radius-aware stair support changes combat timing; allow up to 45 seconds.
    for _tick in range(1575):
        state = advance(frozen, state, controller.controls(state), 1)
        assert ((state >= lower) & (state <= upper)).all()
        if state[S.WON] > .5:
            break
    assert state[S.WON] == 1
    assert state[S.HEALTH] > 0
    assert state[S.KILLS] == 6
    assert state[S.TIME] < 45


def test_frozen_graph_changes_with_live_controls_without_python_callbacks(data):
    model = Simulation(data).eval()
    state = model.spawn.clone()
    dt = torch.tensor([1 / 35])
    with warnings.catch_warnings(record=True) as notes:
        frozen = torch.jit.freeze(torch.jit.trace(model, (state, controls(), dt)))
    tracer_notes = [str(note.message) for note in notes
                    if isinstance(note.message, torch.jit.TracerWarning)]
    assert not tracer_notes
    graph = str(frozen.inlined_graph)
    assert "PythonOp" not in graph
    assert "prim::Call" not in graph
    assert "prim::If" not in graph
    assert "aten::Int" not in graph
    assert "aten::linear" in graph and "aten::relu" in graph and "aten::gt" in graph
    varied = [controls(forward=1), controls(strafe=1, turn=1),
              controls(fire=1, weapon=0), controls(use=1), controls(restart=1)]
    outputs = []
    for control in varied:
        expected = model(state, control, dt)
        actual = frozen(state, control, dt)
        assert torch.equal(actual, expected)
        outputs.append(actual)
    assert not torch.equal(outputs[0], outputs[1])
    assert outputs[2][S.FIRE_FLASH] > 0


def test_game_action_circuits_encode_independent_predicate_policies(data):
    model = Simulation(data)
    policies = {
        "play_rule": lambda p: [p[0] and p[1] and p[2]],
        "fire_rule": lambda p: [p[0] and p[1] and p[2]],
        "use_edge_rule": lambda p: [p[0] and not p[1] and p[2]],
        "use_stop_rule": lambda p: [(p[0] or p[1] or p[2] or p[3]) and p[4] and p[5]],
        "use_action_rule": lambda p: [all(p)],
        "pellet_hit_rule": lambda p: [p[0] and p[1] and p[2] and (p[3] or p[4]) and p[5] and p[6]],
        "pellet_action_rule": lambda p: [p[0] and p[1] and (p[2] or p[3]) and p[4]],
        "door_motion_rule": lambda p: [p[1] and p[2] and not p[4],
                                         (p[0] or (p[1] and p[2])) and p[3] and not p[4]],
        "door_finish_rule": lambda p: [p[0] and p[1], p[2] and p[3]],
        "step_clear": lambda p: [p[0] and p[1]],
    }
    for name, policy in policies.items():
        circuit = getattr(model, name)
        rows = list(itertools.product((False, True), repeat=circuit.input_count))
        inputs = torch.tensor(rows).float()
        expected = torch.tensor([policy(row) for row in rows]).float()
        assert torch.equal(circuit(inputs), expected), name
        assert circuit.input_count > 1
        assert circuit.construction_spec()["affine_layers"] == 2


def test_body_clear_network_uses_overlap_solidity_owner_and_all_bodies(data):
    model = Simulation(data)
    rows = list(itertools.product((False, True), repeat=3))
    predicates = torch.zeros(len(rows), model.layout.entity_count, 3)
    predicates[:, 0] = torch.tensor(rows).float()
    expected = torch.tensor([not all(row) for row in rows])
    assert torch.equal(model.body_clear(predicates), expected)
    predicates[:, -1] = 1
    assert not model.body_clear(predicates).any()
    assert model.body_clear.construction_spec()["body_predicates"] == [
        "overlap", "solid", "different_actor"]


def test_door_occupancy_network_requires_live_center_or_boundary_contact(data):
    model = Simulation(data)
    count = model.layout.entity_count + 1
    boundaries = torch.zeros(count, model.layout.line_count)
    centers = torch.zeros(count, len(model.door_index))
    alive = torch.zeros(count)
    boundaries[0, 4] = 1  # player touches the compiled first doorway edge
    assert not model.door_occupancy(boundaries, centers, alive).item()
    alive[0] = 1
    assert model.door_occupancy(boundaries, centers, alive).item()
    alive[0] = 0
    centers[-1, 0] = 1  # a different body is wholly within the doorway
    assert not model.door_occupancy(boundaries, centers, alive).item()
    alive[-1] = 1
    assert model.door_occupancy(boundaries, centers, alive).item()
    assert model.door_occupancy.construction_spec()["actor_count"] == count


def test_neural_selectors_receive_bounded_scores_for_inside_and_behind_bodies(data):
    model = Simulation(data)
    frozen = frozen_transition(model)
    captured = {"use": [], "pellet": []}

    def record(name):
        def hook(circuit, arguments):
            circuit.validate_input(arguments[0])
            captured[name].append(arguments[0].clone())
        return hook

    handles = [model.use_minimum.register_forward_pre_hook(record("use")),
               model.pellet_minimum.register_forward_pre_hook(record("pellet"))]
    try:
        inside = place(model, model.spawn.clone(), 250, 96)
        actual = model(inside, controls(fire=1, use=1), torch.tensor([1 / 35]))
        expected = advance(frozen, inside, controls(fire=1, use=1), 1)
        assert torch.equal(actual, expected)
        assert (captured["pellet"][-1][:, 0] == 0).all()
        assert actual[model.layout.entity_hp][0] == 15
        behind = place(model, model.spawn.clone(), 300, 96)
        actual = model(behind, controls(fire=1), torch.tensor([1 / 35]))
        assert (captured["pellet"][-1][:, 0] == 1e6).all()
        assert actual[model.layout.entity_hp][0] == 20
        # Circle contacts can tie at zero; the lowest original entity index
        # receives the shot, matching the selector's declared stable ordering.
        tied = inside.clone()
        tied[model.layout.entity_x][5] = 250
        tied[model.layout.entity_y][5] = 96
        actual = model(tied, controls(fire=1), torch.tensor([1 / 35]))
        assert (captured["pellet"][-1][:, [0, 5]] == 0).all()
        assert actual[model.layout.entity_hp][0] == 15
        assert actual[model.layout.entity_hp][5] == 20
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured["use"]) == len(captured["pellet"]) == 3
    operations = {node.kind() for node in frozen.inlined_graph.nodes()}
    assert "aten::min" not in operations and "aten::argmin" not in operations
    assert "aten::relu" in operations and "aten::index_select" in operations


def test_neural_use_selector_keeps_stable_first_line_when_stops_tie(data):
    # The two collinear copies encode the same ray depth; normal physical stop
    # ordering chooses the lower original linedef index instead of the exit.
    data["lines"] = torch.cat((data["lines"], data["lines"][4:5]))
    for key in ("line_front_sector", "line_back_sector", "line_flags", "line_special", "line_tag"):
        data[key] = torch.cat((data[key], data[key][4:5]))
    data["line_special"][-1] = 11
    model = Simulation(data)
    frozen = frozen_transition(model)
    state = place(model, model.spawn.clone(), 96, 64)
    actual = advance(frozen, state, controls(use=1), 1)
    assert actual[model.layout.line_used][4] == 1
    assert actual[model.layout.line_used][-1] == 0
    assert actual[S.WON] == 0
