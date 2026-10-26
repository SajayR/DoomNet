"""Projection, portal visibility, changing activations, and portable graph checks."""

import io
import math
from pathlib import Path

import pytest
import torch

from doomnet.assets import build_assets
from doomnet.rendering import Renderer
from doomnet.state import (ANGLE, ARMOR, BULLETS, FIRE_FLASH, HEALTH, WEAPON,
                           WON, X, Y, Z, initial_state)


@pytest.fixture(scope="module")
def assets():
    path = Path(__file__).resolve().parents[1] / "assets" / "doom1.wad"
    if not path.exists():
        pytest.skip("Fetch the original WAD with python scripts/fetch_shareware.py")
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield build_assets(path)[0]
    torch.set_num_threads(threads)


def two_rooms(assets):
    """Two analytical rooms share an opening with a raised floor and low ceiling."""
    data = dict(assets)
    data.update({
        "vertices": torch.tensor([[0., -64.], [0., 64.], [128., 64.], [128., -64.],
                                   [256., 64.], [256., -64.]]),
        "lines": torch.tensor([[0, 1], [1, 2], [3, 0], [2, 3], [2, 4], [4, 5], [5, 3]]),
        "line_front_sector": torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        "line_back_sector": torch.tensor([-1, -1, -1, 1, -1, -1, -1]),
        "line_flags": torch.zeros(7, dtype=torch.long),
        "line_offset": torch.zeros(7, 2, 2),
        "line_texture": torch.tensor([[[0, 1, 2], [0, 1, 2]]] * 7),
        "node_weight": torch.tensor([[1., 0.]]),
        "node_bias": torch.tensor([-128.]),
        "node_origin": torch.tensor([[128., 0.]]),
        "node_children": torch.tensor([[-1, -2]]),
        "subsector_sector": torch.tensor([0, 1]),
        "sector_floor": torch.tensor([0., 24.]),
        "sector_ceiling": torch.tensor([96., 72.]),
        "sector_light": torch.tensor([255., 255.]),
        "sector_floor_tex": torch.tensor([0, 0]),
        "sector_ceiling_tex": torch.tensor([1, 1]),
        "sector_sky": torch.tensor([False, False]),
        "sky_texture": torch.tensor(3),
        "player_start": torch.tensor([64., 0., 0.]),
        "entity_type": torch.tensor([2014]),
        "entity_pos": torch.tensor([[176., 0.]]),
        "entity_angle": torch.zeros(1),
        "entity_walk_sprite": torch.zeros(1, 4, 8, dtype=torch.long),
        "entity_attack_sprite": torch.zeros(1, 8, dtype=torch.long),
        "entity_dead_sprite": torch.tensor([-1]),
        "entity_drop_sprite": torch.tensor([-1]),
        "sprite_size": torch.tensor([[48, 48]]),
        "sprite_offset": torch.tensor([[24, 48]]),
        "texture_size": torch.tensor([[64, 64]] * 4),
    })
    data["line_texture"][3, :, 2] = -1
    colors = torch.tensor([[240, 0, 0, 255], [0, 240, 0, 255],
                            [0, 0, 240, 255], [180, 180, 0, 255]], dtype=torch.uint8)
    data["texture_rgba"] = colors[:, None, None, :].expand(-1, 64, 64, -1).clone()
    data["flat_rgba"] = torch.tensor([[100, 70, 40, 255], [110, 110, 110, 255]],
                                      dtype=torch.uint8)[:, None, None, :].expand(-1, 64, 64, -1).clone()
    data["sprite_rgba"] = torch.tensor([255, 0, 255, 255], dtype=torch.uint8).expand(1, 48, 48, 4).clone()
    for key in ("projectile_sprite", "weapon_sprite", "weapon_fire_sprite", "weapon_flash_sprite",
                "hud_face_sprite", "hud_digit_sprite", "hud_bar_sprite", "hud_percent_sprite",
                "hud_dead_face_sprite"):
        data[key] = torch.zeros_like(assets[key])
    return data


def surface(renderer, state):
    return renderer._surfaces(state, torch.tensor([1., 0.]), torch.tensor([0., -1.]), state[3] + 41)


def colors(renderer, address, illumination):
    return renderer.artwork_rgb[address].float() * illumination[..., None]


def test_texture_coordinates_wrap_exactly_for_each_artwork_size(assets):
    renderer = Renderer(assets, width=64, height=64)
    texture = torch.arange(len(assets["texture_size"]))[:, None]
    coordinate = torch.tensor([[-257.2, -128.7, -32., -0.1, 0., 7.9, 8., 23.75, 24.,
                                31.25, 32., 55.9, 56., 71.9, 72., 127.9, 128., 255.9,
                                256., 1000.8]])
    size = assets["texture_size"]
    u = torch.remainder(coordinate.floor().long(), size[:, 0:1])
    v = torch.remainder((-coordinate).floor().long(), size[:, 1:2])
    expected = texture * (renderer.texture_height * renderer.texture_width) + v * renderer.texture_width + u
    assert torch.equal(renderer._texture_index(texture, coordinate, -coordinate), expected)


def test_sentinel_crossings_do_not_project_unused_wall_coordinates(assets):
    data = two_rooms(assets)
    state = initial_state(data)
    renderer = Renderer(data, width=64, height=64)
    artwork_coordinates = []
    selected_depths = []
    texture_index = renderer._texture_index

    def capture_coordinates(texture, u, v):
        artwork_coordinates.append((u, v))
        return texture_index(texture, u, v)

    renderer._texture_index = capture_coordinates
    handle = renderer.line_selection.register_forward_hook(
        lambda module, inputs, output: selected_depths.append(output[0]))
    try:
        with torch.inference_mode():
            renderer._surfaces(state, torch.tensor([1., 1e-8]),
                               torch.tensor([1e-8, -1.]), state[Z] + 41)
    finally:
        handle.remove()
    assert bool((selected_depths[0] == 1e6).any())
    u, v = artwork_coordinates[0]
    assert bool(((u >= 0) & (u <= 128)).all())
    assert bool(((v >= 0) & (v <= 160)).all())


def test_sector_intervals_render_steps_lower_ceiling_and_closed_door(assets):
    data = two_rooms(assets)
    state = initial_state(data)
    renderer = Renderer(data, width=64, height=64)
    _, _, depth, _ = surface(renderer, state)
    assert float(depth[0, 32]) == pytest.approx(64)  # Upper wall at the opening.
    assert float(depth[10, 32]) == pytest.approx(31 / (13.5 / 32))  # Far room ceiling.
    assert float(depth[20, 32]) == pytest.approx(192)  # View through the open portal.
    assert float(depth[36, 32]) == pytest.approx(64)  # Raised floor's lower wall.
    assert float(depth[46, 32]) == pytest.approx(41 / (22.5 / 32))  # Near room floor.
    closed = state.clone()
    closed[renderer.layout.sector_ceiling][1] = 24
    _, _, closed_depth, _ = surface(renderer, closed)
    assert float(closed_depth[20, 32]) == pytest.approx(64)


def test_sky_ceiling_continues_across_sector_height_changes(assets):
    data = two_rooms(assets)
    data["sector_sky"] = torch.ones(2, dtype=torch.bool)
    state = initial_state(data)
    renderer = Renderer(data, width=64, height=64)
    address, illumination, _, _ = surface(renderer, state)
    image = colors(renderer, address, illumination)
    assert image[0, 32].tolist() == [180, 180, 0]
    data["sector_sky"] = torch.tensor([True, False])
    renderer = Renderer(data, width=64, height=64)
    address, illumination, depth, _ = surface(renderer, state)
    image = colors(renderer, address, illumination)
    assert float(depth[0, 32]) == pytest.approx(64)
    assert float(image[0, 32, 0]) > 200
    assert float(image[0, 32, 1]) == 0


def test_masked_portal_selects_the_first_opaque_surface(assets):
    data = two_rooms(assets)
    data["line_texture"][3, :, 2] = 2
    state = initial_state(data)
    renderer = Renderer(data, width=64, height=64)
    address, illumination, depth, _ = surface(renderer, state)
    image = colors(renderer, address, illumination)
    assert float(depth[20, 32]) == pytest.approx(64)
    opaque = image[20, 32].clone()
    data["texture_rgba"][2, ..., 3] = 0
    renderer = Renderer(data, width=64, height=64)
    address, illumination, depth, _ = surface(renderer, state)
    image = colors(renderer, address, illumination)
    assert float(depth[20, 32]) == pytest.approx(192)
    assert not torch.equal(image[20, 32], opaque)


def test_exterior_sky_cannot_use_a_nonintersecting_linedefs_sector(assets):
    state = initial_state(assets)
    state[X], state[Y], state[ANGLE] = 2624, -3120, math.pi
    renderer = Renderer(assets)
    sector = renderer.geometry.sector_at(state[X:Y + 1].reshape(1, 2))[0]
    state[Z] = state[renderer.layout.sector_floor][sector]
    forward = torch.stack((state[ANGLE].cos(), state[ANGLE].sin()))
    right = torch.stack((forward[1], -forward[0]))
    address, illumination, depth, _ = renderer._surfaces(state, forward, right, state[Z] + 41)
    image = colors(renderer, address, illumination)
    # This sky ray exits the original outdoor area before any plane crossing.
    # A sentinel's arbitrary linedef must not supply a remote indoor ceiling.
    assert float(depth[43, 28]) == 1e6
    assert image[43, 28].tolist() == [27, 27, 27]


def test_billboards_use_surface_depth_and_patch_transparency(assets):
    data = two_rooms(assets)
    state = initial_state(data)
    renderer = Renderer(data, width=64, height=64)
    address, illumination, depth, floor = surface(renderer, state)
    sprite_address, sprite_illumination = renderer._sprites(
        state, address, illumination, depth, floor,
        torch.tensor([1., 0.]), torch.tensor([0., -1.]), state[3] + 41)
    sprite = colors(renderer, sprite_address, sprite_illumination)
    assert float(sprite[20, 32, 0]) > 200
    assert float(sprite[20, 32, 2]) > 200
    behind = state.clone()
    behind[renderer.layout.entity_x] = 300
    hidden_address, hidden_illumination = renderer._sprites(
        behind, address, illumination, depth, floor,
        torch.tensor([1., 0.]), torch.tensor([0., -1.]), state[3] + 41)
    assert torch.equal(hidden_address, address)
    assert torch.equal(hidden_illumination, illumination)
    data["sprite_rgba"][..., 3] = 0
    transparent = Renderer(data, width=64, height=64)
    hidden_address, hidden_illumination = transparent._sprites(
        state, address, illumination, depth, floor,
        torch.tensor([1., 0.]), torch.tensor([0., -1.]), state[3] + 41)
    assert torch.equal(hidden_address, address)
    assert torch.equal(hidden_illumination, illumination)


def test_moving_actor_and_corpse_use_ledge_support_but_pickup_and_drop_keep_center_floor(assets):
    data = two_rooms(assets)
    data["sector_floor"] = torch.tensor([104., -8.])
    data["sector_ceiling"] = torch.full((2,), 192.)
    data["entity_pos"] = torch.tensor([[136., 0.]])
    data["entity_type"] = torch.tensor([3004])
    data["entity_dead_sprite"] = torch.tensor([0])
    data["entity_drop_sprite"] = torch.tensor([0])
    state = initial_state(data)
    renderer = Renderer(data, width=64, height=64)
    address, illumination, depth, floor = surface(renderer, state)

    def billboard(current, model=renderer):
        return model._sprites(current, address, illumination, depth, floor,
                              torch.tensor([1., 0.]), torch.tensor([0., -1.]), state[Z] + 41)[0]

    living = billboard(state)
    assert living[30, 32] != address[30, 32]
    corpse = state.clone()
    corpse[renderer.layout.entity_alive] = corpse[renderer.layout.entity_hp] = 0
    assert torch.equal(billboard(corpse), living)
    data["entity_dead_sprite"] = torch.tensor([-1])
    drop_renderer = Renderer(data, width=64, height=64)
    corpse[renderer.layout.entity_drop] = 1
    assert torch.equal(billboard(corpse, drop_renderer), address)
    data["entity_type"] = torch.tensor([2014])
    pickup_renderer = Renderer(data, width=64, height=64)
    assert torch.equal(billboard(state, pickup_renderer), address)


def test_weapon_and_hud_select_unlit_art_and_transparent_background(assets):
    data = two_rooms(assets)
    data["sector_light"].zero_()
    state = initial_state(data)
    renderer = Renderer(data, width=64, height=64)
    with torch.inference_mode():
        image = renderer(state)
        assert image[8, 2].tolist() == [255, 0, 255]
        assert image[renderer.scene_height, 0].tolist() == [255, 0, 255]
        data["sprite_rgba"][..., 3] = 0
        transparent = Renderer(data, width=64, height=64)(state)
        assert transparent[8, 2].tolist() == [21, 21, 21]
        assert not bool(transparent[renderer.scene_height:].any())


def test_original_scene_weapon_status_and_end_messages_react_to_state(assets):
    state = initial_state(assets)
    renderer = Renderer(assets)
    with torch.inference_mode():
        image = renderer(state)
        assert image.shape == (120, 160, 3) and image.dtype == torch.uint8
        assert len(torch.unique(image[:100].reshape(-1, 3), dim=0)) > 100
        turned = state.clone()
        turned[ANGLE] += math.pi / 2
        assert not torch.equal(image[:100], renderer(turned)[:100])
        changed = state.clone()
        changed[HEALTH], changed[ARMOR], changed[BULLETS] = 47, 73, 12
        assert not torch.equal(image[100:], renderer(changed)[100:])
        firing = state.clone()
        firing[FIRE_FLASH] = 0.1
        assert not torch.equal(image[40:100], renderer(firing)[40:100])
        dead, won = state.clone(), state.clone()
        dead[HEALTH], won[WON] = 0, 1
        assert not torch.equal(renderer(dead)[30:65], image[30:65])
        assert not torch.equal(renderer(won)[30:65], image[30:65])
        assert not torch.equal(renderer(dead)[30:65], renderer(won)[30:65])


def test_frozen_renderer_keeps_live_state_and_reads_only_final_pixel_rgb(assets):
    state = initial_state(assets)
    renderer = Renderer(assets, width=80, height=64).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(renderer, (state,), check_trace=False)
        frozen = torch.jit.freeze(traced)
        changed = state.clone()
        changed[ANGLE], changed[HEALTH], changed[WEAPON], changed[FIRE_FLASH] = 1.2, 28, 2, 0.1
        changed[renderer.layout.sector_ceiling][4] += 40
        assert torch.equal(frozen(changed), renderer(changed))
        graph = str(frozen.inlined_graph)
        assert "PythonOp" not in graph and "prim::If" not in graph and "prim::Loop" not in graph
        kinds = {node.kind() for node in frozen.inlined_graph.nodes()}
        assert not kinds & {"aten::topk", "aten::sort", "aten::kthvalue", "aten::min", "aten::max",
                            "aten::argmin", "aten::argmax"}
        assert {"aten::linear", "aten::relu"} <= kinds
        rgb_reads = [node for node in traced.inlined_graph.nodes()
                     if node.kind() == "aten::index"
                     and node.inputsAt(0).node().kind() == "prim::GetAttr"
                     and node.inputsAt(0).node().s("name") == "artwork_rgb"]
        assert len(rgb_reads) == 1
        assert rgb_reads[0].output().type().sizes() == [renderer.height, renderer.width, 3]
        address = rgb_reads[0].inputsAt(1).node().inputsAt(0)
        assert address.type().sizes() == [renderer.height, renderer.width]
        rgb_tables = []
        for node in frozen.inlined_graph.nodes():
            if node.kind() == "aten::remainder":
                modulus = node.inputsAt(1).toIValue()
                assert isinstance(modulus, (int, float))
            if node.kind() == "aten::index":
                table = node.inputsAt(0).toIValue()
                if isinstance(table, torch.Tensor) and table.dtype == torch.uint8 and table.ndim == 2:
                    rgb_tables.append(table)
        assert len(rgb_tables) == 1
        assert torch.equal(rgb_tables[0], renderer.artwork_rgb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No Torch GPU executor available")
def test_cpu_frozen_renderer_relocates_its_entire_graph_to_gpu(assets):
    state = initial_state(assets)
    renderer = Renderer(assets, width=80, height=64).eval()
    with torch.inference_mode():
        frozen = torch.jit.freeze(torch.jit.trace(renderer, (state,), check_trace=False))
        artifact = io.BytesIO()
        torch.jit.save(frozen, artifact)
        artifact.seek(0)
        gpu = torch.jit.load(artifact, map_location="cuda")
        actual = gpu(state.cuda()).cpu()
        expected = frozen(state)
        assert actual.shape == expected.shape
        assert (actual.to(torch.int16) - expected.to(torch.int16)).abs().max() <= 2
