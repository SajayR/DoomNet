"""Constructed geometry policies checked against ordinary scalar geometry."""

from itertools import product

import numpy as np
import pytest
import torch

from doomnet.geometry import BSPSectorClassifier, Geometry


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def map_data(back=1, flags=0):
    return {
        "vertices": torch.tensor([[0., -64.], [0., 64.]]),
        "lines": torch.tensor([[0, 1]]),
        "line_front_sector": torch.tensor([0]),
        "line_back_sector": torch.tensor([back]),
        "line_flags": torch.tensor([flags]),
        "node_weight": torch.tensor([[1., 0.]]),
        "node_bias": torch.tensor([0.]),
        "node_origin": torch.tensor([[0., 0.]]),
        "node_children": torch.tensor([[-1, -2]]),
        "subsector_sector": torch.tensor([0, 1]),
        "sector_floor": torch.zeros(2),
        "sector_ceiling": torch.full((2,), 128.),
    }


def reference_move(data, points, feet, floor, ceiling, radius, height):
    vertices = data["vertices"].numpy()
    result = []
    for point, foot, size in zip(points.numpy(), feet.numpy(), radius.numpy()):
        clear = True
        # This fixture's native BSP places x <= 0 in sector0, otherwise sector1.
        center = int(point[0] > 0)
        support, clearance = float(floor[center]), float(ceiling[center])
        for line, front, back, flag in zip(data["lines"].tolist(),
                                         data["line_front_sector"].tolist(),
                                         data["line_back_sector"].tolist(),
                                         data["line_flags"].tolist()):
            start, end = vertices[line]
            delta = end - start
            fraction = np.clip(np.dot(point - start, delta) / np.dot(delta, delta), 0, 1)
            difference = point - (start + fraction * delta)
            if np.dot(difference, difference) >= size * size:
                continue
            bottom = max(float(floor[front]), float(floor[max(back, 0)]))
            top = min(float(ceiling[front]), float(ceiling[max(back, 0)]))
            if back < 0 or flag & 1:
                clear = False
                break
            support, clearance = max(support, bottom), min(clearance, top)
        result.append(clear and support <= foot + 24 and clearance >= max(support, foot) + height)
    return torch.tensor(result, dtype=torch.bool)


def reference_visible(data, origins, targets, origin_z, target_z, floor, ceiling):
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    vertices = data["vertices"].numpy()
    result = []
    for origin, target, oz, tz in zip(origins.numpy(), targets.numpy(), origin_z.numpy(), target_z.numpy()):
        clear = True
        for line, front, back in zip(data["lines"].tolist(),
                                    data["line_front_sector"].tolist(),
                                    data["line_back_sector"].tolist()):
            start, end = vertices[line]
            relative, direction, delta = start - origin, target - origin, end - start
            denominator = cross(direction, delta)
            if abs(denominator) < np.float32(1e-7):
                continue
            ray = cross(relative, delta) / denominator
            segment = cross(relative, direction) / denominator
            if not (np.float32(1e-4) < ray < np.float32(0.9999) and 0 <= segment <= 1):
                continue
            height = oz + ray * (tz - oz)
            bottom = max(float(floor[front]), float(floor[max(back, 0)]))
            top = min(float(ceiling[front]), float(ceiling[max(back, 0)]))
            if back < 0 or height <= bottom or height >= top:
                clear = False
                break
        result.append(clear)
    return torch.tensor(result, dtype=torch.bool)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("back,flags,floors,ceilings,feet,height", [
    (-1, 0, (0, 0), (128, 128), 0, 56),
    (1, 1, (0, 0), (128, 128), 0, 56),
    (1, 0, (0, 0), (128, 128), 0, 56),
    (1, 0, (0, 24), (128, 80), 0, 56),
    (1, 0, (0, 25), (128, 128), 0, 56),
    (1, 0, (0, 28), (128, 84), 4, 56),
    (1, 0, (0, 24), (128, 79), 0, 56),
    (1, 0, (0, 0), (128, 88), 32, 56),
    (1, 0, (0, 0), (128, 87), 32, 56),
    (1, 0, (0, 0), (128, 8), 0, 8),
])
def test_collision_matches_scalar_segment_reference(device, back, flags, floors, ceilings, feet, height):
    data = map_data(back, flags)
    geometry = Geometry(data).to(device)
    points = torch.tensor([[-16., 0.], [-15.5, 0.], [0., 0.], [16., 0.], [16.5, 0.],
                           [0., 80.], [0., 79.5], [0., 64.], [8., 70.], [8., 80.]])
    foot = torch.full((len(points),), float(feet))
    radius = torch.full((len(points),), 16.)
    floor, ceiling = torch.tensor(floors).float(), torch.tensor(ceilings).float()
    expected = reference_move(data, points, foot, floor, ceiling, radius, height)
    actual = geometry.can_move(points.to(device), foot.to(device), floor.to(device),
                               ceiling.to(device), radius.to(device), height)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("back,flags", [(-1, 0), (1, 0), (1, 1)])
def test_visibility_matches_scalar_portal_reference(device, back, flags):
    data = map_data(back, flags)
    geometry = Geometry(data).to(device)
    # Crossings at height equality, varying slopes, wall endpoints, parallel
    # rays, and each end of the paired LOS segment exercise distinct policies.
    origins = torch.tensor([[-32., 0.]] * 9 + [[-32., 64.], [-32., 65.], [-32., 0.],
                                                [0., 0.], [-32., 0.], [0., -80.]])
    targets = torch.tensor([[32., 0.]] * 9 + [[32., 64.], [32., 65.], [0., 0.],
                                              [32., 0.], [-16., 0.], [0., 80.]])
    origin_z = torch.tensor([24., 24.25, 79.75, 80., 23., 81., 0., 0., 20.,
                             40., 40., 40., 40., 40., 40.])
    target_z = torch.tensor([24., 24.25, 79.75, 80., 23., 81., 80., 48., 140.,
                             40., 40., 40., 40., 40., 40.])
    floor, ceiling = torch.tensor([0., 24.]), torch.tensor([128., 80.])
    expected = reference_visible(data, origins, targets, origin_z, target_z, floor, ceiling)
    actual = geometry.visible(*(value.to(device) for value in
                               (origins, targets, origin_z, target_z, floor, ceiling)))
    assert torch.equal(actual.cpu(), expected)
    assert actual[10:].all()  # outside wall segment, target/origin equality, no crossing, parallel
    if back == 1:
        assert expected[:4].tolist() == [False, True, True, False]


@pytest.mark.parametrize("device", DEVICES)
def test_visibility_preserves_strict_ray_fraction_cutoffs(device):
    data = map_data(back=-1)
    geometry = Geometry(data).to(device)
    origins = torch.tensor([[-9999., 0.], [-9998., 0.], [-19999., 0.],
                            [-1., 0.], [-2., 0.]])
    targets = torch.tensor([[1., 0.], [2., 0.], [1., 0.], [9999., 0.], [9998., 0.]])
    height = torch.full((5,), 40.)
    floor, ceiling = torch.zeros(2), torch.full((2,), 128.)
    expected = reference_visible(data, origins, targets, height, height, floor, ceiling)
    assert expected.tolist() == [True, False, True, True, False]
    actual = geometry.visible(*(value.to(device) for value in
                               (origins, targets, height, height, floor, ceiling)))
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_map_line_reduction_handles_mixed_batches_and_empty_batches(device):
    data = map_data()
    data.update({"vertices": torch.tensor([[0., -64.], [0., 64.], [64., -64.], [64., 64.],
                                            [128., -64.], [128., 64.]]),
                 "lines": torch.tensor([[0, 1], [2, 3], [4, 5]]),
                 "line_front_sector": torch.tensor([0, 0, 1]),
                 "line_back_sector": torch.tensor([-1, 1, 2]),
                 "line_flags": torch.tensor([0, 0, 1]),
                 "sector_floor": torch.tensor([0., 24., 25.]),
                 "sector_ceiling": torch.tensor([128., 80., 81.])})
    geometry = Geometry(data).to(device)
    floor, ceiling = torch.tensor([0., 24., 25.]), torch.tensor([128., 80., 81.])
    points = torch.tensor([[0., 0.], [16., 0.], [64., 0.], [128., 0.], [64., 80.]])
    feet, radius = torch.tensor([0., 0., 0., 25., 32.]), torch.tensor([16., 16., 16., 16., 0.])
    expected = reference_move(data, points, feet, floor, ceiling, radius, 56)
    actual = geometry.can_move(*(value.to(device) for value in (points, feet, floor, ceiling, radius)))
    assert actual.cpu().tolist() == expected.tolist() == [False, True, True, False, False]
    origins, targets = points[:4], points[:4].flip(0)
    origin_z, target_z = torch.full((4,), 40.), torch.full((4,), 40.)
    expected = reference_visible(data, origins, targets, origin_z, target_z, floor, ceiling)
    assert torch.equal(geometry.visible(*(value.to(device) for value in
                                        (origins, targets, origin_z, target_z, floor, ceiling))).cpu(), expected)
    assert geometry.can_move(points[:0].to(device), feet[:0].to(device), floor.to(device),
                             ceiling.to(device), radius[:0].to(device)).shape == (0,)
    assert geometry.visible(origins[:0].to(device), targets[:0].to(device), origin_z[:0].to(device),
                            target_z[:0].to(device), floor.to(device), ceiling.to(device)).shape == (0,)


@pytest.mark.parametrize("name", ["collision_decision", "visibility_decision"])
def test_policies_execute_affine_threshold_layers_without_boolean_reductions(name):
    geometry = Geometry(map_data())
    module = getattr(geometry, name).eval()
    count = len(module.construction_spec()["line_predicates"])
    inputs = torch.tensor(list(product((0., 1.), repeat=count))).reshape(2 ** count, 1, count)
    if name == "collision_decision":
        expected = ~(inputs[..., 0].bool() & inputs[..., 1:].bool().any(-1)).any(-1)
    else:
        expected = ~(inputs[..., 0].bool() & inputs[..., 1].bool() & inputs[..., 2:].bool().any(-1)).any(-1)
    frozen = torch.jit.freeze(torch.jit.trace(module, inputs))
    assert torch.equal(module(inputs), expected)
    assert torch.equal(frozen(inputs), expected)
    operators = [node.kind() for node in frozen.inlined_graph.nodes()]
    assert operators.count("aten::linear") == 3
    assert operators.count("aten::gt") == 3
    assert not {"aten::any", "aten::all", "aten::min", "aten::amin", "aten::topk"}.intersection(operators)
    assert not hasattr(geometry, "construction_spec")
    assert module.construction_spec()["line_predicates"][0] in {"overlap", "ray_hit"}


@pytest.mark.parametrize("dx,dy", [(0, 8), (0, -8), (8, 0), (-8, 0),
                                    (8, 8), (-8, 8), (8, -8), (-8, -8)])
def test_extracted_bsp_preserves_native_partition_ties_and_origin(dx, dy):
    data = map_data()
    nx, ny = 32, -16
    data["node_origin"] = torch.tensor([[nx, ny]]).float()
    data["node_weight"] = torch.tensor([[-dy, dx]]).float()
    data["node_bias"] = torch.tensor([dy * nx - dx * ny]).float()
    points = [(nx + x, ny + y) for x, y in product(range(-2, 3), repeat=2)]
    points += [(nx + dx, ny + dy), (nx - dx, ny - dy)]

    def native_side(x, y):
        if dx == 0:
            return dy > 0 if x <= nx else dy < 0
        if dy == 0:
            return dx < 0 if y <= ny else dx > 0
        if (dy ^ dx ^ (x - nx) ^ (y - ny)) < 0:
            return (dy ^ (x - nx)) < 0
        return (y - ny) * dx >= (x - nx) * dy

    geometry = Geometry(data)
    classifier = geometry.sector_classifier
    assert isinstance(classifier, BSPSectorClassifier)
    expected = torch.tensor([native_side(x, y) for x, y in points]).long()
    inputs = torch.tensor(points).float()
    assert torch.equal(geometry.sector_at(inputs), expected)
    frozen = torch.jit.freeze(torch.jit.trace(classifier.eval(), inputs))
    assert torch.equal(frozen(inputs), expected)
    operators = [node.kind() for node in frozen.inlined_graph.nodes()]
    assert operators.count("aten::linear") == 3 and "aten::relu" in operators
    assert classifier.construction_spec()["leaf_paths"] == 2
