"""Geometry blueprint consumed by the strict constructed-neural compiler.

Map-derived BSP and clearance policies already have fixed affine/activation
circuits. The tensor source also describes distances, openings, intersections,
and state-dependent indexing; the compiler must lower every deployed operation
to the permitted affine/ReLU/hard-threshold target IR.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .circuits import BooleanCircuit, MinimumCircuit
from .state import PROJECTILE_HEIGHT, PROJECTILE_RADIUS, sector_height_bounds


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class BSPSectorClassifier(nn.Module):
    """Map-derived partition neurons, leaf path neurons, and sector decoder."""

    def __init__(self, data):
        super().__init__()
        weights = data["node_weight"].float()
        children = data["node_children"].long()
        sectors = data["subsector_sector"].long()
        if weights.ndim != 2 or weights.shape[1] != 2 or len(weights) == 0:
            raise ValueError("the level requires a nonempty two-dimensional BSP")
        if children.shape != weights.shape:
            raise ValueError("BSP children must match the partition planes")
        incidence = torch.zeros(len(sectors), len(weights))
        bias = torch.ones(len(sectors))
        visited = set()
        leaves = set()

        def visit(node, path):
            if node < 0:
                leaf = -1 - node
                if leaf < 0 or leaf >= len(sectors) or leaf in leaves:
                    raise ValueError("BSP has an invalid or repeated leaf")
                leaves.add(leaf)
                for ancestor, branch in path:
                    incidence[leaf, ancestor] = 1 if branch else -1
                    bias[leaf] -= branch
                return
            if node >= len(children) or node in visited:
                raise ValueError("BSP has an invalid, cyclic, or repeated node")
            visited.add(node)
            visit(int(children[node, 0]), path + [(node, 0)])
            visit(int(children[node, 1]), path + [(node, 1)])

        visit(len(children) - 1, [])
        if len(leaves) != len(sectors) or len(visited) != len(children):
            raise ValueError("BSP must account for every partition and subsector")
        self.register_buffer("partition_weight", weights)
        self.register_buffer("partition_bias", data["node_bias"].float())
        # Doom's axis shortcuts place equality on the coordinate's <= side;
        # non-axis partitions place equality on the general back side.
        tie_front = ((weights[:, 0] == 0) & (weights[:, 1] > 0)) | (
            (weights[:, 1] == 0) & (weights[:, 0] > 0))
        self.register_buffer("partition_tie_back", ~tie_front)
        self.register_buffer("partition_origin", data["node_origin"].float())
        self.register_buffer("partition_northwest", (weights < 0).all(-1))
        self.register_buffer("path_weight", incidence)
        self.register_buffer("path_bias", bias)
        self.register_buffer("sector_decoder", sectors.float().reshape(1, -1))

    def forward(self, points):
        plane = F.linear(points, self.partition_weight, self.partition_bias)
        at_origin = (points[:, None, :] == self.partition_origin).all(-1)
        tie_back = self.partition_tie_back & ~(at_origin & self.partition_northwest)
        partition = ((plane > 0) | ((plane == 0) & tie_back)).float()
        leaves = F.relu(F.linear(partition, self.path_weight, self.path_bias))
        return F.linear(leaves, self.sector_decoder).squeeze(-1).long()

    def construction_spec(self):
        return {"type": "BSPSectorClassifier", "input_count": 2,
                "partitions": len(self.partition_weight), "leaf_paths": len(self.path_weight),
                "affine_layers": 3, "activations": ["hard_threshold", "relu"],
                "connectivity": "BSP ancestor/branch incidence and subsector sector IDs",
                "boundary_support": "exact coordinate equality selects Doom axis/node-origin threshold ties"}


class LineClearCircuit(nn.Module):
    """Per-line DNF blockers followed by one fixed all-lines-clear neuron."""

    def __init__(self, line_count, predicate_names, rules):
        super().__init__()
        self.predicate_names = tuple(predicate_names)
        self.blocking = BooleanCircuit(len(self.predicate_names), [rules])
        self.register_buffer("clear_weight", -torch.ones(1, line_count))
        self.register_buffer("clear_bias", torch.tensor([0.5]))

    def forward(self, predicates):
        blocked = self.blocking(predicates).squeeze(-1)
        return (F.linear(blocked, self.clear_weight, self.clear_bias) > 0).squeeze(-1)

    def construction_spec(self):
        return {"type": "LineClearCircuit", "line_count": self.clear_weight.shape[1],
                "line_predicates": list(self.predicate_names),
                "affine_layers": 1, "activation": "hard_threshold",
                "connectivity": "one weight -1 per line block, bias 0.5; child compiles DNF line blocks"}


class Geometry(nn.Module):
    """Map geometry source and constructed decisions for strict neural lowering."""

    def __init__(self, data):
        super().__init__()
        self.sector_classifier = BSPSectorClassifier(data)
        vertices = data["vertices"].float()
        lines = data["lines"].long()
        starts, ends = vertices[lines[:, 0]], vertices[lines[:, 1]]
        line_delta = ends - starts
        line_square = (line_delta * line_delta).sum(-1)
        if not bool((line_square > 0).all()):
            raise ValueError("linedefs must have nonzero length")
        self.register_buffer("line_start", starts)
        self.register_buffer("line_delta", line_delta)
        self.register_buffer("line_square", line_square)
        front = data["line_front_sector"].long()
        back = data["line_back_sector"].long()
        if not bool((front >= 0).all()):
            raise ValueError("linedefs require a front sector")
        self.register_buffer("front", front)
        self.register_buffer("back", back.clamp_min(0))
        self.register_buffer("one_sided", back < 0)
        self.register_buffer("impassable", (data["line_flags"].long() & 1) != 0)
        floor_low, floor_high, ceiling_high = sector_height_bounds(data)
        lower, upper = float(floor_low), max(float(floor_high), float(ceiling_high))
        self.support_lower, self.support_upper = lower, upper
        self.support_minimum = MinimumCircuit(len(lines) + 1, bound=upper - lower)
        self.collision_decision = LineClearCircuit(
            len(lines), ("overlap", "one_sided", "impassable"), [(1, 2), (1, 3)])
        self.visibility_decision = LineClearCircuit(
            len(lines), ("ray_hit", "before_target", "one_sided", "at_or_below_floor", "at_or_above_ceiling"),
            [(1, 2, 3), (1, 2, 4), (1, 2, 5)])
        self.projectile_endpoint_decision = LineClearCircuit(
            len(lines), ("overlap", "one_sided", "at_or_below_floor", "above_ceiling"),
            [(1, 2), (1, 3), (1, 4)])
        self.projectile_decision = BooleanCircuit(4, [[(1, 2, 3, 4)]])

    def sector_at(self, points):
        return self.sector_classifier(points)

    def distance_square(self, points):
        relative = points[:, None, :] - self.line_start[None, :, :]
        projection = ((relative * self.line_delta).sum(-1)
                      / self.line_square).clamp(0, 1)
        closest = self.line_start + projection[..., None] * self.line_delta
        difference = points[:, None, :] - closest
        return (difference * difference).sum(-1)

    def openings(self, floor, ceiling):
        return (torch.maximum(floor[self.front], floor[self.back]),
                torch.minimum(ceiling[self.front], ceiling[self.back]))

    def body_opening(self, points, floor, ceiling, radius):
        """Highest support and lowest ceiling contacted by each circular body.

        A center crossing a descending portal retains the higher floor until
        its radius clears the edge. Exact tangency has no overlap, matching
        collision clearance. Center-sector bounds apply even without contacts.
        """
        sector = self.sector_at(points)
        contact = (self.distance_square(points) < radius[:, None].square()) & ~self.one_sided
        bottom, top = self.openings(floor, ceiling)
        floors = torch.cat((floor[sector, None],
                            torch.where(contact, bottom[None, :], floor[sector, None])), -1)
        ceilings = torch.cat((ceiling[sector, None],
                              torch.where(contact, top[None, :], ceiling[sector, None])), -1)
        inverted, _ = self.support_minimum(self.support_upper - floors)
        clearance, _ = self.support_minimum(ceilings - self.support_lower)
        return self.support_upper - inverted, clearance + self.support_lower

    def can_move(self, points, feet, floor, ceiling, radius, height=56.0):
        support, clearance = self.body_opening(points, floor, ceiling, radius)
        overlap = self.distance_square(points) < radius[:, None].square()
        predicates = torch.stack((
            overlap, self.one_sided.expand_as(overlap), self.impassable.expand_as(overlap)), dim=-1)
        return (self.collision_decision(predicates.float()) & (support <= feet + 24.0)
                & (clearance >= torch.maximum(support, feet) + height))

    def ray_intersections(self, origins, directions, include_origin=False):
        relative = self.line_start[None, :, :] - origins[:, None, :]
        denominator = cross(directions[:, None, :], self.line_delta)
        parallel = denominator.abs() < 1e-7
        safe_denominator = torch.where(parallel, torch.ones_like(denominator),
                                       denominator)
        ray = cross(relative, self.line_delta) / safe_denominator
        segment = cross(relative, directions[:, None, :]) / safe_denominator
        forward = ray >= 0 if include_origin else ray > 1e-4
        hit = ~parallel & forward & (segment >= 0) & (segment <= 1)
        return ray, hit

    def visible(self, origins, targets, origin_z, target_z, floor, ceiling, include_endpoint=False):
        """Paired LOS segments, including interpolated height through portals."""
        ray, hit = self.ray_intersections(origins, targets - origins, include_origin=include_endpoint)
        bottom, top = self.openings(floor, ceiling)
        before_target = ray <= 1 if include_endpoint else ray < 0.9999
        # Height contributes only at intersections on this segment. Bound its
        # interpolation factor before lowering variable multiplication.
        fraction = torch.where(hit & before_target, ray, torch.zeros_like(ray))
        height = origin_z[:, None] + fraction * (target_z - origin_z)[:, None]
        predicates = torch.stack((
            hit, before_target, self.one_sided.expand_as(hit),
            height <= bottom, height >= top), dim=-1)
        return self.visibility_decision(predicates.float())

    def projectile_clear(self, origins, targets, origin_z, target_z, floor, ceiling):
        """Swept contact and radius-6, height-8 fireball endpoint clearance."""
        clear = self.visible(origins, targets, origin_z, target_z, floor, ceiling,
                             include_endpoint=True)
        bottom, top = self.openings(floor, ceiling)
        overlap = self.distance_square(targets) < PROJECTILE_RADIUS ** 2
        # Z is the missile's bottom. P_ZMovement expires at z<=floor or
        # z+height>ceiling; ceiling equality fits the original missile body.
        # https://github.com/id-Software/DOOM/blob/master/linuxdoom-1.10/p_mobj.c#L269-L329
        endpoint = self.projectile_endpoint_decision(torch.stack((
            overlap, self.one_sided.expand_as(overlap),
            target_z[:, None] <= bottom, target_z[:, None] + PROJECTILE_HEIGHT > top), -1).float())
        sector = self.sector_at(targets)
        predicates = torch.stack((clear, endpoint, target_z > floor[sector],
                                  target_z + PROJECTILE_HEIGHT <= ceiling[sector]), -1)
        return self.projectile_decision(predicates.float()).squeeze(-1) > 0
