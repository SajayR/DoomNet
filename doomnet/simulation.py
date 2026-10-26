"""Recurrent E1M1 blueprint consumed by the strict constructed-neural compiler.

Body clearance, door occupancy/transitions, basic action eligibility, and
nearest use/pellet selection already have fixed affine/activation circuits.
Geometry has its own constructed classification and blocking circuits.
The tensor source specifies integration, timers/RNG, circle intersections,
resources, state routing, and enemy/pickup policies. Construction lowers every
deployed operation into the permitted affine/ReLU/hard-threshold neural IR;
this source module is not itself the strict deployed artifact.
All mutable values are recurrent state; forward neither reads tensors in
Python nor calls an engine. Fixed weapon/pickup tables remain affine data maps.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import state as S
from .circuits import BooleanCircuit, MinimumCircuit
from .geometry import Geometry, cross
from .runtime import MAX_DT


class BodyClearCircuit(nn.Module):
    """Compile overlap, solidity, and ownership into an all-bodies-clear neuron."""

    def __init__(self, body_count):
        super().__init__()
        self.blocking = BooleanCircuit(3, [[(1, 2, 3)]])
        self.register_buffer("clear_weight", -torch.ones(1, body_count))
        self.register_buffer("clear_bias", torch.tensor([.5]))

    def forward(self, predicates):
        blocked = self.blocking(predicates).squeeze(-1)
        return (F.linear(blocked, self.clear_weight, self.clear_bias) > 0).squeeze(-1)

    def construction_spec(self):
        return {"type": "BodyClearCircuit", "body_count": self.clear_weight.shape[1],
                "body_predicates": ["overlap", "solid", "different_actor"],
                "affine_layers": 1, "activation": "hard_threshold",
                "connectivity": "weight -1 per DNF body block, clear bias 0.5"}


class DoorOccupancyCircuit(nn.Module):
    """Boundary decoder, live-body DNF, and a fixed any-actor neuron per door."""

    def __init__(self, boundary_weight, actor_count):
        super().__init__()
        self.register_buffer("boundary_weight", boundary_weight)
        self.live_contact = BooleanCircuit(3, [[(1, 3), (2, 3)]])
        self.register_buffer("occupied_weight", torch.ones(1, actor_count))
        self.register_buffer("occupied_bias", torch.tensor([-.5]))

    def forward(self, boundary_contact, center_inside, actor_alive):
        touches = (F.linear(boundary_contact, self.boundary_weight) > 0).float()
        predicates = torch.stack((center_inside, touches,
                                   actor_alive[:, None].expand_as(center_inside)), -1)
        live = self.live_contact(predicates).squeeze(-1)
        return (F.linear(live.transpose(0, 1), self.occupied_weight,
                         self.occupied_bias) > 0).squeeze(-1)

    def construction_spec(self):
        return {"type": "DoorOccupancyCircuit", "doors": self.boundary_weight.shape[0],
                "boundary_lines": self.boundary_weight.shape[1],
                "actor_count": self.occupied_weight.shape[1],
                "actor_predicates": ["center_inside", "boundary_contact", "alive"],
                "affine_layers": 2, "activation": "hard_threshold",
                "connectivity": "map boundary-to-door incidence; unit connection per live contact"}


class Simulation(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.layout = S.StateLayout.from_data(data)
        self.geometry = Geometry(data)
        self.body_clear = BodyClearCircuit(self.layout.entity_count)
        self.step_clear = BooleanCircuit(2, [[(1, 2)]])
        self.play_rule = BooleanCircuit(3, [[(1, 2, 3)]])
        self.use_edge_rule = BooleanCircuit(3, [[(1, -2, 3)]])
        self.fire_rule = BooleanCircuit(3, [[(1, 2, 3)]])
        self.use_stop_rule = BooleanCircuit(6, [[(1, 5, 6), (2, 5, 6),
                                                (3, 5, 6), (4, 5, 6)]])
        self.use_minimum = MinimumCircuit(self.layout.line_count, bound=1e6)
        self.use_action_rule = BooleanCircuit(4, [[(1, 2, 3, 4)]])
        self.pellet_hit_rule = BooleanCircuit(7, [[(1, 2, 3, 4, 6, 7),
                                                  (1, 2, 3, 5, 6, 7)]])
        self.pellet_minimum = MinimumCircuit(self.layout.entity_count, bound=1e6)
        self.pellet_action_rule = BooleanCircuit(5, [[(1, 2, 3, 5), (1, 2, 4, 5)]])
        self.door_motion_rule = BooleanCircuit(5, [[(2, 3, -5)],
                                                   [(1, 4, -5), (2, 3, 4, -5)]])
        self.door_finish_rule = BooleanCircuit(4, [[(1, 2)], [(3, 4)]])
        self.register_buffer("spawn", S.initial_state(data))
        thing = data["entity_type"].long()
        enemy = S.moving_actors(thing)
        barrel = thing == 2035
        self.register_buffer("thing", thing)
        self.register_buffer("enemy", enemy)
        self.register_buffer("barrel", barrel)
        self.register_buffer("enemy_index", torch.where(enemy)[0])
        self.register_buffer("barrel_index", torch.where(barrel)[0])
        self.register_buffer("shootable_index", torch.where(enemy | barrel)[0])
        self.register_buffer("drop_index", torch.where((thing == 3004) | (thing == 9))[0])
        self.barrel_count = int(barrel.sum())
        self.register_buffer("entity_index", torch.arange(len(thing)))
        self.register_buffer("line_index", torch.arange(self.layout.line_count))
        self.register_buffer("sector_index", torch.arange(self.layout.sector_count))
        self.register_buffer("player_index", torch.tensor([-1]))
        self.register_buffer("player_radius", torch.tensor([S.PLAYER_RADIUS]))
        blocking_props = (35, 48, 2028)
        prop = torch.zeros(len(thing), dtype=torch.bool)
        for kind in blocking_props:
            prop |= thing == kind
        self.register_buffer("solid_kind", enemy | barrel | prop)
        self.register_buffer("radius", torch.where(enemy, S.ENEMY_RADIUS,
                                                   torch.where(barrel, 10.0, 16.0)))
        self.register_buffer("ambush", (data["entity_flags"].long() & 8) != 0)
        self.register_buffer("enemy_speed", torch.where(thing == 3001, 70.0, 45.0))
        self.register_buffer("enemy_period", torch.where(thing == 3001, 2.0, 1.5))
        self.register_buffer("pellet_index", torch.arange(7, dtype=torch.float32))
        self.register_buffer("pellet_offset", torch.tensor([0, -.09, -.06, -.03,
                                                            .03, .06, .09]))
        # Outputs: cooldown, range, bullet consumption, shell consumption.
        self.register_buffer("weapon_weight", torch.tensor([
            [.4, .28, .9], [64., 2048., 2048.], [0., 1., 0.], [0., 0., 1.]
        ]))
        self.register_buffer("weapon_codes", torch.arange(3))
        # Each column is an entity; a collected-entity activation supplies its
        # health, armor-bonus, bullet, and shell contributions through an affine.
        effects = torch.zeros(4, len(thing))
        for kind, vector in {
            2011: (10, 0, 0, 0), 2012: (25, 0, 0, 0),
            2014: (1, 0, 0, 0),
            2015: (0, 1, 0, 0), 2007: (0, 0, 10, 0),
            2048: (0, 0, 50, 0), 2008: (0, 0, 0, 4),
            2049: (0, 0, 0, 20), 2001: (0, 0, 0, 8),
        }.items():
            effects[:, thing == kind] = torch.tensor(vector).float()[:, None]
        self.register_buffer("pickup_weight", effects)
        special = data["line_special"].long()
        allowed = torch.tensor([0, 1, 11, 36, 48, 88])
        if not bool((special[:, None] == allowed).any(-1).all()):
            raise ValueError("E1M1 simulation received an unsupported linedef special")
        self.register_buffer("line_special", special)
        self.register_buffer("sector_special", data["sector_special"].long())
        self.register_buffer("original_floor", data["sector_floor"].float())
        door_weight = torch.zeros(self.layout.sector_count, len(special))
        floor_weight = torch.zeros_like(door_weight)
        lift = torch.zeros(self.layout.sector_count, dtype=torch.bool)
        turbo = torch.zeros_like(lift)
        adjacency = [set() for _ in range(self.layout.sector_count)]
        front = data["line_front_sector"].tolist()
        back = data["line_back_sector"].tolist()
        for a, b in zip(front, back):
            if b >= 0 and a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)
        for line, kind in enumerate(special.tolist()):
            if kind == 1:
                if back[line] < 0:
                    raise ValueError("a manual door requires a back sector")
                door_weight[back[line], line] = 1
            if kind in (36, 88):
                tag = data["line_tag"][line]
                targets = (data["sector_tag"] == tag) & (tag != 0)
                if not bool(targets.any()):
                    raise ValueError("a floor trigger must name at least one sector")
                floor_weight[targets, line] = 1
                if kind == 88:
                    lift |= targets
                else:
                    turbo |= targets
        door = door_weight.any(-1)
        door_index = torch.where(door)[0]
        front_sector = data["line_front_sector"]
        back_sector = data["line_back_sector"]
        door_boundary = ((front_sector[None, :] == door_index[:, None])
                         | (back_sector[None, :] == door_index[:, None]))
        door_boundary &= (front_sector != back_sector)[None, :]
        door_target = data["sector_ceiling"].clone().float()
        floor_target = data["sector_floor"].clone().float()
        for sector, neighbors in enumerate(adjacency):
            if bool(door[sector] | lift[sector] | turbo[sector]):
                if not neighbors:
                    raise ValueError("a moving sector requires an adjacent sector")
                indices = torch.tensor(sorted(neighbors))
                if bool(door[sector]):
                    door_target[sector] = data["sector_ceiling"][indices].min() - 4
                if bool(lift[sector]):
                    floor_target[sector] = torch.minimum(
                        data["sector_floor"][indices].min(), floor_target[sector])
                if bool(turbo[sector]):
                    highest = data["sector_floor"][indices].max()
                    floor_target[sector] = highest + 8 * (
                        highest != data["sector_floor"][sector])
        self.register_buffer("door_weight", door_weight)
        self.register_buffer("door_index", door_index)
        self.door_occupancy = DoorOccupancyCircuit(door_boundary.float(), self.layout.entity_count + 1)
        self.register_buffer("floor_weight", floor_weight)
        self.register_buffer("door_target", door_target)
        self.register_buffer("floor_target", floor_target)
        self.register_buffer("lift", lift)
        self.register_buffer("turbo", turbo)

    def _body_clear(self, candidate, radius, positions, solid, own_index):
        distance = (candidate[:, None, :] - positions[None, :, :]).square().sum(-1)
        overlap = distance < (radius[:, None] + self.radius[None, :]).square()
        other = self.entity_index[None, :] != own_index[:, None]
        return self.body_clear(torch.stack((overlap, solid[None, :].expand_as(overlap), other), -1).float())

    def _slide(self, positions, delta, feet, floor, ceiling, radius,
               bodies, solid, own_index):
        x_candidate = torch.stack((positions[:, 0] + delta[:, 0],
                                   positions[:, 1]), -1)
        x_clear = self.step_clear(torch.stack((
            self.geometry.can_move(x_candidate, feet, floor, ceiling, radius),
            self._body_clear(x_candidate, radius, bodies, solid, own_index)), -1).float()).squeeze(-1) > 0
        after_x = torch.where(x_clear[:, None], x_candidate, positions)
        after_x_feet, _ = self.geometry.body_opening(after_x, floor, ceiling, radius)
        y_candidate = torch.stack((after_x[:, 0], after_x[:, 1] + delta[:, 1]), -1)
        y_clear = self.step_clear(torch.stack((
            self.geometry.can_move(y_candidate, after_x_feet, floor, ceiling, radius),
            self._body_clear(y_candidate, radius, bodies, solid, own_index)), -1).float()).squeeze(-1) > 0
        return torch.where(y_clear[:, None], y_candidate, after_x)

    def _use(self, position, facing, floor, ceiling, edge):
        distance, hit = self.geometry.ray_intersections(position[None, :],
                                                       facing[None, :])
        bottom, top = self.geometry.openings(floor, ceiling)
        candidates = self.use_stop_rule(torch.stack((
            self.line_special == 1, self.line_special == 11,
            self.geometry.one_sided, top - bottom < 56,
            hit[0], distance[0] <= 64), -1).float()).squeeze(-1) > 0
        score = torch.where(candidates, distance[0], 1e6)
        nearest, chosen = self.use_minimum(score)
        front_side = cross(self.geometry.line_delta,
                           position - self.geometry.line_start) <= 0
        return self.use_action_rule(torch.stack((
            self.line_index == chosen, (nearest <= 64).expand_as(front_side),
            edge.expand_as(front_side), front_side), -1).float()).squeeze(-1)

    def _sectors(self, state, dt, use, crossed, position, entities, alive):
        layout = self.layout
        floor = state[layout.sector_floor]
        ceiling = state[layout.sector_ceiling]
        motion = state[layout.sector_motion]
        timer = F.relu(state[layout.sector_timer] - dt)
        trigger = F.linear(use, self.door_weight) > 0
        # A body straddling a doorway also occupies it when its center remains
        # in the neighboring sector. Segment distance includes corner contact.
        actors = torch.cat((position[None, :], entities))
        actor_sector = self.geometry.sector_at(actors)
        actor_alive = torch.cat(((state[S.HEALTH] > 0).reshape(1), alive > 0))
        actor_radius = torch.cat((self.player_radius, self.radius))
        boundary_contact = self.geometry.distance_square(actors) < actor_radius[:, None].square()
        center_inside = actor_sector[:, None] == self.door_index[None, :]
        door_occupied = self.door_occupancy(boundary_contact.float(), center_inside.float(),
                                           actor_alive.float())
        occupied = torch.zeros_like(floor).index_copy(0, self.door_index,
                                                     door_occupied.float()) > 0
        commands = self.door_motion_rule(torch.stack((
            motion < 0, motion == 2, timer <= 0, occupied, trigger), -1).float()) > 0
        motion = torch.where(trigger, torch.ones_like(motion), motion)
        motion = torch.where(commands[:, 0], -torch.ones_like(motion), motion)
        motion = torch.where(commands[:, 1], torch.ones_like(motion), motion)
        step = torch.where(motion == 1, 70 * dt,
                           torch.where(motion == -1, -70 * dt, torch.zeros_like(motion)))
        ceiling = torch.maximum(floor, torch.minimum(self.door_target, ceiling + step))
        finished = self.door_finish_rule(torch.stack((
            motion == 1, ceiling >= self.door_target, motion == -1, ceiling <= floor), -1).float()) > 0
        opened, closed = finished[:, 0], finished[:, 1]
        motion = torch.where(opened, torch.full_like(motion, 2),
                             torch.where(closed, torch.zeros_like(motion), motion))
        timer = torch.where(opened, torch.full_like(timer, 4), timer)
        floor_motion = state[layout.sector_floor_motion]
        floor_timer = F.relu(state[layout.sector_floor_timer] - dt)
        floor_trigger = F.linear(crossed, self.floor_weight) > 0
        floor_motion = torch.where(floor_trigger & (floor_motion == 0),
                                   -torch.ones_like(floor_motion), floor_motion)
        floor_motion = torch.where((floor_motion == 2) & (floor_timer <= 0),
                                   torch.ones_like(floor_motion), floor_motion)
        floor_step = torch.where(floor_motion == -1, -140 * dt,
                                 torch.where(floor_motion == 1, 140 * dt,
                                             torch.zeros_like(floor_motion)))
        floor = torch.maximum(self.floor_target, torch.minimum(self.original_floor,
                                                               floor + floor_step))
        lowered = (floor_motion == -1) & (floor <= self.floor_target)
        raised = (floor_motion == 1) & (floor >= self.original_floor)
        floor_motion = torch.where(lowered & self.lift, torch.full_like(floor_motion, 2),
                                   torch.where((lowered & self.turbo) | raised,
                                               torch.zeros_like(floor_motion), floor_motion))
        floor_timer = torch.where(lowered & self.lift, torch.full_like(floor_timer, 3),
                                  floor_timer)
        return floor, ceiling, motion, timer, floor_motion, floor_timer

    def forward(self, state, controls, dt):
        layout = self.layout
        dt = dt.reshape(()).clamp(0, MAX_DT)
        playing = self.play_rule(torch.stack((
            state[S.HEALTH] > 0, state[S.WON] < .5, dt > 0)).float()).squeeze(-1) > 0
        elapsed = dt * playing.float()
        position = state[S.X:S.Y + 1]
        entities = torch.stack((state[layout.entity_x], state[layout.entity_y]), -1)
        hp = state[layout.entity_hp]
        alive = state[layout.entity_alive]
        solid = self.solid_kind & (alive > .5)
        floor, ceiling = state[layout.sector_floor], state[layout.sector_ceiling]
        angle = torch.remainder(state[S.ANGLE] + controls[2].clamp(-1, 1) * 2.8 * elapsed,
                                2 * math.pi)
        facing = torch.stack((torch.cos(angle), torch.sin(angle)))
        right = torch.stack((facing[1], -facing[0]))
        movement = controls[0].clamp(-1, 1) * facing + controls[1].clamp(-1, 1) * right
        movement = movement / torch.maximum(movement.square().sum().sqrt(),
                                             torch.ones_like(elapsed))
        speed = 180 + 120 * (controls[7] > .5).float()
        moved = self._slide(position[None, :], (movement * speed * elapsed)[None, :],
                            state[S.Z].reshape(1), floor, ceiling,
                            self.player_radius, entities,
                            solid, self.player_index)[0]
        use_edge = self.use_edge_rule(torch.stack((
            controls[4] > .5, state[S.USE_HELD] >= .5, playing)).float()).squeeze(-1) > 0
        used = self._use(moved, facing, floor, ceiling, use_edge)
        crossing_distance, crossing_hit = self.geometry.ray_intersections(
            position[None, :], (moved - position)[None, :])
        crossed = (crossing_hit[0] & (crossing_distance[0] <= 1)
                   & ((self.line_special == 88)
                      | ((self.line_special == 36) & (state[layout.line_used] < .5))))
        crossed = crossed.float() * playing.float()
        floor, ceiling, motion, timer, floor_motion, floor_timer = self._sectors(
            state, elapsed, used, crossed, moved, entities, alive * self.solid_kind.float())
        player_sector = self.geometry.sector_at(moved[None, :])[0]
        support, _ = self.geometry.body_opening(moved[None, :], floor, ceiling, self.player_radius)
        feet = support[0]
        entity_ground = floor[self.geometry.sector_at(entities)]
        moving_feet, _ = self.geometry.body_opening(entities[self.enemy_index], floor, ceiling,
                                                   self.radius[self.enemy_index])
        entity_feet = entity_ground.index_copy(0, self.enemy_index, moving_feet)
        distance = (entities - moved).square().sum(-1).sqrt()
        sight = self.geometry.visible(moved[None, :].expand_as(entities), entities,
                                      (feet + 41).expand_as(entity_feet), entity_feet + S.AIM_HEIGHT,
                                      floor, ceiling)
        requested = controls[6].round().long()
        request_valid = (requested >= 0) & (requested <= 2) & (
            (requested != 2) | (state[S.SHOTGUN_OWNED] > .5))
        weapon = torch.where(request_valid & playing, requested.float(), state[S.WEAPON])
        empty = ((weapon == 1) & (state[S.BULLETS] <= 0)) | (
            (weapon == 2) & (state[S.SHELLS] <= 0))
        available = torch.where((state[S.SHOTGUN_OWNED] > .5) & (state[S.SHELLS] > 0),
                                 torch.full_like(weapon, 2),
                                 (state[S.BULLETS] > 0).float())
        weapon = torch.where(empty, available, weapon)
        weapon_activation = (weapon == self.weapon_codes).float()
        weapon_rule = F.linear(weapon_activation, self.weapon_weight)
        cooldown = F.relu(state[S.COOLDOWN] - elapsed)
        fired = self.fire_rule(torch.stack((
            controls[3] > .5, cooldown <= 0, playing)).float()).squeeze(-1) > 0
        rng = torch.remainder(state[S.RNG] * 75 + 74, 32749)
        noise = torch.frac((rng + self.entity_index.float() * 139) / 32749)
        # Each pellet has its own circle-impact ray and vertical auto-aim.
        # Center sight is useful for AI and pickups, but cannot establish that
        # an offset pellet avoids walls or traverses the same portal openings.
        pellet_angle = angle + self.pellet_offset * (weapon == 2).float()
        pellet_dir = torch.stack((torch.cos(pellet_angle), torch.sin(pellet_angle)), -1)
        relative = entities - moved
        forward = (pellet_dir[:, None, :] * relative).sum(-1)
        sideways = cross(pellet_dir[:, None, :], relative)
        half_chord = F.relu(self.radius.square() - sideways.square()).sqrt()
        # The forward ray's first contact is zero when its origin is inside
        # the circle. An intersection wholly behind the origin is ineligible.
        ray_distance = F.relu(forward - half_chord)
        impacts = moved + pellet_dir[:, None, :] * ray_distance[..., None]
        shootable = self.shootable_index
        impact_points = impacts[:, shootable, :].reshape(-1, 2)
        impact_height = (entity_feet[shootable] + S.AIM_HEIGHT)[None, :].expand(7, -1).reshape(-1)
        pellet_clear = self.geometry.visible(
            moved[None, :].expand_as(impact_points), impact_points,
            (feet + 41).expand_as(impact_height), impact_height, floor, ceiling).reshape(7, -1)
        visibility = torch.zeros_like(ray_distance).index_copy(1, shootable, pellet_clear.float())
        hits = self.pellet_hit_rule(torch.stack((
            forward + half_chord >= 0, sideways.abs() <= self.radius,
            (hp > 0).expand_as(ray_distance), self.enemy.expand_as(ray_distance),
            self.barrel.expand_as(ray_distance), ray_distance <= weapon_rule[1],
            visibility > 0), -1).float()).squeeze(-1) > 0
        hit_score = torch.where(hits, ray_distance, 1e6)
        nearest, target = self.pellet_minimum(hit_score)
        hit_activation = self.pellet_action_rule(torch.stack((
            target[:, None] == self.entity_index[None, :],
            (nearest < 1e6)[:, None].expand_as(hit_score),
            (self.pellet_index == 0)[:, None].expand_as(hit_score),
            (weapon == 2).expand_as(hit_score), fired.expand_as(hit_score)), -1).float()).squeeze(-1)
        pellet_damage = 5 * (1 + torch.floor(torch.frac(
            rng / 32749 + self.pellet_index * .381966) * 3))
        pellet_damage = torch.where(weapon == 0,
                                    2 * (1 + torch.floor(rng / 32749 * 10)), pellet_damage)
        shot_damage = (hit_activation * pellet_damage[:, None]).sum(0)
        hp = F.relu(hp - shot_damage)
        # Barrels are destructible. A chained barrel detonates next evaluation;
        # alive distinguishes a pending zero-health barrel from its spent shell.
        exploding = self.barrel & (hp <= 0) & (alive > .5) & playing
        barrels = self.barrel_index
        barrel_positions = entities[barrels]
        blast_distance = (barrel_positions[:, None, :] - entities[None, :, :]).square().sum(-1).sqrt()
        blast = F.relu(128 - blast_distance) * exploding[barrels, None].float()
        # Rays for barrel blasts are paired into a fixed dense tensor batch.
        origin = barrel_positions[:, None, :].expand(-1, self.layout.entity_count, -1).reshape(-1, 2)
        destination = entities[None, :, :].expand(self.barrel_count, -1, -1).reshape(-1, 2)
        source_z = (entity_feet[barrels] + S.AIM_HEIGHT)[:, None].expand(-1, self.layout.entity_count).reshape(-1)
        dest_z = (entity_feet + S.AIM_HEIGHT)[None, :].expand(self.barrel_count, -1).reshape(-1)
        blast_visible = self.geometry.visible(origin, destination, source_z, dest_z,
                                              floor, ceiling).reshape(self.barrel_count,
                                                                      self.layout.entity_count)
        blast *= blast_visible.float()
        hp = F.relu(hp - blast.sum(0) * (self.enemy | self.barrel).float())
        barrel_player_damage = (F.relu(128 - distance) * exploding.float() * sight.float()).sum()
        died = self.enemy & (state[layout.entity_hp] > 0) & (hp <= 0)
        drops = torch.maximum(state[layout.entity_drop],
                               (died & ((self.thing == 3004) | (self.thing == 9))).float())
        alive = torch.where(self.enemy & (hp <= 0) | exploding, torch.zeros_like(alive), alive)
        active = state[layout.entity_active]
        entity_angle = state[layout.entity_angle]
        to_player = moved - entities
        look = torch.stack((torch.cos(entity_angle), torch.sin(entity_angle)), -1)
        sees = sight & ((look * to_player).sum(-1) >= 0) & (distance < 2200)
        hears = fired & (distance < 2200) & ~self.ambush
        awakened = self.enemy & (hp > 0) & (sees | hears | (shot_damage > 0)) & playing
        newly_active = awakened & (active < .5)
        active = torch.maximum(active, awakened.float())
        entity_cooldown = F.relu(state[layout.entity_cooldown] - elapsed)
        entity_cooldown = torch.where(newly_active, torch.full_like(entity_cooldown, .6),
                                      entity_cooldown)
        ready = self.enemy & (hp > 0) & (active > .5) & sight & (
            distance < 1536) & (entity_cooldown <= 0) & playing
        imp = self.thing == 3001
        imp_melee = ready & imp & (distance < S.IMP_MELEE_RANGE)
        projectile_alive = state[layout.projectile_alive]
        launch = ready & imp & ~imp_melee & (projectile_alive < .5)
        attack = (ready & ~imp) | imp_melee | launch
        entity_cooldown = torch.where(attack, self.enemy_period, entity_cooldown)
        accuracy = (.85 - distance / 2200).clamp(.15, .85)
        hitscan_damage = (3 * (1 + torch.floor(noise * 5))
                          * torch.where(self.thing == 9, 3., 1.)
                          * (noise < accuracy).float() * (ready & ~imp).float()).sum()
        melee_damage = (3 * (1 + torch.floor(noise * 8)) * imp_melee.float()).sum()
        # Each imp owns a bounded projectile slot, whose full position and
        # velocity are recurrent activations rather than an external object.
        projectile_pos = torch.stack((state[layout.projectile_x], state[layout.projectile_y]), -1)
        projectile_delta = torch.stack((state[layout.projectile_dx],
                                        state[layout.projectile_dy]), -1) * elapsed
        projectile_next = torch.where(projectile_alive[:, None] > .5,
                                      projectile_pos + projectile_delta, projectile_pos)
        projectile_z = torch.where(projectile_alive > .5,
                                   state[layout.projectile_z] + state[layout.projectile_dz] * elapsed,
                                   state[layout.projectile_z])
        projectile_visible = self.geometry.projectile_clear(
            projectile_pos, projectile_next, state[layout.projectile_z], projectile_z, floor, ceiling)
        projectile_hit_player = (projectile_next - moved).square().sum(-1) < 22 ** 2
        projectile_hit_player &= (projectile_z >= feet) & (projectile_z <= feet + 56)
        projectile_hit_player &= (projectile_alive > .5) & projectile_visible & playing
        projectile_damage = (3 * (1 + torch.floor(noise * 8))
                              * projectile_hit_player.float()).sum()
        projectile_alive = (projectile_alive * projectile_visible.float()
                            * (~projectile_hit_player).float())
        launch_direction = to_player / distance.clamp_min(1)[:, None]
        projectile_next = torch.where(launch[:, None], entities, projectile_next)
        projectile_z = torch.where(launch, entity_feet + S.PROJECTILE_LAUNCH_HEIGHT, projectile_z)
        projectile_dx = torch.where(launch, launch_direction[:, 0] * S.PROJECTILE_SPEED,
                                    state[layout.projectile_dx])
        projectile_dy = torch.where(launch, launch_direction[:, 1] * S.PROJECTILE_SPEED,
                                    state[layout.projectile_dy])
        projectile_dz = torch.where(launch, (feet + S.AIM_HEIGHT - entity_feet - S.PROJECTILE_LAUNCH_HEIGHT)
                                    / distance.clamp_min(1) * S.PROJECTILE_SPEED, state[layout.projectile_dz])
        projectile_alive = torch.maximum(projectile_alive, launch.float())
        # Fixed eight-direction-style pursuit with a perpendicular probe when
        # the direct slide is blocked. Only living active enemies move.
        indices = self.enemy_index
        enemy_positions = entities[indices]
        chase_direction = to_player[indices] / distance[indices].clamp_min(1)[:, None]
        chase = (hp[indices] > 0) & (active[indices] > .5) & ~attack[indices]
        chase &= distance[indices] > 48
        enemy_delta = chase_direction * self.enemy_speed[indices, None] * elapsed * chase[:, None].float()
        enemy_moved = self._slide(enemy_positions, enemy_delta, entity_feet[indices],
                                  floor, ceiling, self.radius[indices], entities,
                                  self.solid_kind & (alive > .5), indices)
        stuck = (enemy_moved - enemy_positions).square().sum(-1) < .01
        side = torch.where(torch.frac((state[S.TIME] + indices.float()) / 3) < .5, 1., -1.)
        alternate = torch.stack((-chase_direction[:, 1] * side,
                                  chase_direction[:, 0] * side), -1)
        alternate_delta = alternate * self.enemy_speed[indices, None] * elapsed * chase[:, None].float()
        alternate_moved = self._slide(enemy_positions, alternate_delta, entity_feet[indices],
                                      floor, ceiling, self.radius[indices], entities,
                                      self.solid_kind & (alive > .5), indices)
        enemy_moved = torch.where(stuck[:, None], alternate_moved, enemy_moved)
        player_overlap = (enemy_moved - moved).square().sum(-1) < (self.radius[indices] + self.player_radius).square()
        enemy_moved = torch.where(player_overlap[:, None], enemy_positions, enemy_moved)
        entities = entities.index_copy(0, indices, enemy_moved)
        entity_angle = torch.where(self.enemy & (active > .5) & (hp > 0),
                                    torch.atan2(to_player[:, 1], to_player[:, 0]), entity_angle)
        # Nukage applies five damage every 32 original 35-Hz ticks.
        unwrapped_time = state[S.TIME] + elapsed
        hazard_tick = torch.floor(unwrapped_time * 35 / 32) > torch.floor(state[S.TIME] * 35 / 32)
        # 96 seconds spans 32 pursuit cycles, 672 animation frames, 105 hazard
        # periods. Test the hazard crossing before wrapping the recurrent clock.
        now = torch.remainder(unwrapped_time, S.CLOCK_PERIOD)
        player_special = self.sector_special.gather(0, player_sector.reshape(1)).squeeze(0)
        grounded = feet == floor.gather(0, player_sector.reshape(1)).squeeze(0)
        hazard_damage = ((player_special == 7) & grounded).float() * 5 * hazard_tick.float()
        damage = hitscan_damage + melee_damage + projectile_damage + barrel_player_damage + hazard_damage
        armor_save = torch.minimum(state[S.ARMOR], torch.floor(damage * torch.where(
            state[S.ARMOR_TYPE] == 2, .5, 1 / 3)))
        health = F.relu(state[S.HEALTH] - damage + armor_save)
        armor = F.relu(state[S.ARMOR] - armor_save)
        armor_type = torch.where(armor > 0, state[S.ARMOR_TYPE],
                                 torch.zeros_like(armor))
        bullets = state[S.BULLETS] - weapon_rule[2] * fired.float()
        shells = state[S.SHELLS] - weapon_rule[3] * fired.float()
        # Reach, vertical distance, and wall visibility gate item contacts.
        touching = (distance < 32) & ((entity_feet - feet).abs() < 56) & sight
        touching &= (health > 0) & playing
        collectible = (((self.thing == 2011) | (self.thing == 2012)) & (health < 100)
                       | (self.thing == 2014) & (health < 200)
                       | (self.thing == 2015) & (armor < 200)
                       | (self.thing == 2018) & (armor < 100)
                       | (self.thing == 2019) & (armor < 200)
                       | ((self.thing == 2007) | (self.thing == 2048)) & (bullets < 200)
                       | ((self.thing == 2008) | (self.thing == 2049)) & (shells < 50)
                       | (self.thing == 2001) & ((state[S.SHOTGUN_OWNED] < .5) | (shells < 50)))
        picked = touching & collectible & (alive > .5)
        drop_sight = self.geometry.visible(moved[None, :].expand_as(entities[self.drop_index]),
                                          entities[self.drop_index], (feet + 41).expand_as(entity_ground[self.drop_index]),
                                          entity_ground[self.drop_index] + S.AIM_HEIGHT, floor, ceiling)
        drop_contact = ((distance[self.drop_index] < 32)
                        & ((entity_ground[self.drop_index] - feet).abs() < 56) & drop_sight
                        & (health > 0) & playing)
        drop_touching = torch.zeros_like(drops).index_copy(0, self.drop_index, drop_contact.float()) > .5
        drop_pick = drop_touching & (drops > .5) & (
            ((self.thing == 3004) & (bullets < 200))
            | ((self.thing == 9) & ((state[S.SHOTGUN_OWNED] < .5) | (shells < 50))))
        picked_effects = F.linear(picked.float(), self.pickup_weight)
        ordinary_health = ((picked & ((self.thing == 2011) | (self.thing == 2012))).float()
                           * self.pickup_weight[0]).sum()
        extra_health = picked_effects[0] - ordinary_health
        health = torch.maximum(health, torch.minimum(health + ordinary_health,
                                                     torch.full_like(health, 100)))
        health = torch.minimum(health + extra_health, torch.full_like(health, 200))
        green = (picked & (self.thing == 2018)).any()
        blue = (picked & (self.thing == 2019)).any()
        armor = torch.maximum(armor, 100 * green.float())
        armor = torch.maximum(armor, 200 * blue.float())
        armor = torch.minimum(armor + picked_effects[1], torch.full_like(armor, 200))
        armor_type = torch.where(blue, torch.full_like(armor, 2),
                                 torch.where(green, torch.ones_like(armor), armor_type))
        armor_type = torch.where((armor_type == 0) & (armor > 0),
                                 torch.ones_like(armor_type), armor_type)
        bullets += picked_effects[2] + (drop_pick & (self.thing == 3004)).float().sum() * 5
        shells += picked_effects[3] + (drop_pick & (self.thing == 9)).float().sum() * 4
        bullets = bullets.clamp_max(200)
        shells = shells.clamp_max(50)
        shotgun_pick = (picked & (self.thing == 2001)).any() | (drop_pick & (self.thing == 9)).any()
        shotgun_owned = torch.maximum(state[S.SHOTGUN_OWNED], shotgun_pick.float())
        weapon = torch.where(shotgun_pick & (state[S.SHOTGUN_OWNED] < .5),
                              torch.full_like(weapon, 2), weapon)
        alive *= (~picked).float()
        drops *= (~drop_pick).float()
        secret = state[layout.sector_secret]
        discovered = ((self.sector_special == 9)
                      & (self.sector_index == player_sector)
                      & (secret < .5) & playing & grounded)
        secret = torch.maximum(secret, discovered.float())
        won = torch.maximum(state[S.WON], ((used > 0) & (self.line_special == 11)).any().float())
        damage_flash = torch.maximum(F.relu(state[S.DAMAGE_FLASH] - dt * 2),
                                     (damage / 30).clamp(0, 1))
        pickup_flash = torch.maximum(F.relu(state[S.PICKUP_FLASH] - dt * 2),
                                     (picked.any() | drop_pick.any()).float() * .65)
        cooldown = torch.where(fired, weapon_rule[0], cooldown)
        fire_flash = torch.where(fired, torch.full_like(health, .12),
                                 F.relu(state[S.FIRE_FLASH] - dt))
        scalars = torch.stack((moved[0], moved[1], angle, feet, health, armor,
                               bullets, shells, weapon, shotgun_owned, cooldown,
                               damage_flash, pickup_flash, now,
                               state[S.KILLS] + died.float().sum(), won, fire_flash,
                               state[S.SECRETS] + discovered.float().sum(),
                               (controls[4] > .5).float(), armor_type, rng))
        output = torch.cat((scalars, entities[:, 0], entities[:, 1], hp, active,
                             entity_cooldown, alive, drops, entity_angle,
                             projectile_next[:, 0], projectile_next[:, 1], projectile_z,
                             projectile_dx, projectile_dy, projectile_dz, projectile_alive,
                             floor, ceiling, motion, timer, floor_motion, floor_timer, secret,
                             torch.maximum(state[layout.line_used], torch.maximum(used, crossed))))
        output = torch.where(playing, output, state)
        return torch.where(controls[5] > .5, self.spawn, output)
