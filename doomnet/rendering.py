"""Renderer source with deferred artwork reads for neural compilation.

Visibility circuits route artwork addresses and illumination through world,
sprite, weapon, and HUD selection. Only the final pixel addresses read RGB;
candidate transparency uses separate alpha tables. Source arithmetic and reads
must be lowered by the neural compiler before strict network execution.
"""

import math

import torch
from torch import nn

from .circuits import MinimumCircuit, NearestCircuit
from .geometry import Geometry, cross
from .state import (ANGLE, ARMOR, BULLETS, DAMAGE_FLASH, FIRE_FLASH, HEALTH,
                    PICKUP_FLASH, SHELLS, TIME, WEAPON, WON, X, Y, Z,
                    ENEMY_RADIUS, StateLayout, moving_actors)


_FAR = 1e6
_PATCH_BOUND = 14.0


def _message_art(width, height, scene_height):
    """Construct fixed lettering once, just as the WAD compiler decodes patches."""
    glyphs = {
        "A": (14, 17, 17, 31, 17, 17, 17),
        "C": (14, 17, 16, 16, 16, 17, 14),
        "D": (30, 17, 17, 17, 17, 17, 30),
        "E": (31, 16, 16, 30, 16, 16, 31),
        "I": (31, 4, 4, 4, 4, 4, 31),
        "L": (16, 16, 16, 16, 16, 16, 31),
        "M": (17, 27, 21, 21, 17, 17, 17),
        "O": (14, 17, 17, 17, 17, 17, 14),
        "P": (30, 17, 17, 30, 16, 16, 16),
        "R": (30, 17, 17, 30, 20, 18, 17),
        "S": (15, 16, 16, 14, 1, 1, 30),
        "T": (31, 4, 4, 4, 4, 4, 4),
        "U": (17, 17, 17, 17, 17, 17, 14),
        "V": (17, 17, 17, 17, 17, 10, 4),
        "Y": (17, 17, 10, 4, 4, 4, 4),
        " ": (0,) * 7,
    }
    art = torch.zeros(2, height, width, 4, dtype=torch.uint8)
    titles = ("YOU DIED", "LEVEL COMPLETE") if width >= 100 else ("DEAD", "COMPLETE")
    for page, title in enumerate(titles):
        scale = max(1, min(2, (width - 20) // (6 * len(title) - 1)))
        top = max(2, scene_height // 3)
        bottom = min(scene_height, top + 7 * scale + 18)
        art[page, top - 2:bottom, 8:width - 8, 3] = 190
        for text, y, text_scale in ((title, top, scale),
                                    ("PRESS R", top + 7 * scale + 5, 1)):
            left = (width - (6 * len(text) - 1) * text_scale) // 2
            for column, character in enumerate(text):
                for row, bits in enumerate(glyphs[character]):
                    for bit in range(5):
                        if bits & (1 << (4 - bit)):
                            x = left + (column * 6 + bit) * text_scale
                            yy = y + row * text_scale
                            if x >= 0 and x + text_scale <= width and yy + text_scale <= height:
                                art[page, yy:yy + text_scale, x:x + text_scale] = torch.tensor(
                                    [255, 190, 68, 255], dtype=torch.uint8)
    return art


class Renderer(nn.Module):
    """Return completed uint8 ``[H,W,3]`` frames through fixed tensor connectivity.

    Each column retains the nearest 64 map crossings and 24 billboard candidates.
    These fixed bounds keep intermediates small enough for live CPU play and are
    part of the exported graph's scope.
    """

    def __init__(self, data, width=160, height=120):
        super().__init__()
        if width < 64 or height < 48:
            raise ValueError("rendering requires width >= 64 and height >= 48")
        self.width, self.height = int(width), int(height)
        self.hud_height = max(16, self.height // 6)
        self.scene_height = self.height - self.hud_height
        self.focal = self.width / 2.0
        self.layout = StateLayout.from_data(data)
        self.geometry = Geometry(data)
        self.line_hits = min(64, len(data["lines"]))
        self.sprite_hits = min(24, 3 * len(data["entity_type"]))
        self.line_selection = NearestCircuit(len(data["lines"]), self.line_hits, bound=_FAR)
        self.billboard_selection = NearestCircuit(3 * self.layout.entity_count, self.sprite_hits, bound=_FAR)
        self.interval_selection = MinimumCircuit(self.line_hits, bound=_FAR)
        self.surface_selection = MinimumCircuit(2, bound=_FAR)
        self.sprite_selection = MinimumCircuit(self.sprite_hits, bound=_FAR)
        self.weapon_patch_selection = MinimumCircuit(2, bound=_PATCH_BOUND)
        self.hud_patch_selection = MinimumCircuit(13, bound=_PATCH_BOUND)

        for name in ("line_front_sector", "line_back_sector", "line_flags",
                     "line_texture", "line_offset", "sector_light",
                     "sector_floor_tex", "sector_ceiling_tex", "sector_sky",
                     "sky_texture", "texture_size", "sprite_size",
                     "sprite_offset", "entity_type", "entity_walk_sprite",
                     "entity_attack_sprite", "entity_dead_sprite", "entity_drop_sprite",
                     "projectile_sprite", "weapon_sprite", "weapon_fire_sprite", "weapon_flash_sprite",
                     "hud_face_sprite", "hud_digit_sprite", "hud_percent_sprite",
                     "hud_bar_sprite", "hud_dead_face_sprite"):
            self.register_buffer(name, data[name])
        texture = data["texture_rgba"]
        sprite = data["sprite_rgba"]
        flat = data["flat_rgba"]
        if texture.ndim != 4 or sprite.ndim != 4 or flat.shape[1:] != (64, 64, 4):
            raise ValueError("artwork requires padded RGBA texture/sprite atlases and 64x64 flats")
        self.texture_height, self.texture_width = texture.shape[1:3]
        self.sprite_height, self.sprite_width = sprite.shape[1:3]
        self.texture_moduli = tuple(tuple(torch.unique(data["texture_size"][:, axis]).tolist())
                                    for axis in range(2))
        self.sprite_base = texture[..., 3].numel()
        self.flat_base = self.sprite_base + sprite[..., 3].numel()
        self.black_address = self.flat_base + flat[..., 3].numel()
        self.register_buffer("artwork_rgb", torch.cat((texture[..., :3].reshape(-1, 3),
                                                       sprite[..., :3].reshape(-1, 3),
                                                       flat[..., :3].reshape(-1, 3),
                                                       texture.new_zeros(1, 3))))
        self.register_buffer("texture_alpha", texture[..., 3].reshape(-1))
        self.register_buffer("sprite_alpha", sprite[..., 3].reshape(-1))
        self.register_buffer("line_length", self.geometry.line_square.sqrt())
        self.register_buffer("can_die", torch.isin(data["entity_type"],
                                                   torch.tensor([3004, 9, 3001, 2035])))
        moving = moving_actors(data["entity_type"])
        self.register_buffer("moving_index", torch.where(moving)[0])
        self.register_buffer("support_radius", torch.full((int(moving.sum()),), ENEMY_RADIUS))
        self.register_buffer("entity_index", torch.arange(self.layout.entity_count))
        screen_x = (torch.arange(self.width).float() + 0.5 - self.width / 2) / self.focal
        slope = (self.scene_height / 2 - torch.arange(self.scene_height).float() - 0.5) / self.focal
        self.register_buffer("screen_x", screen_x)
        self.register_buffer("slope", slope)
        self.register_buffer("scene_x", torch.arange(self.width).float()[None, :] + 0.5)
        self.register_buffer("scene_y", torch.arange(self.scene_height).float()[:, None] + 0.5)
        self.register_buffer("hud_x", torch.arange(self.width).float()[None, :] + 0.5)
        self.register_buffer("hud_y", torch.arange(self.hud_height).float()[:, None] + 0.5)
        self.register_buffer("digit_place", torch.tensor([100, 10, 1]))
        self.register_buffer("hud_right", torch.tensor([44., 90., 221.]))
        self.register_buffer("digit_spacing", torch.tensor([42., 28., 14.]))
        self.register_buffer("digit_y", torch.full((9,), 3.0))
        self.register_buffer("hud_origin", torch.tensor([[0., 0.], [148., 3.]]))
        self.register_buffer("hud_scale", torch.tensor([self.width / 320, self.hud_height / 32]))
        self.register_buffer("hud_active", torch.ones(2, dtype=torch.bool))
        self.register_buffer("hud_address", torch.full((self.hud_height, self.width),
                                                       self.black_address, dtype=torch.long))
        self.register_buffer("hud_illumination", torch.ones(self.hud_height, self.width))
        self.register_buffer("percent_origin", torch.tensor([[90., 3.], [221., 3.]]))
        self.register_buffer("weapon_scale", torch.full((2,), self.width / 320))
        self.register_buffer("weapon_origin", torch.tensor([0., self.scene_height - 168 * self.width / 320]))
        self.register_buffer("patch_order", torch.arange(1, 14))
        self.register_buffer("damage_color", torch.tensor([255., 0., 0.]))
        self.register_buffer("pickup_color", torch.tensor([255., 220., 50.]))
        self.register_buffer("message_rgba", _message_art(self.width, self.height,
                                                           self.scene_height))

    def _texture_coordinate(self, coordinate, size, moduli):
        """Select exact wraps from the artwork's fixed scalar dimension classes."""
        coordinate = torch.floor(coordinate).long()
        wrapped = torch.zeros_like(coordinate)
        for modulus in moduli:
            wrapped = torch.where(size == modulus, torch.remainder(coordinate, modulus), wrapped)
        return wrapped

    def _texture_index(self, texture, u, v):
        texture = texture.clamp_min(0)
        size = self.texture_size[texture]
        u = self._texture_coordinate(u, size[..., 0], self.texture_moduli[0])
        v = self._texture_coordinate(v, size[..., 1], self.texture_moduli[1])
        return texture * (self.texture_height * self.texture_width) + v * self.texture_width + u

    def _sprite_index(self, sprite, u, v):
        return (sprite.clamp_min(0) * (self.sprite_height * self.sprite_width)
                + v.long().clamp(0, self.sprite_height - 1) * self.sprite_width
                + u.long().clamp(0, self.sprite_width - 1))

    def _illumination(self, light, depth):
        return (0.2 + 0.8 * light / 255.0) * (1.0 - depth / 2500.0).clamp(0.4, 1.0)

    def _surfaces(self, state, forward, right, eye):
        """Nearest wall and plane intersections, with sector-height portal clipping."""
        origin = state[X:Y + 1]
        rays = forward[None, :] + self.screen_x[:, None] * right[None, :]
        relative = self.geometry.line_start[None, :, :] - origin[None, None, :]
        denominator = cross(rays[:, None, :], self.geometry.line_delta)
        safe = torch.where(denominator.abs() > 1e-7, denominator,
                           torch.ones_like(denominator))
        distance = cross(relative, self.geometry.line_delta) / safe
        along = cross(relative, rays[:, None, :]) / safe
        intersects = ((denominator.abs() > 1e-7) & (distance > 1e-4) & (distance < _FAR)
                      & (along >= 0) & (along <= 1))
        depth, line = self.line_selection(torch.where(intersects, distance, _FAR))
        along = torch.gather(torch.where(intersects, along, 0), 1, line)
        front_side = torch.gather(denominator, 1, line) < 0
        near = torch.where(front_side, self.line_front_sector[line], self.line_back_sector[line])
        far = torch.where(front_side, self.line_back_sector[line], self.line_front_sector[line])
        side = (~front_side).long()
        textures = self.line_texture[line, side]
        offsets = self.line_offset[line, side]
        flags = self.line_flags[line]
        floor = state[self.layout.sector_floor]
        ceiling = state[self.layout.sector_ceiling]
        nf, nc = floor[near.clamp_min(0)], ceiling[near.clamp_min(0)]
        ff, fc = floor[far.clamp_min(0)], ceiling[far.clamp_min(0)]
        z = eye + self.slope[:, None, None] * depth[None, :, :]
        in_sector = ((z >= nf[None, :, :]) & (z <= nc[None, :, :])
                     & (near[None, :, :] >= 0) & (depth[None, :, :] < _FAR))
        solid = far < 0
        lower = (~solid)[None, :, :] & (z < ff[None, :, :])
        both_sky = self.sector_sky[near.clamp_min(0)] & self.sector_sky[far.clamp_min(0)]
        upper = ((~solid & ~both_sky)[None, :, :] & (z > fc[None, :, :]))
        middle = (~solid)[None, :, :] & ~lower & ~upper
        texture = torch.where(solid[None, :, :], textures[None, :, :, 2],
                              torch.where(lower, textures[None, :, :, 1],
                                          torch.where(upper, textures[None, :, :, 0],
                                                      textures[None, :, :, 2])))
        texture_height = self.texture_size[texture.clamp_min(0), 1].float()
        middle_height = self.texture_size[textures[..., 2].clamp_min(0), 1].float()
        middle_top = torch.where((flags & 16) != 0, nf + middle_height, nc)
        masked_top = torch.where((flags & 16) != 0,
                                 torch.maximum(nf, ff) + middle_height,
                                 torch.minimum(nc, fc))
        upper_top = torch.where((flags & 8)[None, :, :] != 0, nc[None, :, :],
                                fc[None, :, :] + texture_height)
        lower_top = torch.where((flags & 16)[None, :, :] != 0, nc[None, :, :], ff[None, :, :])
        top = torch.where(solid[None, :, :], middle_top[None, :, :],
                          torch.where(lower, lower_top,
                                      torch.where(upper, upper_top, masked_top[None, :, :])))
        u = torch.where(front_side, along, 1 - along) * self.line_length[line] + offsets[..., 0]
        # Hidden crossings need no projected UV; retain the original z above
        # for geometry predicates, and only carry in-sector heights into art.
        wall_z = torch.where(in_sector, z, 0)
        v = top - wall_z + offsets[None, :, :, 1]
        texel = self._texture_index(texture, u[None, :, :], v)
        masked_visible = ((v >= 0) & (v < texture_height) & (self.texture_alpha[texel] != 0))
        wall_valid = in_sector & (texture >= 0) & (~middle | masked_visible)
        wall_depth, wall_hit = self.interval_selection(torch.where(wall_valid, depth[None, :, :], _FAR))
        wall_texel = torch.gather(texel, 2, wall_hit[..., None]).squeeze(-1)
        wall_sector = torch.gather(near[None, :, :].expand(self.scene_height, -1, -1),
                                   2, wall_hit[..., None]).squeeze(-1).clamp_min(0)
        wall_illumination = self._illumination(self.sector_light[wall_sector], wall_depth)

        # The interval before a hit belongs to its near sector.  The initial
        # interval is explicitly decoded from the player's BSP leaf.
        start_sector = self.geometry.sector_at(origin.reshape(1, 2))[0]
        interval_sector = torch.cat((start_sector.expand(self.width, 1), near[:, 1:]), dim=1)
        sector = interval_sector.clamp_min(0)
        start = torch.cat((torch.zeros_like(depth[:, :1]), depth[:, :-1]), dim=1)
        looking_up = self.slope > 0
        plane_height = torch.where(looking_up[:, None, None], ceiling[sector][None, :, :],
                                   floor[sector][None, :, :])
        safe_slope = torch.where(self.slope.abs() > 1e-7, self.slope,
                                 torch.full_like(self.slope, 1e-7))
        plane = (plane_height - eye) / safe_slope[:, None, None]
        # Sentinel slots have no ending crossing or associated near sector.
        # They cannot extend a map interval into unrelated sector metadata.
        plane_valid = ((plane > 1e-4) & (plane >= start[None, :, :])
                       & (plane <= depth[None, :, :]) & (depth[None, :, :] < _FAR)
                       & (interval_sector[None, :, :] >= 0))
        plane_depth, plane_hit = self.interval_selection(torch.where(plane_valid, plane, _FAR))
        plane_sector = torch.gather(sector[None, :, :].expand(self.scene_height, -1, -1),
                                    2, plane_hit[..., None]).squeeze(-1)
        plane_distance = torch.where(plane_depth < _FAR, plane_depth, 0)
        world = origin[None, None, :] + plane_distance[..., None] * rays[None, :, :]
        flat = torch.where(looking_up[:, None], self.sector_ceiling_tex[plane_sector],
                            self.sector_floor_tex[plane_sector])
        fu = torch.remainder(torch.floor(world[..., 0]).long(), 64)
        fv = torch.remainder(torch.floor(-world[..., 1]).long(), 64)
        flat_texel = flat.clamp_min(0) * 4096 + fv * 64 + fu
        plane_address = self.flat_base + flat_texel
        plane_illumination = self._illumination(self.sector_light[plane_sector], plane_depth)

        sky_id = self.sky_texture
        sky_u = (state[ANGLE] - torch.atan(self.screen_x)) * (1024 / (2 * math.pi))
        sky_v = (self.scene_y - self.scene_height / 2) * (320 / self.width) + 100
        sky_v = sky_v.clamp(0, self.texture_size[sky_id, 1] - 1)
        sky_address = self._texture_index(sky_id, sky_u[None, :], sky_v)
        plane_sky = looking_up[:, None] & self.sector_sky[plane_sector]
        plane_address = torch.where(plane_sky, sky_address, plane_address)
        plane_illumination = torch.where(plane_sky, torch.ones_like(plane_illumination),
                                         plane_illumination)
        # Plane comes first to preserve the plane-first tie at shared edges.
        nearest, surface = self.surface_selection(torch.stack((plane_depth, wall_depth), dim=-1))
        addresses = torch.stack((plane_address, wall_texel), dim=2)
        illumination = torch.stack((plane_illumination, wall_illumination), dim=2)
        address = torch.gather(addresses, 2, surface[..., None]).squeeze(2)
        illumination = torch.gather(illumination, 2, surface[..., None]).squeeze(2)
        address = torch.where(nearest < _FAR, address, sky_address)
        illumination = torch.where(nearest < _FAR, illumination, torch.ones_like(illumination))
        return address, illumination, nearest, floor

    def _sprites(self, state, address, illumination, scene_depth, floor, forward, right, eye):
        """Depth-tested rotating enemy art, corpses, pickups, props, and fireballs."""
        layout = self.layout
        ex, ey = state[layout.entity_x], state[layout.entity_y]
        positions = torch.stack((ex, ey), dim=1)
        sectors = self.geometry.sector_at(positions)
        ground = floor[sectors]
        support, _ = self.geometry.body_opening(positions[self.moving_index], floor,
                                               state[layout.sector_ceiling], self.support_radius)
        ground = ground.index_copy(0, self.moving_index, support)
        relative_angle = (torch.atan2(state[Y] - ey, state[X] - ex)
                          - state[layout.entity_angle])
        rotation = torch.remainder(torch.floor((relative_angle + math.pi / 8) / (math.pi / 4)).long(), 8)
        frame = torch.remainder(torch.floor(state[TIME] * 7).long(), 4)
        sprite = self.entity_walk_sprite[self.entity_index, frame, rotation]
        attacking = (state[layout.entity_cooldown] > 1.2) & (state[layout.entity_active] > 0.5)
        sprite = torch.where(attacking, self.entity_attack_sprite[self.entity_index, rotation], sprite)
        alive = state[layout.entity_alive] > 0.5
        corpse = self.can_die & ~alive & (state[layout.entity_hp] <= 0)
        sprite = torch.where(corpse, self.entity_dead_sprite, sprite)
        active = (alive | corpse) & (sprite >= 0)
        packed_pos = torch.cat((positions, positions,
                                torch.stack((state[layout.projectile_x], state[layout.projectile_y]), dim=1)))
        packed_ground = torch.cat((ground, floor[sectors], state[layout.projectile_z]))
        packed_sprite = torch.cat((sprite, self.entity_drop_sprite,
                                   self.projectile_sprite.expand(layout.entity_count)))
        packed_active = torch.cat((active, (state[layout.entity_drop] > 0.5) & (self.entity_drop_sprite >= 0),
                                   state[layout.projectile_alive] > 0.5))
        packed_light = torch.cat((self.sector_light[sectors], self.sector_light[sectors],
                                  torch.full_like(ground, 255)))
        delta = packed_pos - state[X:Y + 1][None, :]
        depth = (delta * forward[None, :]).sum(-1)
        horizontal = (delta * right[None, :]).sum(-1)
        size = self.sprite_size[packed_sprite.clamp_min(0)]
        offset = self.sprite_offset[packed_sprite.clamp_min(0)]
        top_v = offset[:, 1] + packed_ground - eye - self.slope[0] * depth
        bottom_v = offset[:, 1] + packed_ground - eye - self.slope[-1] * depth
        vertical = (top_v < size[:, 1]) & (bottom_v >= 0)
        u = self.screen_x[:, None] * depth[None, :] - horizontal[None, :] + offset[None, :, 0]
        candidates = ((u >= 0) & (u < size[None, :, 0])
                      & (depth[None, :] > 2) & packed_active[None, :] & vertical[None, :])
        selected_depth, entity = self.billboard_selection(torch.where(candidates, depth[None, :], _FAR))
        selected_sprite = packed_sprite[entity]
        selected_u = torch.gather(u, 1, entity)
        v = (offset[entity, 1][None, :, :] + packed_ground[entity][None, :, :] - eye
             - self.slope[:, None, None] * selected_depth[None, :, :])
        texel = self._sprite_index(selected_sprite[None, :, :], selected_u[None, :, :], v)
        valid = ((v >= 0) & (v < size[entity, 1][None, :, :])
                 & (self.sprite_alpha[texel] != 0) & (selected_depth[None, :, :] < scene_depth[..., None]))
        sprite_depth, hit = self.sprite_selection(torch.where(valid, selected_depth[None, :, :], _FAR))
        winning_texel = torch.gather(texel, 2, hit[..., None]).squeeze(-1)
        light = torch.gather(packed_light[entity][None, :, :].expand(self.scene_height, -1, -1),
                             2, hit[..., None]).squeeze(-1)
        visible = sprite_depth < _FAR
        address = torch.where(visible, self.sprite_base + winning_texel, address)
        illumination = torch.where(visible, self._illumination(light, sprite_depth), illumination)
        return address, illumination

    def _patches(self, address, illumination, sprite, origin, scale, active, grid_x, grid_y, selector):
        """Neurally route the highest visible patch's address and illumination."""
        size = self.sprite_size[sprite.clamp_min(0)]
        u = (grid_x[..., None] - origin[None, None, :, 0]) / scale[0]
        v = (grid_y[..., None] - origin[None, None, :, 1]) / scale[1]
        texel = self._sprite_index(sprite[None, None, :], u, v)
        visible = ((u >= 0) & (u < size[None, None, :, 0]) & (v >= 0)
                   & (v < size[None, None, :, 1]) & active[None, None, :]
                   & (self.sprite_alpha[texel] != 0))
        order = self.patch_order[:sprite.shape[0]]
        score, winning = selector(_PATCH_BOUND - visible.float() * order[None, None, :])
        selected = torch.gather(texel.expand(address.shape[0], address.shape[1], -1),
                                2, winning[..., None]).squeeze(-1)
        visible = score < _PATCH_BOUND
        address = torch.where(visible, self.sprite_base + selected, address)
        illumination = torch.where(visible, torch.ones_like(illumination), illumination)
        return address, illumination

    def _weapon(self, state, address, illumination):
        weapon = state[WEAPON].long().clamp(0, 2)
        body = torch.where(state[FIRE_FLASH] > 0, self.weapon_fire_sprite[weapon], self.weapon_sprite[weapon])
        sprite = torch.stack((body, self.weapon_flash_sprite[weapon]))
        offset = self.sprite_offset[sprite.clamp_min(0)].float()
        origin = -offset * self.weapon_scale[None, :] + self.weapon_origin[None, :]
        active = torch.stack((state[HEALTH] > 0, (state[HEALTH] > 0) & (state[FIRE_FLASH] > 0)))
        active = active & (sprite >= 0)
        return self._patches(address, illumination, sprite, origin, self.weapon_scale, active,
                             self.scene_x, self.scene_y,
                             self.weapon_patch_selection)

    def _hud(self, state):
        weapon = state[WEAPON].long().clamp(0, 2)
        ammo = torch.stack((state[BULLETS] * 0, state[BULLETS], state[SHELLS]))[weapon]
        value = torch.stack((ammo, state[HEALTH], state[ARMOR])).clamp(0, 999).long()
        digits = torch.remainder(torch.div(value[:, None], self.digit_place[None, :], rounding_mode="floor"), 10)
        tier = torch.div((100 - state[HEALTH]).clamp(0, 99).long(), 20, rounding_mode="floor")
        face = torch.where(state[HEALTH] <= 0, self.hud_dead_face_sprite, self.hud_face_sprite[tier])
        sprite = torch.cat((self.hud_bar_sprite.reshape(1), face.reshape(1),
                             self.hud_digit_sprite[digits].reshape(-1), self.hud_percent_sprite.expand(2)))
        digit_x = self.hud_right[:, None] - self.digit_spacing[None, :]
        digit_origin = torch.stack((digit_x.reshape(-1), self.digit_y), dim=1)
        origin = torch.cat((self.hud_origin, digit_origin,
                             self.percent_origin), dim=0)
        active_digits = (value[:, None] >= self.digit_place[None, :]) | (self.digit_place[None, :] == 1)
        active = torch.cat((self.hud_active, active_digits.reshape(-1), self.hud_active))
        return self._patches(self.hud_address, self.hud_illumination, sprite,
                             origin * self.hud_scale[None, :],
                             self.hud_scale, active, self.hud_x, self.hud_y, self.hud_patch_selection)

    def forward(self, state):
        forward = torch.stack((torch.cos(state[ANGLE]), torch.sin(state[ANGLE])))
        right = torch.stack((forward[1], -forward[0]))
        eye = state[Z] + 41
        address, illumination, depth, floor = self._surfaces(state, forward, right, eye)
        address, illumination = self._sprites(state, address, illumination, depth, floor,
                                              forward, right, eye)
        address, illumination = self._weapon(state, address, illumination)
        hud_address, hud_illumination = self._hud(state)
        address = torch.cat((address, hud_address), dim=0)
        illumination = torch.cat((illumination, hud_illumination), dim=0)
        image = self.artwork_rgb[address].float() * illumination[..., None]
        damage = (state[DAMAGE_FLASH] * 2).clamp(0, 0.6)
        pickup = (state[PICKUP_FLASH] * 2).clamp(0, 0.35)
        image = image * (1 - damage) + self.damage_color * damage
        image = image * (1 - pickup) + self.pickup_color * pickup
        message = self.message_rgba[(state[WON] > 0.5).long()].float()
        alpha = message[..., 3:4] / 255 * ((state[HEALTH] <= 0) | (state[WON] > 0.5)).float()
        image = image * (1 - alpha) + message[..., :3] * alpha
        return image.clamp(0, 255).to(torch.uint8)
