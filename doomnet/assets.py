"""Compile the original shareware E1M1 into immutable network data tensors."""

from collections import Counter
import hashlib
import math
from pathlib import Path

import torch

from .wad import (
    Patch, SHAREWARE_MD5, SHAREWARE_README_URL, SHAREWARE_URL, Wad,
    name_of, records, verify_shareware,
)


# Original Doom editor-number to artwork mappings, from id Software's info.c.
ENEMIES = {3004: ("POSS", "L"), 9: ("SPOS", "L"), 3001: ("TROO", "M")}
THING_PATCH = {
    10: "PLAYW0", 12: "PLAYW0", 15: "PLAYN0", 24: "POL5A0",
    35: "CBRAA0", 48: "ELECA0", 2028: "COLUA0", 2035: "BAR1A0",
    2001: "SHOTA0", 2007: "CLIPA0",
    2008: "SHELA0", 2011: "STIMA0", 2012: "MEDIA0", 2014: "BON1A0",
    2015: "BON2A0", 2018: "ARM1A0", 2019: "ARM2A0",
    2048: "AMMOA0", 2049: "SBOXA0",
}
SKILL_NAMES = ("I'm Too Young to Die", "Hey, Not Too Rough", "Hurt Me Plenty",
               "Ultra-Violence", "Nightmare")


def _tensor(values, dtype=torch.int64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


def _check_indices(values, count: int, label: str, absent: bool = False) -> None:
    for value in values:
        if not (0 <= value < count or (absent and value == -1)):
            raise ValueError(f"{label}: index {value} lies outside [0,{count})")


def _atlas(images: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    assert images, "An artwork atlas must contain at least one image"
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    atlas = torch.zeros((len(images), height, width, 4), dtype=torch.uint8)
    for i, image in enumerate(images):
        atlas[i, :image.shape[0], :image.shape[1]] = image
    sizes = _tensor([[image.shape[1], image.shape[0]] for image in images])
    return atlas, sizes


class _Sprites:
    def __init__(self, wad: Wad):
        self.wad = wad
        self.names: list[str] = []
        self.patches: list[Patch] = []
        self.ids: dict[str, int] = {}
        self.frames: dict[tuple[str, str, int], tuple[str, bool]] = {}
        start, end = wad.lookup["S_START"], wad.lookup["S_END"]
        if start >= end:
            raise ValueError("Sprite marker order is invalid")
        for lump in wad.lumps[start + 1:end]:
            name = lump.name
            if len(name) not in (6, 8):
                raise ValueError(f"Invalid sprite lump name {name}")
            for offset in (4, 6) if len(name) == 8 else (4,):
                if not name[offset + 1].isdigit() or int(name[offset + 1]) > 8:
                    raise ValueError(f"Invalid sprite rotation in {name}")
                key = (name[:4], name[offset], int(name[offset + 1]))
                if key in self.frames:
                    raise ValueError(f"Duplicate sprite frame {key}")
                self.frames[key] = (name, offset == 6)

    def add(self, name: str, mirror: bool = False) -> int:
        key = name + ("~mirror" if mirror else "")
        if key not in self.ids:
            patch = self.wad.patch(name)
            if mirror:
                patch = Patch(patch.rgba.flip(1), patch.width - patch.left, patch.top)
            self.ids[key] = len(self.names)
            self.names.append(key)
            self.patches.append(patch)
        return self.ids[key]

    def rotations(self, prefix: str, frame: str) -> list[int]:
        if (prefix, frame, 0) in self.frames:
            name, mirror = self.frames[prefix, frame, 0]
            return [self.add(name, mirror)] * 8
        result = []
        for rotation in range(1, 9):
            key = (prefix, frame, rotation)
            if key not in self.frames:
                raise ValueError(f"Missing required directional sprite {key}")
            result.append(self.add(*self.frames[key]))
        return result


def build_assets(wad_path: str | Path, skill: int = 2) -> tuple[dict[str, torch.Tensor], dict]:
    """Prepare fixed map and artwork tensors; ``skill`` is Doom's 0..4 enum.

    Heights and positions use original map units; angles use radians. Texture
    sizes and sprite offsets are [width,height] and [left,top]. Line side order
    is front/back and texture slot order is upper/lower/middle. Missing line
    textures and back sectors use -1. Sprite pixels are padded, never rescaled.
    BSP affine values below/above zero select child 0/1. Exact equality follows
    vanilla R_PointOnSide: child 0 for horizontal dx > 0, vertical dy < 0, or a
    NW diagonal (dx < 0, dy > 0) exactly at its node_origin; other equality
    selects child 1. Negative child values encode
    ``-1-subsector``. Player and multiplayer-only starts
    are excluded from the unified entity table, which retains WAD thing order.
    """
    if not isinstance(skill, int) or isinstance(skill, bool) or not 0 <= skill <= 4:
        raise ValueError("skill must be a Doom skill enum integer from 0 to 4")
    wad = Wad(wad_path)
    verify_shareware(wad.raw)
    lumps = wad.map_lumps("E1M1")
    vertices = records(lumps["VERTEXES"], "<hh", "VERTEXES")
    lines = records(lumps["LINEDEFS"], "<7H", "LINEDEFS")
    sides = records(lumps["SIDEDEFS"], "<hh8s8s8sH", "SIDEDEFS")
    sectors = records(lumps["SECTORS"], "<hh8s8shhh", "SECTORS")
    things = records(lumps["THINGS"], "<hhhhh", "THINGS")
    segs = records(lumps["SEGS"], "<6H", "SEGS")
    subsectors = records(lumps["SSECTORS"], "<HH", "SSECTORS")
    nodes = records(lumps["NODES"], "<12h2H", "NODES")
    if not all((vertices, lines, sides, sectors, things, segs, subsectors, nodes)):
        raise ValueError("E1M1 requires nonempty geometry and BSP lumps")
    _check_indices([v for line in lines for v in line[:2]], len(vertices), "LINEDEFS vertex")
    line_sides = [[-1 if side == 65535 else side for side in line[5:7]] for line in lines]
    _check_indices([row[0] for row in line_sides], len(sides), "LINEDEFS front side")
    _check_indices([row[1] for row in line_sides], len(sides), "LINEDEFS back side", True)
    _check_indices([side[5] for side in sides], len(sectors), "SIDEDEFS sector")
    for i, line in enumerate(lines):
        if line[0] == line[1] or vertices[line[0]] == vertices[line[1]]:
            raise ValueError(f"LINEDEFS {i}: zero-length line")
        if bool(line[2] & 4) != (line_sides[i][1] != -1):
            raise ValueError(f"LINEDEFS {i}: two-sided flag and back side disagree")

    front = [sides[row[0]][5] for row in line_sides]
    back = [sides[row[1]][5] if row[1] != -1 else -1 for row in line_sides]
    data = {
        "vertices": _tensor(vertices, torch.float32),
        "lines": _tensor([line[:2] for line in lines]),
        "line_front_sector": _tensor(front), "line_back_sector": _tensor(back),
        "line_flags": _tensor([line[2] for line in lines]),
        "line_special": _tensor([line[3] for line in lines]),
        "line_tag": _tensor([line[4] for line in lines]),
        "sector_floor": _tensor([sector[0] for sector in sectors], torch.float32),
        "sector_ceiling": _tensor([sector[1] for sector in sectors], torch.float32),
        "sector_light": _tensor([sector[4] for sector in sectors], torch.float32),
        "sector_special": _tensor([sector[5] for sector in sectors]),
        "sector_tag": _tensor([sector[6] for sector in sectors]),
    }

    _check_indices([v for seg in segs for v in seg[:2]], len(vertices), "SEGS vertex")
    _check_indices([seg[3] for seg in segs], len(lines), "SEGS linedef")
    _check_indices([seg[4] for seg in segs], 2, "SEGS side")
    seg_sectors = []
    for seg in segs:
        side = line_sides[seg[3]][seg[4]]
        if side == -1:
            raise ValueError("SEGS references an absent linedef side")
        seg_sectors.append(sides[side][5])
    subsector_sectors = []
    for count, start in subsectors:
        if count == 0 or start + count > len(segs):
            raise ValueError("SSECTORS references an empty or out-of-range segment span")
        child_sectors = seg_sectors[start:start + count]
        if len(set(child_sectors)) != 1:
            raise ValueError("SSECTORS spans more than one sector")
        subsector_sectors.append(child_sectors[0])
    children = [[-1 - (child & 32767) if child & 32768 else child
                 for child in node[12:14]] for node in nodes]
    visited = set()
    leaves = set()

    def depth(index: int) -> int:
        if index < 0:
            leaf = -1 - index
            if not 0 <= leaf < len(subsectors) or leaf in leaves:
                raise ValueError("BSP references an invalid or repeated subsector")
            leaves.add(leaf)
            return 0
        if not 0 <= index < len(nodes) or index in visited:
            raise ValueError("BSP references an invalid, repeated, or cyclic node")
        visited.add(index)
        return 1 + max(depth(child) for child in children[index])

    bsp_depth = depth(len(nodes) - 1)
    if len(visited) != len(nodes) or len(leaves) != len(subsectors):
        raise ValueError("BSP contains unreachable nodes or subsectors")
    data.update({
        "node_origin": _tensor([node[:2] for node in nodes], torch.float32),
        "node_weight": _tensor([[-node[3], node[2]] for node in nodes], torch.float32),
        "node_bias": _tensor([node[3] * node[0] - node[2] * node[1]
                              for node in nodes], torch.float32),
        "node_children": _tensor(children), "subsector_sector": _tensor(subsector_sectors),
        "bsp_root": _tensor(len(nodes) - 1), "bsp_depth": _tensor(bsp_depth),
    })

    definitions = wad.texture_definitions()
    texture_names = {name_of(side[slot]) for side in sides for slot in (2, 3, 4)} - {"-"}
    texture_names.add("SKY1")
    # Retain each original switch's other state for the fixed runtime graph.
    texture_names.update("SW2" + name[3:] for name in tuple(texture_names) if name.startswith("SW1"))
    texture_names = sorted(texture_names)
    texture_ids = {name: i for i, name in enumerate(texture_names)}
    texture_ids["-"] = -1
    data["texture_rgba"], data["texture_size"] = _atlas(
        [wad.texture(name, definitions) for name in texture_names]
    )
    data["line_texture"] = _tensor([
        [[texture_ids[name_of(sides[side][slot])] for slot in (2, 3, 4)]
         if side != -1 else [-1, -1, -1] for side in row] for row in line_sides
    ])
    data["line_offset"] = _tensor([
        [sides[side][:2] if side != -1 else (0, 0) for side in row] for row in line_sides
    ], torch.float32)
    flat_names = sorted({name_of(sector[slot]) for sector in sectors for slot in (2, 3)})
    flat_ids = {name: i for i, name in enumerate(flat_names)}
    data["flat_rgba"] = torch.stack([wad.flat(name) for name in flat_names])
    data["sector_floor_tex"] = _tensor([flat_ids[name_of(sector[2])] for sector in sectors])
    data["sector_ceiling_tex"] = _tensor([flat_ids[name_of(sector[3])] for sector in sectors])
    data["sector_sky"] = _tensor([name_of(sector[3]) == "F_SKY1" for sector in sectors], torch.bool)
    data["sky_texture"] = _tensor(texture_ids["SKY1"])

    starts = [thing for thing in things if thing[3] == 1]
    if len(starts) != 1:
        raise ValueError("E1M1 must have exactly one player-one start")
    data["player_start"] = _tensor([starts[0][0], starts[0][1],
                                    math.radians(starts[0][2])], torch.float32)
    skill_flag = (1, 1, 2, 4, 4)[skill]
    entities = [thing for thing in things
                if thing[3] not in (1, 2, 3, 4, 11)
                and thing[4] & skill_flag and not thing[4] & 16]
    unsupported = sorted({thing[3] for thing in entities} - ENEMIES.keys() - THING_PATCH.keys())
    if unsupported:
        raise ValueError(f"Unsupported E1M1 thing types: {unsupported}")
    data["entity_pos"] = _tensor([thing[:2] for thing in entities], torch.float32).reshape(-1, 2)
    data["entity_type"] = _tensor([thing[3] for thing in entities])
    data["entity_angle"] = _tensor([math.radians(thing[2]) for thing in entities], torch.float32)
    data["entity_flags"] = _tensor([thing[4] for thing in entities])
    sprites = _Sprites(wad)
    default, walk, attack, dead, drops = [], [], [], [], []
    for thing in entities:
        kind = thing[3]
        if kind in ENEMIES:
            prefix, death_frame = ENEMIES[kind]
            walks = [sprites.rotations(prefix, frame) for frame in "ABCD"]
            attacks = sprites.rotations(prefix, "E")
            corpse = sprites.rotations(prefix, death_frame)[0]
            drop = sprites.add(THING_PATCH[2007 if kind == 3004 else 2001]) if kind != 3001 else -1
        else:
            base = sprites.add(THING_PATCH[kind])
            walks, attacks, corpse, drop = [[base] * 8] * 4, [base] * 8, base, -1
            if kind == 2035:
                corpse = sprites.add("BEXPE0")
        default.append(walks[0][0])
        walk.append(walks)
        attack.append(attacks)
        dead.append(corpse)
        drops.append(drop)
    data.update({
        "entity_sprite": _tensor(default), "entity_walk_sprite": _tensor(walk).reshape(-1, 4, 8),
        "entity_attack_sprite": _tensor(attack).reshape(-1, 8),
        "entity_dead_sprite": _tensor(dead), "entity_drop_sprite": _tensor(drops),
        "weapon_sprite": _tensor([sprites.add(name) for name in ("PUNGA0", "PISGA0", "SHTGA0")]),
        "weapon_fire_sprite": _tensor([sprites.add(name) for name in ("PUNGC0", "PISGB0", "SHTGA0")]),
        "weapon_flash_sprite": _tensor([-1, sprites.add("PISFA0"), sprites.add("SHTFA0")]),
        "hud_face_sprite": _tensor([sprites.add(f"STFST{i}0") for i in range(5)]),
        "hud_digit_sprite": _tensor([sprites.add(f"STTNUM{i}") for i in range(10)]),
        "hud_bar_sprite": _tensor(sprites.add("STBAR")),
        "hud_percent_sprite": _tensor(sprites.add("STTPRCNT")),
        "hud_dead_face_sprite": _tensor(sprites.add("STFDEAD0")),
        "projectile_sprite": _tensor(sprites.add("BAL1A0")),
    })
    data["sprite_rgba"], data["sprite_size"] = _atlas([patch.rgba for patch in sprites.patches])
    data["sprite_offset"] = _tensor([[patch.left, patch.top] for patch in sprites.patches])
    metadata = {
        "level": "E1M1", "level_name": "Hangar", "skill": skill, "skill_name": SKILL_NAMES[skill],
        "wad": {"filename": Path(wad_path).name, "size": len(wad.raw), "md5": SHAREWARE_MD5,
                "sha256": hashlib.sha256(wad.raw).hexdigest(), "source_url": SHAREWARE_URL},
        "counts": {"vertices": len(vertices), "lines": len(lines), "sides": len(sides),
                   "sectors": len(sectors), "segments": len(segs), "subsectors": len(subsectors),
                   "nodes": len(nodes), "raw_things": len(things), "entities": len(entities),
                   "enemies": sum(thing[3] in ENEMIES for thing in entities),
                   "textures": len(texture_names), "flats": len(flat_names), "sprites": len(sprites.names)},
        "texture_names": texture_names, "flat_names": flat_names, "sprite_names": sprites.names,
        "entity_types": [thing[3] for thing in entities],
        "thing_counts": {str(kind): count for kind, count in sorted(Counter(thing[3] for thing in entities).items())},
        "line_specials": [{"line": i, "special": line[3], "tag": line[4],
                           "front_sector": front[i], "back_sector": back[i]}
                          for i, line in enumerate(lines) if line[3]],
        "sector_specials": [{"sector": i, "special": sector[5], "tag": sector[6]}
                            for i, sector in enumerate(sectors) if sector[5]],
        "map_bounds": {"min": [min(v[i] for v in vertices) for i in (0, 1)],
                       "max": [max(v[i] for v in vertices) for i in (0, 1)]},
        "bsp": {"root": len(nodes) - 1, "depth": bsp_depth,
                "branch_convention": (
                    "affine < 0: child 0; affine > 0: child 1; affine == 0: child 0 "
                    "for horizontal dx > 0, vertical dy < 0, or NW diagonal "
                    "(dx < 0, dy > 0) exactly at node_origin; else child 1 "
                    "(original R_PointOnSide); leaf=-1-child"
                )},
        "artwork": {"texture_slots": ["upper", "lower", "middle"],
                    "size_order": ["width", "height"], "sprite_offset_order": ["left", "top"],
                    "palette": "PLAYPAL palette 0", "rescaled": False},
        "notices": [{"owner": "id Software", "content": "Original Doom shareware 1.9 map and artwork",
                     "terms_source": SHAREWARE_README_URL,
                     "note": "WAD and extracted artwork are external assets, excluded from this source repository."}],
        "format_sources": ["https://github.com/id-Software/DOOM/blob/master/linuxdoom-1.10/doomdata.h",
                           "https://github.com/id-Software/DOOM/blob/master/linuxdoom-1.10/r_data.c",
                           "https://github.com/id-Software/DOOM/blob/master/linuxdoom-1.10/info.c"],
    }
    return data, metadata
