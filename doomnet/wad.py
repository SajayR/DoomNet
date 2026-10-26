"""Read vanilla Doom WAD records and decode their original indexed artwork.

This is parameter preparation only; no WAD parsing runs in the exported game.
Record layouts follow id Software's ``doomdata.h`` and ``r_data.c``.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import torch


SHAREWARE_MD5 = "f0cefca49926d00903cf57551d901abe"
SHAREWARE_URL = (
    "https://raw.githubusercontent.com/Doom-Utils/shareware-collection/"
    "6591a922701e5769b32278e9c555b1bf93f5714c/Doom%201.9/doom1.wad"
)
SHAREWARE_README_URL = SHAREWARE_URL.removesuffix("doom1.wad") + "readme.txt"
MAP_LUMPS = (
    "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS", "SSECTORS",
    "NODES", "SECTORS", "REJECT", "BLOCKMAP",
)


def verify_shareware(raw: bytes) -> None:
    digest = hashlib.md5(raw).hexdigest()
    if digest != SHAREWARE_MD5:
        raise ValueError(
            "Expected unmodified Doom shareware 1.9 doom1.wad "
            f"(MD5 {SHAREWARE_MD5}), received {digest}"
        )


def name_of(raw: bytes) -> str:
    try:
        return raw.split(b"\0", 1)[0].decode("ascii").upper()
    except UnicodeDecodeError as error:
        raise ValueError("Non-ASCII WAD name") from error


def records(raw: bytes, fmt: str, label: str) -> list[tuple]:
    size = struct.calcsize(fmt)
    if len(raw) % size:
        raise ValueError(f"{label}: {len(raw)} bytes is not a multiple of {size}")
    return list(struct.iter_unpack(fmt, raw))


@dataclass(frozen=True)
class Lump:
    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class Patch:
    rgba: torch.Tensor
    left: int
    top: int

    @property
    def width(self) -> int:
        return self.rgba.shape[1]

    @property
    def height(self) -> int:
        return self.rgba.shape[0]


def decode_patch(raw: bytes, palette: torch.Tensor, label: str = "patch") -> Patch:
    """Decode vanilla column posts, retaining transparency and origin offsets."""
    if len(raw) < 8:
        raise ValueError(f"{label}: truncated patch header")
    width, height, left, top = struct.unpack_from("<hhhh", raw)
    header_size = 8 + width * 4
    if width <= 0 or height <= 0 or header_size > len(raw):
        raise ValueError(f"{label}: invalid patch dimensions or column directory")
    if palette.shape != (256, 3) or palette.dtype != torch.uint8:
        raise ValueError("Patch palette must be uint8 [256,3]")
    indices = torch.zeros((height, width), dtype=torch.uint8)
    alpha = torch.zeros((height, width), dtype=torch.uint8)
    for x in range(width):
        pos = struct.unpack_from("<I", raw, 8 + x * 4)[0]
        if not header_size <= pos < len(raw):
            raise ValueError(f"{label}: column {x} offset lies outside pixel data")
        while True:
            if pos >= len(raw):
                raise ValueError(f"{label}: unterminated column {x}")
            y = raw[pos]
            if y == 255:
                break
            if pos + 3 >= len(raw):
                raise ValueError(f"{label}: truncated post in column {x}")
            length = raw[pos + 1]
            end = pos + length + 4
            if end > len(raw) or y + length > height:
                raise ValueError(f"{label}: post extends outside patch")
            indices[y:y + length, x] = torch.tensor(
                list(raw[pos + 3:pos + 3 + length]), dtype=torch.uint8
            )
            alpha[y:y + length, x] = 255
            pos = end
    rgba = torch.cat((palette[indices.long()], alpha[..., None]), dim=-1)
    rgba[..., :3] *= (alpha != 0)[..., None]
    return Patch(rgba, left, top)


class Wad:
    """Validated WAD directory, with native last-lump-wins name lookup."""

    def __init__(self, source: bytes | str | Path):
        self.raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        if len(self.raw) < 12:
            raise ValueError("Truncated WAD header")
        kind, count, offset = struct.unpack_from("<4sii", self.raw)
        if kind not in (b"IWAD", b"PWAD"):
            raise ValueError(f"Invalid WAD identification {kind!r}")
        if count < 0 or offset < 12 or offset + count * 16 > len(self.raw):
            raise ValueError("WAD directory lies outside file")
        self.lumps = []
        self.lookup = {}
        for i in range(count):
            start, size, name = struct.unpack_from("<ii8s", self.raw, offset + i * 16)
            if start < 0 or size < 0 or start + size > len(self.raw):
                raise ValueError(f"WAD lump {i} lies outside file")
            lump = Lump(name_of(name), start, size)
            self.lookup[lump.name] = i
            self.lumps.append(lump)
        self._palette = None
        self._patches: dict[str, Patch] = {}

    def lump(self, name: str) -> bytes:
        key = name.upper()
        if key not in self.lookup:
            raise ValueError(f"WAD is missing required lump {key}")
        item = self.lumps[self.lookup[key]]
        return self.raw[item.offset:item.offset + item.size]

    def map_lumps(self, name: str) -> dict[str, bytes]:
        if name not in self.lookup:
            raise ValueError(f"WAD is missing map {name}")
        start = self.lookup[name] + 1
        entries = self.lumps[start:start + len(MAP_LUMPS)]
        if tuple(entry.name for entry in entries) != MAP_LUMPS:
            raise ValueError(f"{name}: missing or misordered map lumps")
        return {entry.name: self.raw[entry.offset:entry.offset + entry.size]
                for entry in entries}

    def palette(self) -> torch.Tensor:
        if self._palette is None:
            raw = self.lump("PLAYPAL")
            if not raw or len(raw) % 768:
                raise ValueError("PLAYPAL must contain complete 256-color RGB palettes")
            self._palette = torch.tensor(list(raw[:768]), dtype=torch.uint8).reshape(256, 3)
        return self._palette

    def patch(self, name: str) -> Patch:
        key = name.upper()
        if key not in self._patches:
            self._patches[key] = decode_patch(self.lump(key), self.palette(), key)
        return self._patches[key]

    def flat(self, name: str) -> torch.Tensor:
        raw = self.lump(name)
        if len(raw) != 4096:
            raise ValueError(f"{name}: Doom flat must contain exactly 4096 pixels")
        indices = torch.tensor(list(raw), dtype=torch.int64).reshape(64, 64)
        return torch.cat((self.palette()[indices],
                          torch.full((64, 64, 1), 255, dtype=torch.uint8)), dim=-1)

    def texture_definitions(self) -> dict[str, tuple[int, int, list[tuple[int, int, str]]]]:
        names_raw = self.lump("PNAMES")
        if len(names_raw) < 4:
            raise ValueError("PNAMES: truncated patch-name directory")
        count = struct.unpack_from("<I", names_raw)[0]
        if len(names_raw) != 4 + count * 8:
            raise ValueError("PNAMES: invalid patch-name directory length")
        names = [name_of(names_raw[4 + i * 8:12 + i * 8]) for i in range(count)]
        definitions = {}
        texture_lumps = ["TEXTURE1"] + (["TEXTURE2"] if "TEXTURE2" in self.lookup else [])
        for label in texture_lumps:
            raw = self.lump(label)
            if len(raw) < 4:
                raise ValueError(f"{label}: truncated texture directory")
            count = struct.unpack_from("<I", raw)[0]
            directory_end = 4 + count * 4
            if directory_end > len(raw):
                raise ValueError(f"{label}: texture directory lies outside lump")
            for i in range(count):
                pos = struct.unpack_from("<I", raw, 4 + i * 4)[0]
                if pos < directory_end or pos + 22 > len(raw):
                    raise ValueError(f"{label}: invalid texture {i} offset")
                name, _, width, height, _, patch_count = struct.unpack_from("<8sihhih", raw, pos)
                name = name_of(name)
                if width <= 0 or height <= 0 or patch_count <= 0 or pos + 22 + patch_count * 10 > len(raw):
                    raise ValueError(f"{label}/{name}: invalid texture dimensions or patch list")
                patches = []
                for j in range(patch_count):
                    x, y, patch_id, _, _ = struct.unpack_from("<hhhhh", raw, pos + 22 + j * 10)
                    if not 0 <= patch_id < len(names):
                        raise ValueError(f"{label}/{name}: patch index {patch_id} is out of range")
                    patches.append((x, y, names[patch_id]))
                if name in definitions:
                    raise ValueError(f"Duplicate texture definition {name}")
                definitions[name] = (width, height, patches)
        return definitions

    def texture(self, name: str, definitions: dict | None = None) -> torch.Tensor:
        definitions = self.texture_definitions() if definitions is None else definitions
        if name not in definitions:
            raise ValueError(f"WAD is missing texture definition {name}")
        width, height, parts = definitions[name]
        rgba = torch.zeros((height, width, 4), dtype=torch.uint8)
        for x, y, patch_name in parts:
            patch = self.patch(patch_name)
            left, top = max(x, 0), max(y, 0)
            right, bottom = min(x + patch.width, width), min(y + patch.height, height)
            if left < right and top < bottom:
                source = patch.rgba[top - y:bottom - y, left - x:right - x]
                target = rgba[top:bottom, left:right]
                target.copy_(torch.where(source[..., 3:4] != 0, source, target))
        return rgba
