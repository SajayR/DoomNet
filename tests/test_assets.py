"""Format validation and fidelity checks for original Doom asset preparation."""

from collections import Counter
import json
from pathlib import Path
import struct

import pytest
import torch

from doomnet.assets import build_assets
from doomnet.wad import Wad, decode_patch, records, verify_shareware
from scripts.fetch_shareware import fetch_shareware


def make_wad(lumps):
    blob = bytearray(b"\0" * 12)
    entries = []
    for name, raw in lumps:
        entries.append((len(blob), len(raw), name.encode().ljust(8, b"\0")))
        blob.extend(raw)
    directory = len(blob)
    for entry in entries:
        blob.extend(struct.pack("<ii8s", *entry))
    struct.pack_into("<4sii", blob, 0, b"IWAD", len(entries), directory)
    return bytes(blob)


def make_patch(width, height, columns, left=0, top=0):
    assert len(columns) == width
    blob = bytearray(struct.pack("<hhhh", width, height, left, top) + b"\0" * (width * 4))
    for x, posts in enumerate(columns):
        struct.pack_into("<I", blob, 8 + x * 4, len(blob))
        for y, pixels in posts:
            blob.extend(bytes([y, len(pixels), 123, *pixels, 234]))
        blob.append(255)
    return bytes(blob)


@pytest.fixture
def palette():
    return torch.arange(256, dtype=torch.uint8)[:, None].repeat(1, 3)


def test_directory_lookup_obeys_wad_order_and_checks_bounds():
    wad = Wad(make_wad([("THING", b"first"), ("THING", b"second")]))
    assert wad.lump("thing") == b"second"
    with pytest.raises(ValueError, match="missing required lump"):
        wad.lump("MISSING")
    broken = bytearray(wad.raw)
    directory = struct.unpack_from("<i", broken, 8)[0]
    struct.pack_into("<i", broken, directory, len(broken) + 1)
    with pytest.raises(ValueError, match="lump 0 lies outside"):
        Wad(bytes(broken))
    with pytest.raises(ValueError, match="directory lies outside"):
        Wad(struct.pack("<4sii", b"IWAD", 1, 12))
    with pytest.raises(ValueError, match="Invalid WAD identification"):
        Wad(struct.pack("<4sii", b"NOPE", 0, 12))
    with pytest.raises(ValueError, match="not a multiple"):
        records(b"123", "<hh", "VERTEXES")


def test_column_posts_keep_transparency_palette_255_and_offsets(palette):
    raw = make_patch(2, 4, [[(0, [1, 255]), (3, [2])], [(1, [3])]], left=-4, top=17)
    patch = decode_patch(raw, palette)
    assert (patch.width, patch.height, patch.left, patch.top) == (2, 4, -4, 17)
    assert patch.rgba[..., 3].tolist() == [[255, 0], [255, 255], [0, 0], [255, 0]]
    assert patch.rgba[1, 0].tolist() == [255, 255, 255, 255]
    assert patch.rgba[2, 0].tolist() == [0, 0, 0, 0]
    assert patch.rgba[3, 0].tolist() == [2, 2, 2, 255]


def test_malformed_patch_offsets_posts_and_termination_fail(palette):
    raw = bytearray(make_patch(1, 2, [[(0, [1, 2])]]))
    with pytest.raises(ValueError, match="unterminated column"):
        decode_patch(bytes(raw[:-1]), palette)
    struct.pack_into("<I", raw, 8, len(raw))
    with pytest.raises(ValueError, match="offset lies outside"):
        decode_patch(bytes(raw), palette)
    raw = make_patch(1, 2, [[(1, [1, 2])]])
    with pytest.raises(ValueError, match="post extends outside"):
        decode_patch(raw, palette)


def test_texture_composition_clips_origins_and_preserves_lower_layers(palette):
    base = make_patch(2, 2, [[(0, [1, 1])], [(0, [1, 1])]])
    overlay = make_patch(2, 2, [[(1, [2])], [(0, [3])]])
    pnames = struct.pack("<I8s8s", 2, b"BASE", b"OVERLAY")
    definition = struct.pack("<8sihhih", b"WALL", 0, 3, 2, 0, 2)
    definition += struct.pack("<hhhhh", -1, 0, 0, 0, 0)
    definition += struct.pack("<hhhhh", 0, 0, 1, 0, 0)
    textures = struct.pack("<II", 1, 8) + definition
    wad = Wad(make_wad([("PLAYPAL", bytes(palette.flatten().tolist())),
                        ("PNAMES", pnames), ("TEXTURE1", textures),
                        ("BASE", base), ("OVERLAY", overlay)]))
    image = wad.texture("WALL")
    assert image[..., 0].tolist() == [[1, 3, 0], [2, 0, 0]]
    assert image[..., 3].tolist() == [[255, 255, 0], [255, 0, 0]]
    broken = bytearray(textures)
    struct.pack_into("<h", broken, 8 + 22 + 4, 2)
    malformed = Wad(make_wad([("PNAMES", pnames), ("TEXTURE1", bytes(broken))]))
    with pytest.raises(ValueError, match="patch index 2 is out of range"):
        malformed.texture_definitions()


def test_fetch_rejects_invalid_existing_file_and_bad_download(tmp_path):
    destination = tmp_path / "doom1.wad"
    destination.write_bytes(b"incorrect cached content")
    with pytest.raises(ValueError, match="Expected unmodified"):
        fetch_shareware(destination, "https://example.invalid/unreachable")
    assert destination.read_bytes() == b"incorrect cached content"
    source = tmp_path / "not-shareware.wad"
    source.write_bytes(b"incorrect download")
    output = tmp_path / "new.wad"
    with pytest.raises(ValueError, match="Expected unmodified"):
        fetch_shareware(output, source.as_uri())
    assert not output.exists()
    with pytest.raises(ValueError, match="Expected unmodified"):
        verify_shareware(b"incorrect data")


@pytest.fixture(scope="module")
def shareware():
    path = Path(__file__).resolve().parents[1] / "assets" / "doom1.wad"
    if not path.exists():
        pytest.skip("Fetch the original WAD with python scripts/fetch_shareware.py")
    return path


@pytest.fixture(scope="module")
def compiled(shareware):
    return build_assets(shareware)


def test_original_level_counts_geometry_and_interactions(compiled):
    data, metadata = compiled
    assert metadata["counts"] == {
        "vertices": 467, "lines": 475, "sides": 648, "sectors": 85,
        "segments": 732, "subsectors": 237, "nodes": 236, "raw_things": 138,
        "entities": 91, "enemies": 6, "textures": 34, "flats": 22, "sprites": 127,
    }
    assert data["player_start"][:2].tolist() == [1056, -3616]
    assert float(data["player_start"][2]) == pytest.approx(1.5707963267948966)
    assert data["sector_floor"][[59, 70]].tolist() == [96, 104]
    assert data["sector_tag"][[59, 70]].tolist() == [1, 2]
    assert data["line_special"][330].item() == 11
    assert Counter(data["line_special"].tolist()) == {0: 456, 1: 8, 11: 1, 36: 1, 48: 8, 88: 1}
    assert data["bsp_root"].item() == 235
    assert data["bsp_depth"].item() == 18
    assert data["node_origin"].shape == (236, 2)
    assert data["node_origin"].dtype == torch.float32
    assert data["node_origin"][[0, 235]].tolist() == [[1552, -2432], [2176, -3776]]
    assert metadata["wad"]["sha256"] == "1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771"
    json.dumps(metadata)
    assert all(isinstance(value, torch.Tensor) and not value.requires_grad for value in data.values())


def test_medium_entities_preserve_wad_skill_ambush_and_singleplayer_flags(compiled, shareware):
    data, metadata = compiled
    assert Counter(data["entity_type"].tolist())[3004] == 4
    assert Counter(data["entity_type"].tolist())[3001] == 2
    assert 9 not in data["entity_type"]
    assert not bool(((data["entity_flags"] & 16) != 0).any())
    assert bool(((data["entity_flags"] & 2) != 0).all())
    assert not any(kind in (1, 2, 3, 4, 11) for kind in metadata["entity_types"])
    hard, hard_metadata = build_assets(shareware, skill=3)
    assert Counter(hard["entity_type"].tolist())[9] == 16
    assert hard_metadata["counts"]["enemies"] == 29
    for skill in (-1, 5, True, 2.0):
        with pytest.raises(ValueError, match="skill must be"):
            build_assets(shareware, skill)


def test_original_artwork_keeps_native_sizes_and_mirrored_origins(compiled, shareware):
    data, metadata = compiled
    wad = Wad(shareware)
    sky = data["sky_texture"].item()
    assert metadata["texture_names"][sky] == "SKY1"
    assert data["texture_size"][sky].tolist() == [256, 128]
    assert tuple(data["texture_rgba"].shape[1:]) == (128, 256, 4)
    assert tuple(data["sprite_rgba"].shape[1:]) == (128, 320, 4)
    bar = data["hud_bar_sprite"].item()
    assert data["sprite_size"][bar].tolist() == [320, 32]
    assert metadata["sprite_names"][bar] == "STBAR"
    assert torch.equal(data["sprite_rgba"][bar, :32, :320], wad.patch("STBAR").rgba)
    mirrored = metadata["sprite_names"].index("POSSA2A8~mirror")
    patch = wad.patch("POSSA2A8")
    assert torch.equal(data["sprite_rgba"][mirrored, :patch.height, :patch.width], patch.rgba.flip(1))
    assert data["sprite_offset"][mirrored].tolist() == [patch.width - patch.left, patch.top]
    assert data["entity_walk_sprite"].shape == (91, 4, 8)
    assert data["entity_drop_sprite"][[5, 6, 56, 57]].ge(0).all()
    assert data["entity_drop_sprite"][[4, 8]].tolist() == [-1, -1]
