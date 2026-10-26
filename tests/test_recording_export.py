"""Recording media export preserves completed RGB pixels and host timing."""

import hashlib
import json

from PIL import Image
import pytest

from scripts.export_recording import export_recording


def recording(tmp_path, rows):
    directory = tmp_path / "capture"
    directory.mkdir()
    metadata = {"artifact_files": {"model.pt": {"sha256": "a" * 64},
                                    "initial_state.pt": {"sha256": "b" * 64}},
                "contract": {"width": 3, "height": 2},
                "notes": "Test recording provenance, not a claim of human play."}
    (directory / "recording.json").write_text(json.dumps(metadata))
    (directory / "inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for row in rows:
        if "image" in row:
            pixels = bytes((row["index"] * 29 + i * 17) % 256 for i in range(18))
            Image.frombytes("RGB", (3, 2), pixels).save(directory / row["image"])
    return directory


def image_row(index, elapsed):
    return {"index": index, "wall_seconds": elapsed, "image": f"frame_{index:06d}.png"}


def test_export_retains_exact_rgb_wall_time_and_provenance(tmp_path):
    directory = recording(tmp_path, [image_row(1, 0.001), {"index": 2, "wall_seconds": 0.02},
                                     image_row(3, 0.1134), image_row(4, 0.4041),
                                     {"index": 5, "wall_seconds": 0.6059}])
    output = tmp_path / "demo.webp"
    metadata = export_recording(directory, output)
    assert metadata["input_count"] == 5
    assert metadata["frame_count"] == metadata["encoded_frame_count"] == 3
    assert metadata["duration_seconds"] == pytest.approx(0.6049)
    assert metadata["encoded_duration_ms"] == 605
    assert metadata["timing"]["frame_durations_ms"] == [112, 291, 202]
    assert metadata["source_artifact_hashes"] == {"model.pt": "a" * 64, "initial_state.pt": "b" * 64}
    assert metadata["source_recording_metadata"]["notes"] == "Test recording provenance, not a claim of human play."
    assert json.loads(output.with_suffix(".json").read_text()) == metadata
    with Image.open(output) as animation:
        assert animation.info["loop"] == 1
        for i, index in enumerate((1, 3, 4)):
            animation.seek(i)
            animation.load()
            with Image.open(directory / f"frame_{index:06d}.png") as original:
                assert animation.mode == "RGB"
                assert animation.tobytes() == original.tobytes()
            assert animation.info["duration"] == (112, 291, 202)[i]
    for name, digest in metadata["source_file_hashes"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    assert metadata["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_final_image_at_last_input_retains_zero_duration(tmp_path):
    directory = recording(tmp_path, [image_row(1, 0.02), image_row(2, 0.32)])
    output = tmp_path / "last.webp"
    metadata = export_recording(directory, output)
    assert metadata["timing"]["frame_durations_ms"] == [300, 0]
    with Image.open(output) as animation:
        animation.seek(1)
        animation.load()
        assert animation.info["duration"] == 0


@pytest.mark.parametrize("rows,match", [
    ([], "no inputs"),
    ([{"index": 1, "wall_seconds": 0.1}], "no recorded RGB frames"),
    ([image_row(1, 0.1), image_row(2, 0.1)], "nonmonotonic"),
    ([image_row(1, 0.2), image_row(2, 0.1)], "nonmonotonic"),
    ([image_row(1, 0.1), image_row(2, float("nan"))], "invalid wall_seconds"),
    ([image_row(1, 0.1)], "at least two distinct"),
])
def test_invalid_recording_fails_without_output(tmp_path, rows, match):
    directory = recording(tmp_path, rows)
    output = tmp_path / "invalid.webp"
    with pytest.raises(ValueError, match=match):
        export_recording(directory, output)
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_missing_frame_and_source_overwrite_fail(tmp_path):
    directory = recording(tmp_path, [image_row(1, 0.1), image_row(2, 0.2)])
    output = tmp_path / "missing.webp"
    (directory / "frame_000002.png").unlink()
    with pytest.raises(ValueError, match="Missing recorded frame"):
        export_recording(directory, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="overwrite the source"):
        export_recording(directory, directory / "recording.webp")


def test_identical_images_are_not_presented_as_animation(tmp_path):
    directory = recording(tmp_path, [image_row(1, 0.1), image_row(2, 0.2)])
    (directory / "frame_000002.png").write_bytes((directory / "frame_000001.png").read_bytes())
    with pytest.raises(ValueError, match="at least two distinct"):
        export_recording(directory, tmp_path / "unchanged.webp")
