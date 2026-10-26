"""Encode recorded model RGB images as a lossless animated WebP and audit JSON."""

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image


def export_recording(recording: str | Path, output: str | Path) -> dict:
    """Preserve wall-time image intervals, from first image to last input.

    WebP stores integer milliseconds. Rounding cumulative boundaries avoids
    accumulated timing drift. The final image ends at the last input timestamp,
    including a zero-duration final frame when both timestamps are identical.
    """
    recording, output = Path(recording), Path(output)
    sidecar = output.with_suffix(".json")
    if output.suffix.lower() != ".webp":
        raise ValueError("Output must have a .webp extension")
    if sidecar.resolve() == (recording / "recording.json").resolve():
        raise ValueError("Output metadata would overwrite the source recording.json")
    metadata_raw = (recording / "recording.json").read_bytes()
    source_metadata = json.loads(metadata_raw)
    if not isinstance(source_metadata, dict):
        raise ValueError("recording.json must contain a metadata object")
    artifacts = source_metadata.get("artifact_files")
    if not isinstance(artifacts, dict) or not all(name in artifacts for name in ("model.pt", "initial_state.pt")):
        raise ValueError("recording.json must identify model.pt and initial_state.pt artifact hashes")
    artifact_hashes = {}
    for name, info in artifacts.items():
        digest = info.get("sha256") if isinstance(info, dict) else None
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"recording.json contains an invalid SHA256 for {name}")
        artifact_hashes[name] = digest
    contract = source_metadata.get("contract")
    if not isinstance(contract, dict) or any(
        isinstance(contract.get(key), bool) or not isinstance(contract.get(key), int)
        or contract[key] < 1 for key in ("width", "height")
    ):
        raise ValueError("recording.json must identify positive integer image width and height")
    size = (contract["width"], contract["height"])
    inputs_raw = (recording / "inputs.jsonl").read_bytes()
    rows = inputs_raw.splitlines()
    if not rows:
        raise ValueError("Recording contains no inputs")
    source_hashes = {"recording.json": hashlib.sha256(metadata_raw).hexdigest(),
                     "inputs.jsonl": hashlib.sha256(inputs_raw).hexdigest()}
    images, times = [], []
    previous_time, previous_index = -math.inf, -1
    for line_number, raw in enumerate(rows, 1):
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"inputs.jsonl line {line_number} must contain an input object")
        elapsed, index = row.get("wall_seconds"), row.get("index")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (float, int))
                or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError(f"inputs.jsonl line {line_number} has invalid wall_seconds")
        if elapsed <= previous_time:
            raise ValueError(f"inputs.jsonl line {line_number} has nonmonotonic wall_seconds")
        if isinstance(index, bool) or not isinstance(index, int) or index <= previous_index:
            raise ValueError(f"inputs.jsonl line {line_number} has a nonincreasing input index")
        previous_time, previous_index = elapsed, index
        if "image" in row:
            name = row["image"]
            if name != f"frame_{index:06d}.png":
                raise ValueError(f"inputs.jsonl line {line_number} has an invalid frame filename")
            path = recording / name
            if not path.is_file():
                raise ValueError(f"Missing recorded frame: {path}")
            pixels = path.read_bytes()
            source_hashes[name] = hashlib.sha256(pixels).hexdigest()
            with Image.open(BytesIO(pixels)) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != size or image.n_frames != 1:
                    raise ValueError(f"{name} must be one original RGB PNG of size {size}")
                images.append(image.copy())
            times.append(elapsed)
    if not images:
        raise ValueError("Recording contains no recorded RGB frames")
    boundaries = [round((elapsed - times[0]) * 1000) for elapsed in times]
    boundaries.append(round((previous_time - times[0]) * 1000))
    durations = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if any(duration > 16777215 for duration in durations):
        raise ValueError("A recorded image interval exceeds WebP's 24-bit millisecond duration limit")
    encoded = BytesIO()
    images[0].save(encoded, format="WEBP", save_all=True, append_images=images[1:],
                   duration=durations, lossless=True, loop=1, method=4)
    pixels = encoded.getvalue()
    with Image.open(BytesIO(pixels)) as animation:
        if not animation.is_animated:
            raise ValueError("An animated export requires at least two distinct recorded RGB frames")
        encoded_frames = animation.n_frames
    metadata = {
        "provenance": "Lossless encoding of recorded model RGB PNGs only; no overlays, resizing, or world state computation.",
        "source_artifact_hashes": artifact_hashes,
        "source_file_hashes": source_hashes,
        "source_recording_metadata": source_metadata,
        "input_count": len(rows), "frame_count": len(images), "encoded_frame_count": encoded_frames,
        "duration_seconds": previous_time - times[0], "encoded_duration_ms": sum(durations),
        "timing": {"first_image_wall_seconds": times[0], "last_input_wall_seconds": previous_time,
                   "image_wall_seconds": times, "frame_durations_ms": durations,
                   "rounding": "Nearest millisecond at cumulative frame boundaries; final frame ends at last input."},
        "output": {"filename": output.name, "sha256": hashlib.sha256(pixels).hexdigest(),
                   "bytes": len(pixels), "format": "animated lossless WebP", "loop_count": 1},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pixels)
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = export_recording(args.recording, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}: "
          f"{metadata['frame_count']} recorded frames, {metadata['encoded_duration_ms']} ms")


if __name__ == "__main__":
    main()
