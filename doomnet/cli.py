"""Reproducible construction, generic interactive runners, and graph inspection."""

from __future__ import annotations

import argparse
import json


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Play E1M1 through an audited, constructed neural artifact")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Lower complete gameplay/rendering to admitted neural primitives")
    build.add_argument("--wad", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--skill", type=int, choices=range(5), default=2)
    build.add_argument("--width", type=int, default=160)
    build.add_argument("--height", type=int, default=120)
    build.add_argument("--threads", type=int, default=1)
    build.add_argument("--device", default="cpu", help="Device for neural target tracing/verification: cpu or cuda[:index]")
    for name, help_text in (("play", "Present model RGB in a native window"),
                            ("serve", "Serve a browser canvas with live input"),
                            ("benchmark", "Measure the complete transition and RGB graph"),
                            ("inspect", "Verify v2 artifacts and display the exhaustive neural audit")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--model", required=True, help="Artifact directory containing model.pt")
        command.add_argument("--device", default="cpu", help="cpu or cuda[:index] (also ROCm)")
        command.add_argument("--threads", type=int, default=4)
        if name in ("play", "serve"):
            command.add_argument("--record", help="Empty directory for external controls and completed PNG frames")
        if name == "play":
            command.add_argument("--scale", type=int, default=4)
            command.add_argument("--mouse", action="store_true", help="Capture relative mouse movement")
        elif name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
        elif name == "benchmark":
            command.add_argument("--frames", type=int, default=200)
            command.add_argument("--warmup", type=int, default=5)
            command.add_argument("--out", help="Also save the JSON measurement")
        elif name == "inspect":
            command.add_argument("--graph", action="store_true", help="Print the entire recursively inlined graph")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "build":
        from .model import build_model
        build_model(args.wad, args.out, skill=args.skill, width=args.width, height=args.height,
                    threads=args.threads, device=args.device)
        return 0
    from .runtime import MODEL_METHODS, RuntimeModel, audit_model, benchmark, play, serve
    runner = RuntimeModel(args.model, device=args.device, threads=args.threads, record_dir=getattr(args, "record", None))
    try:
        if args.command == "play":
            play(runner, scale=args.scale, mouse=args.mouse)
        elif args.command == "serve":
            serve(runner, host=args.host, port=args.port)
        elif args.command == "benchmark":
            result = json.dumps(benchmark(runner, args.frames, args.warmup), indent=2)
            print(result)
            if args.out:
                from pathlib import Path
                Path(args.out).write_text(result + "\n")
        elif args.command == "inspect":
            print(json.dumps(runner.manifest, indent=2))
            if args.graph:
                _, texts = audit_model(runner.model, runner.manifest["contract"])
                for method in MODEL_METHODS:
                    print(f"Method: {method}\n{texts[method]}")
    except KeyboardInterrupt:
        return 130
    finally:
        runner.close()
    return 0
