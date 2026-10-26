"""Generic execution of audited, bounded, constructed neural artifacts.

This module has no knowledge of map geometry, simulation state fields, enemies,
weapons, or rendering. Its only model-specific information is the external
input/output contract declared in the artifact manifest.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import platform
import threading
import time
from typing import Sequence
from types import SimpleNamespace

import torch

from .neural_ir import audit_neural_graph

MAX_DT = 1.0 / 35.0
CONTROL_NAMES = ("forward", "strafe", "turn", "fire", "use", "restart", "weapon", "run")
NO_CONTROLS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
ARTIFACT_VERSION = 2
MODEL_METHODS = ("forward", "transition", "render")


def execution_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda[:index]")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("The requested CUDA/ROCm device is unavailable")
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_input_shapes(state_shape: Sequence[int]) -> dict:
    transition = (tuple(state_shape), (8,), (1,))
    return {"forward": transition, "transition": transition, "render": (tuple(state_shape),)}


def artifact_contract(state_shape: Sequence[int], width: int, height: int) -> dict:
    dimensions = (width, height, *state_shape) if isinstance(state_shape, (list, tuple, torch.Size)) else ()
    if (len(dimensions) != 3 or any(isinstance(value, bool) or not isinstance(value, int)
                                  or value < 1 for value in dimensions)):
        raise ValueError("Artifact must declare positive output dimensions and a nonempty flat state")
    return {"state_shape": list(state_shape), "state_dtype": "float32",
            "state_bounds_shape": [2, *state_shape], "state_bounds_dtype": "float32",
            "controls": list(CONTROL_NAMES), "control_ranges": [[-1, 1]] * 3 + [[0, 1]] * 3 + [[-1, 2], [0, 1]],
            "controls_dtype": "float32", "weapon_values": [-1, 0, 1, 2],
            "dt_shape": [1], "dt_dtype": "float32", "max_dt": MAX_DT,
            "width": width, "height": height, "frame_shape": [height, width, 3], "frame_dtype": "uint8"}


def audit_model(model: torch.jit.ScriptModule, contract: dict) -> tuple[dict, dict]:
    """Exhaustively admit all methods and verify their declared tensor interface."""
    if set(model._c._method_names()) != set(MODEL_METHODS):
        raise ValueError("Neural artifact must export exactly forward, transition, and render")
    shapes = method_input_shapes(contract["state_shape"])
    state = {"shape": list(contract["state_shape"]), "dtype": "torch.float32", "codec": None}
    frame = {"shape": [contract["height"], contract["width"], 3], "dtype": "torch.uint8", "codec": "uint8"}
    expected = {"forward": [state, frame], "transition": [state], "render": [frame]}
    audits, texts = {}, {}
    for method in MODEL_METHODS:
        audit, text = audit_neural_graph(model, method, input_shapes=shapes[method],
                                         input_dtypes=[torch.float32] * len(shapes[method]))
        if audit["outputs"] != expected[method]:
            raise ValueError(f"Neural {method} outputs do not match the artifact tensor contract")
        audits[method], texts[method] = audit, text
    return audits, texts


def validate_state_bounds(bounds, state_shape: Sequence[int]):
    if not isinstance(bounds, torch.Tensor) or bounds.dtype != torch.float32 or tuple(bounds.shape) != (2, *state_shape):
        raise ValueError("State bounds must be a float32 tensor with shape [2,state_size]")
    if not torch.isfinite(bounds).all().item() or not (bounds[0] <= bounds[1]).all().item():
        raise ValueError("State bounds must be finite, ordered closed intervals")


def validate_state(state, state_shape: Sequence[int], lower: torch.Tensor, upper: torch.Tensor):
    if not isinstance(state, torch.Tensor) or tuple(state.shape) != tuple(state_shape) or state.dtype != torch.float32:
        raise ValueError("Model state must match the flat float32 tensor contract")
    if not torch.isfinite(state).all().item():
        raise ValueError("Model state contains non-finite activations")
    outside = (state < lower) | (state > upper)
    if outside.any().item():
        index = int(torch.nonzero(outside).flatten()[0].item())
        raise ValueError(f"State activation {index}={state[index].item()} violates declared bounds [{lower[index].item()},{upper[index].item()}]")


def neural_executor(model: torch.jit.ScriptModule, shapes: dict, device: torch.device):
    """Evaluate admitted NN graphs; accelerator fusion has no fallback path."""
    if device.type == "cpu":
        return SimpleNamespace(**{method: getattr(model, method) for method in MODEL_METHODS})
    from .neural_ir import neural_method_fx
    functions = {}
    for method in MODEL_METHODS:
        module = neural_method_fx(model, method, input_shapes=shapes[method],
                                  input_dtypes=[torch.float32] * len(shapes[method])).to(device).eval()
        functions[method] = torch.compile(module, fullgraph=True)
    return SimpleNamespace(**functions)


def execution_info(device: torch.device, threads: int) -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "system": platform.platform(), "architecture": platform.machine(),
            "device": str(device), "cpu_threads": threads,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.partition(":")[2].strip()
                break
    if device.type == "cuda":
        info["accelerator"] = torch.cuda.get_device_name(device)
    return info


class Recorder:
    """Record only external inputs and completed images, never world state."""

    def __init__(self, directory: str | Path, manifest: dict, info: dict, fps: float = 10.0):
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError(f"Recording directory must be empty: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.last_image = -math.inf
        self.interval = 1.0 / fps
        self.inputs = (self.directory / "inputs.jsonl").open("w")
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "provenance": "Live host controls and completed frozen-model RGB outputs; no state fields inspected or changed.",
                    "artifact_files": manifest["files"], "contract": manifest["contract"],
                    "execution": info, "image_fps_limit": fps}
        (self.directory / "recording.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def record(self, index: int, controls: torch.Tensor, dt: float, frame: torch.Tensor | None = None):
        elapsed = time.perf_counter() - self.started
        row = {"index": index, "wall_seconds": elapsed,
               "controls": controls.detach().cpu().tolist(), "dt": dt}
        if frame is not None and elapsed - self.last_image >= self.interval:
            from PIL import Image
            name = f"frame_{index:06d}.png"
            Image.fromarray(frame.detach().cpu().numpy()).save(self.directory / name)
            row["image"] = name
            self.last_image = elapsed
        self.inputs.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.inputs.flush()

    def close(self):
        self.inputs.close()


class RuntimeModel:
    """Load immutable artifacts, validate external inputs, and carry returned state."""

    def __init__(self, directory: str | Path, device: str = "cpu", threads: int = 4,
                 record_dir: str | Path | None = None):
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.directory = Path(directory)
        self.device = execution_device(device)
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(threads)
        self.threads = threads
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("Unsupported artifact version; rebuild with the current compiler")
        for name in ("model.pt", "initial_state.pt", "state_bounds.pt"):
            actual = sha256_file(self.directory / name)
            if actual != self.manifest["files"][name]["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        contract = self.manifest["contract"]
        expected = artifact_contract(contract["state_shape"], contract["width"], contract["height"])
        if contract != expected:
            raise ValueError("Artifact tensor/control/timing contract does not match this runner")
        self.width, self.height = contract["width"], contract["height"]
        self.state_shape = tuple(contract["state_shape"])
        self.model = torch.jit.load(str(self.directory / "model.pt"), map_location=self.device).eval()
        self.method_shapes = method_input_shapes(self.state_shape)
        self.audits, _ = audit_model(self.model, contract)
        if self.audits != self.manifest["methods"]:
            raise ValueError("Strict neural audit does not match the artifact manifest")
        if any(audit["neural_ir_version"] != self.manifest["neural_ir_version"] for audit in self.audits.values()):
            raise ValueError("Neural IR version does not match the artifact manifest")
        bounds = torch.load(self.directory / "state_bounds.pt", map_location=self.device, weights_only=True)
        validate_state_bounds(bounds, self.state_shape)
        self.state_lower, self.state_upper = bounds.unbind(0)
        # Match returned recurrent tensors before Dynamo specializes its guards.
        with torch.inference_mode():
            self.state = torch.load(self.directory / "initial_state.pt", map_location=self.device, weights_only=True)
        self._check_state(self.state)
        self.executor = neural_executor(self.model, self.method_shapes, self.device)
        self.index = 0
        self.frame: torch.Tensor | None = None
        self.info = execution_info(self.device, threads)
        self.info["executor"] = "inductor_fullgraph" if self.device.type == "cuda" else "torchscript"
        self.recorder = Recorder(record_dir, self.manifest, self.info) if record_dir is not None else None

    def _check_state(self, state):
        validate_state(state, self.state_shape, self.state_lower, self.state_upper)

    def _inputs(self, controls: Sequence[float] | torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or not 0 <= dt <= MAX_DT:
            raise ValueError(f"dt must be finite and between 0 and {MAX_DT}")
        try:
            # Validate the supplied precision before casting to network float32;
            # otherwise 0.999999999 could silently become weapon code 1.
            tensor = torch.as_tensor(controls, dtype=torch.float64, device="cpu").detach()
        except (TypeError, ValueError) as error:
            raise ValueError("controls must be eight numeric values") from error
        if tuple(tensor.shape) != (8,) or not torch.isfinite(tensor).all().item():
            raise ValueError("controls must be eight finite numeric values")
        if (tensor[:3].abs() > 1).any().item() or (tensor[[3, 4, 5, 7]] < 0).any().item() or (tensor[[3, 4, 5, 7]] > 1).any().item():
            raise ValueError("movement and turn controls must be in [-1,1]; buttons must be in [0,1]")
        if tensor[6].item() not in (-1.0, 0.0, 1.0, 2.0):
            raise ValueError("weapon control must be -1, 0, 1, or 2")
        return tensor.to(device=self.device, dtype=torch.float32), torch.tensor([dt], dtype=torch.float32, device=self.device)

    def _check_frame(self, frame):
        if not isinstance(frame, torch.Tensor) or frame.dtype != torch.uint8 or tuple(frame.shape) != (self.height, self.width, 3):
            raise ValueError("Model image must match the manifest's uint8 RGB tensor contract")

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def step(self, controls: Sequence[float] | torch.Tensor = NO_CONTROLS, dt: float = MAX_DT) -> torch.Tensor:
        control_tensor, dt_tensor = self._inputs(controls, dt)
        self._check_state(self.state)
        with torch.inference_mode():
            state, frame = self.executor.forward(self.state, control_tensor, dt_tensor)
        self._check_state(state)
        self._check_frame(frame)
        self.state, self.frame = state, frame
        self.index += 1
        if self.recorder is not None:
            self.recorder.record(self.index, control_tensor, float(dt), frame)
        return frame

    def advance(self, controls: Sequence[float] | torch.Tensor = NO_CONTROLS,
                dt: float = MAX_DT, steps: int = 1) -> torch.Tensor:
        """Run bounded recurrent ticks, then render the resulting activations once."""
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 8:
            raise ValueError("steps must be an integer from 1 to 8")
        control_tensor, dt_tensor = self._inputs(controls, dt)
        self._check_state(self.state)
        with torch.inference_mode():
            for tick in range(steps):
                state = self.executor.transition(self.state, control_tensor, dt_tensor)
                self._check_state(state)
                self.state = state
                self.index += 1
                if self.recorder is not None and tick < steps - 1:
                    self.recorder.record(self.index, control_tensor, float(dt))
            frame = self.executor.render(self.state)
        self._check_frame(frame)
        self.frame = frame
        if self.recorder is not None:
            self.recorder.record(self.index, control_tensor, float(dt), frame)
        return frame

    def close(self):
        if self.recorder is not None:
            self.recorder.close()


def benchmark(runner: RuntimeModel, frames: int = 200, warmup: int = 5) -> dict:
    import psutil
    if frames < 1 or warmup < 0:
        raise ValueError("frames must be positive and warmup must be nonnegative")
    process = psutil.Process()
    for _ in range(warmup):
        runner.step()
    runner.synchronize()
    rss_before = process.memory_info().rss
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    durations = []
    rss_peak = rss_before
    started = time.perf_counter()
    for _ in range(frames):
        before = time.perf_counter()
        runner.step()
        runner.synchronize()
        durations.append(time.perf_counter() - before)
        rss_peak = max(rss_peak, process.memory_info().rss)
    total = time.perf_counter() - started
    ordered = sorted(durations)
    result = {"execution": runner.info, "frames": frames, "warmup_frames": warmup,
              "resolution": [runner.width, runner.height], "elapsed_seconds": total,
              "fps": frames / total, "p50_ms": ordered[(frames - 1) // 2] * 1000,
              "p95_ms": ordered[math.ceil(frames * 0.95) - 1] * 1000,
              "rss_before_bytes": rss_before, "rss_peak_bytes": rss_peak,
              "model_bytes": runner.manifest["files"]["model.pt"]["bytes"],
              "model_sha256": runner.manifest["files"]["model.pt"]["sha256"],
              "initial_state_sha256": runner.manifest["files"]["initial_state.pt"]["sha256"],
              "state_bounds_sha256": runner.manifest["files"]["state_bounds.pt"]["sha256"],
              "neural_ir_version": runner.manifest["neural_ir_version"],
              "input": "Idle live transition with dt=1/35; every frame includes simulation and rendering."}
    if runner.device.type == "cuda":
        result["accelerator_peak_bytes"] = torch.cuda.max_memory_allocated(runner.device)
    return result


class FrameServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runner: RuntimeModel):
        self.runner = runner
        self.step_lock = threading.Lock()
        super().__init__(address, FrameHandler)


class FrameHandler(BaseHTTPRequestHandler):
    server: FrameServer

    def log_message(self, format, *args):
        # High-frequency step requests would otherwise obscure model failures.
        if len(args) > 1 and args[1] != "200":
            super().log_message(format, *args)

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path == "/status":
            runner = self.server.runner
            payload = {"width": runner.width, "height": runner.height, "max_dt": MAX_DT,
                       "execution": runner.info, "recording": runner.recorder is not None}
            self._send(json.dumps(payload).encode(), "application/json")
        elif self.path in files:
            name, content_type = files[self.path]
            self._send((Path(__file__).with_name("web") / name).read_bytes(), content_type)
        else:
            self._send(b"Not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/step":
            self._send(b"Not found", "text/plain", 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 8192:
                raise ValueError("Step request must have a body of 1 to 8192 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not {"controls", "dt"} <= payload.keys() or payload.keys() - {"controls", "dt", "steps"}:
                raise ValueError("Step request must contain controls, dt, and optionally steps")
            steps = payload.get("steps", 1)
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 8:
                raise ValueError("steps must be an integer from 1 to 8")
            runner = self.server.runner
            runner._inputs(payload["controls"], payload["dt"])
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._send(json.dumps({"error": str(error)}).encode(), "application/json", 400)
            return
        with self.server.step_lock:
            frame = runner.advance(payload["controls"], payload["dt"], steps)
            body = frame.detach().cpu().contiguous().numpy().tobytes()
        self._send(body, "application/octet-stream")


def serve(runner: RuntimeModel, host: str = "127.0.0.1", port: int = 8765):
    print("Preparing neural transition and renderer for live input…", flush=True)
    runner.advance(dt=0)
    runner.synchronize()
    with FrameServer((host, port), runner) as server:
        print(f"Open http://{host}:{server.server_port} — live inputs, frozen graph, completed RGB outputs", flush=True)
        try:
            server.serve_forever()
        finally:
            runner.close()


def play(runner: RuntimeModel, scale: int = 4, mouse: bool = False):
    import pygame
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    print("Preparing neural transition and renderer for live input…", flush=True)
    frame = runner.advance(dt=0)
    runner.synchronize()
    pygame.init()
    screen = pygame.display.set_mode((runner.width * scale, runner.height * scale))
    pygame.display.set_caption("Doomnet — WASD, arrows, Space fire, E use, R restart, 123 weapons, P pause")
    pygame.event.set_grab(mouse)
    pygame.mouse.set_visible(not mouse)
    clock = pygame.time.Clock()
    accumulator, previous = 0.0, time.perf_counter()
    active, paused = True, False
    weapon, mouse_turn = -1, 0.0
    try:
        while active:
            clock.tick(240)
            now = time.perf_counter()
            elapsed, previous = now - previous, now
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    active = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        active = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        weapon = event.key - pygame.K_1
                elif event.type == pygame.WINDOWFOCUSLOST:
                    paused = True
                elif event.type == pygame.MOUSEMOTION and mouse:
                    mouse_turn -= event.rel[0] * 0.03
            if not active:
                break
            keys = pygame.key.get_pressed()
            controls = [float(keys[pygame.K_w]) - float(keys[pygame.K_s]),
                        float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
                        max(-1.0, min(1.0, float(keys[pygame.K_LEFT]) - float(keys[pygame.K_RIGHT]) + mouse_turn)),
                        float(keys[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]),
                        float(keys[pygame.K_e]), float(keys[pygame.K_r]), float(weapon),
                        float(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])]
            accumulator = 0.0 if paused else min(0.25, accumulator + elapsed)
            steps = min(8, int(accumulator / MAX_DT))
            if steps:
                frame = runner.advance(controls, steps=steps)
                weapon, mouse_turn = -1, 0.0
            accumulator -= steps * MAX_DT
            array = frame.detach().cpu().numpy().swapaxes(0, 1)
            surface = pygame.surfarray.make_surface(array)
            screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
            pygame.display.flip()
    finally:
        runner.close()
        pygame.quit()
