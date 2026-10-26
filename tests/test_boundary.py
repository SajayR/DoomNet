"""v2 neural admission, bounded recurrence, and generic execution boundaries.

Tiny networks exercise the artifact/runtime contract; they are not evidence of
Doom completion or numerical accuracy of the complete lowered level.
"""

import json
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from doomnet.model import export_neural_model, _materialize_scalar_coefficients, _trace_methods
from doomnet.neural_ir import neural_method_fx
from doomnet.neural_math import Atan2, SinCos
from doomnet.runtime import (MAX_DT, NO_CONTROLS, FrameServer, RuntimeModel, artifact_contract,
                            audit_model, benchmark, sha256_file)


def carry(value, gate):
    off = 1 - gate
    return F.relu(value - off) - F.relu(-value - off)


class TinyNetwork(nn.Module):
    """A bounded recurrent affine/threshold/ReLU network with terminal RGB codec."""

    def __init__(self, mixed=False, limit=2.0):
        super().__init__()
        self.mixed, self.limit = mixed, limit
        self.register_buffer("motion_weight", torch.tensor([MAX_DT], dtype=torch.float64 if mixed else torch.float32))
        self.register_buffer("pixel_weight", torch.ones(18, 1, dtype=torch.float16))
        self.register_buffer("pixel_bias", torch.arange(18, dtype=torch.float16))

    def forward(self, state, controls, dt):
        state = self.transition(state, controls, dt)
        return state, self.render(state)

    def transition(self, state, controls, dt):
        dtype = self.motion_weight.dtype
        gate = (dt[0] > 0).to(dtype)
        position = state[:1].to(dtype) + carry(controls[:1].to(dtype), gate) * self.motion_weight
        position = position + F.relu(-self.limit - position) - F.relu(position - self.limit)
        active = gate.float()
        firing = carry(controls[3:4], active) + carry(state[1:2], 1 - active)
        return torch.cat((position.float(), firing))

    def render(self, state):
        image = F.linear(state[1:2].half(), self.pixel_weight, self.pixel_bias).float()
        image = image + F.relu(-image) - F.relu(image - 255)
        return image.reshape(2, 3, 3).to(torch.uint8)


class OrdinarySource(TinyNetwork):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.register_buffer("table", torch.arange(4).float())

    def transition(self, state, controls, dt):
        if self.operation == "product":
            position = state[0] + controls[0] * dt[0]
        elif self.operation == "trigonometry":
            position = torch.sin(state[0])
        else:
            position = self.table[(controls[6] + 1).long()]
        return torch.stack((position, controls[3]))


class ScalarMathNetwork(TinyNetwork):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
        self.trigonometry = SinCos(4, segments=16, dtype=dtype)
        self.angle = Atan2(4, segments=8, fractional_bits=8, dtype=dtype)

    def transition(self, state, controls, dt):
        phase = state[0].to(self.dtype) + controls[2].to(self.dtype)
        direction = self.angle(controls[1].to(self.dtype), state[1].to(self.dtype) + 0.5)
        return torch.stack((self.trigonometry(phase)[0], direction)).float()


class ScalarPrecisionNetwork(TinyNetwork):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def transition(self, state, controls, dt):
        zero = controls[0] * 0
        # The same literal rounds away in Float32 and retains a bit in Float64.
        rounded = zero + (1 + 2 ** -30) - 1
        precise = (zero.double() + (1 + 2 ** -30) - 1) * 2 ** 30
        reduced = ((controls[0].to(self.dtype) + 1.25) * 0.5 / 2.0 - 0.25)
        return torch.stack(((rounded + reduced).float(), (1.0 - reduced + precise).float()))


def make_artifact(directory, network=None, bounds=None, device="cpu"):
    network = TinyNetwork() if network is None else network
    bounds = torch.tensor([[-2., 0.], [2., 1.]]) if bounds is None else bounds
    export_neural_model(network.eval(), torch.zeros(2), bounds, directory, width=3, height=2, device=device)
    return directory


@pytest.fixture
def artifact(tmp_path):
    return make_artifact(tmp_path)


def replace_artifact_tensor(artifact, name, value):
    torch.save(value, artifact / name)
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["files"][name] = {"sha256": sha256_file(artifact / name), "bytes": (artifact / name).stat().st_size}
    (artifact / "manifest.json").write_text(json.dumps(manifest))


def test_live_inputs_recur_with_fixed_neural_parameters(artifact):
    runner = RuntimeModel(artifact)
    controls = list(NO_CONTROLS)
    controls[0], controls[3] = 1, 1
    frame = runner.step(controls, MAX_DT)
    torch.testing.assert_close(runner.state, torch.tensor([MAX_DT, 1.0]), rtol=0, atol=0)
    assert frame[0, 0, 0].item() == 1
    assert list(runner.model.named_parameters()) == []
    assert list(runner.model.named_buffers()) == []
    controls[0], controls[3] = -1, 0
    runner.step(controls, MAX_DT)
    torch.testing.assert_close(runner.state, torch.zeros(2), rtol=0, atol=0)
    assert runner.manifest["artifact_version"] == 2
    assert runner.manifest["export"]["exact_lowered_eager_and_full_split_checks"] == ["zero_dt", "live_controls", "restart"]
    assert all(audit["dynamic_weights"] == audit["dynamic_indices"] == 0 for audit in runner.audits.values())
    assert "supporting_tensor_algorithms" not in runner.manifest


def test_catchup_transitions_match_full_forward_and_render_once(artifact):
    runner, comparison = RuntimeModel(artifact), RuntimeModel(artifact)
    controls = list(NO_CONTROLS)
    controls[0], controls[3] = 0.5, 1
    for _ in range(3):
        expected = comparison.step(controls)
    spy = SimpleNamespace(transition=Mock(wraps=runner.executor.transition), render=Mock(wraps=runner.executor.render))
    runner.executor = spy
    actual = runner.advance(controls, steps=3)
    torch.testing.assert_close(runner.state, comparison.state, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runner.index == 3
    assert spy.transition.call_count == 3
    assert spy.render.call_count == 1


def test_execution_enforces_and_records_highest_float32_precision(artifact):
    torch.set_float32_matmul_precision("high")
    runner = RuntimeModel(artifact)
    assert torch.get_float32_matmul_precision() == "highest"
    assert runner.info["float32_matmul_precision"] == "highest"
    assert runner.manifest["export"]["execution"]["float32_matmul_precision"] == "highest"


@pytest.mark.parametrize("operation", ["step", "advance"])
def test_initial_state_preserves_bytes_and_recurrent_compilation_guards(artifact, operation):
    initial = torch.tensor([0.5, -0.0])
    replace_artifact_tensor(artifact, "initial_state.pt", initial)
    runner = RuntimeModel(artifact, threads=1)
    assert torch.is_inference(runner.state)
    assert torch.equal(runner.state.view(torch.uint8), initial.view(torch.uint8))
    compilations = defaultdict(int)

    def compile_method(method):
        def backend(graph, inputs):
            compilations[method] += 1
            return graph.forward
        module = neural_method_fx(runner.model, method, input_shapes=runner.method_shapes[method],
                                  input_dtypes=[torch.float32] * len(runner.method_shapes[method]))
        return torch.compile(module, backend=backend, fullgraph=True)

    runner.executor = SimpleNamespace(**{method: compile_method(method)
                                        for method in ("forward", "transition", "render")})
    execute = getattr(runner, operation)
    execute(dt=0)
    for controls in ([1, 0, 0, 1, 0, 0, -1, 0], NO_CONTROLS, [-1, 0, 0, 0, 0, 0, -1, 0]):
        execute(controls)
        assert torch.is_inference(runner.state)
    assert dict(compilations) == ({"forward": 1} if operation == "step" else {"transition": 1, "render": 1})


@pytest.mark.parametrize("operation", ["product", "trigonometry", "address"])
def test_export_and_load_reject_unlowered_source_even_with_forged_manifest(artifact, operation):
    source = OrdinarySource(operation).eval()
    with pytest.raises(ValueError, match="neural IR rejected"):
        make_artifact(artifact / "source", source)
    frozen, _, _ = _trace_methods(source, torch.zeros(2))
    torch.jit.save(frozen, str(artifact / "model.pt"))
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["files"]["model.pt"] = {"sha256": sha256_file(artifact / "model.pt"), "bytes": (artifact / "model.pt").stat().st_size}
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="neural IR rejected"):
        RuntimeModel(artifact)


def test_v1_artifact_is_rejected_without_compatibility_path(artifact):
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["artifact_version"] = 1
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported artifact version"):
        RuntimeModel(artifact)


@pytest.mark.parametrize("field,value", [
    ("state_dtype", "float64"), ("state_bounds_shape", [2, 3]),
    ("dt_shape", []), ("weapon_values", [-1, 0, 1]),
    ("frame_dtype", "float32"), ("width", True),
])
def test_conflicting_tensor_contract_is_rejected(artifact, field, value):
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["contract"][field] = value
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Artifact"):
        RuntimeModel(artifact)


@pytest.mark.parametrize("bounds", [
    torch.tensor([[-2., 0.], [2., float("inf")]]),
    torch.tensor([[-2., 2.], [2., 1.]]),
    torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.float64),
])
def test_invalid_declared_state_bounds_fail_before_execution(artifact, bounds):
    replace_artifact_tensor(artifact, "state_bounds.pt", bounds)
    with pytest.raises(ValueError, match="State bounds"):
        RuntimeModel(artifact)


def test_invalid_initial_activations_are_rejected(artifact):
    replace_artifact_tensor(artifact, "initial_state.pt", torch.tensor([2.01, 0.]))
    with pytest.raises(ValueError, match="violates declared bounds"):
        RuntimeModel(artifact)


def test_out_of_domain_feedback_is_rejected_without_repair(tmp_path):
    artifact = make_artifact(tmp_path, TinyNetwork(limit=3), torch.tensor([[-3., 0.], [3., 1.]]))
    replace_artifact_tensor(artifact, "state_bounds.pt", torch.tensor([[-2., 0.], [2., 1.]]))
    replace_artifact_tensor(artifact, "initial_state.pt", torch.tensor([2., 0.]))
    runner = RuntimeModel(artifact)
    before = runner.state.clone()
    with pytest.raises(ValueError, match="violates declared bounds"):
        runner.step([1, 0, 0, 0, 0, 0, -1, 0])
    torch.testing.assert_close(runner.state, before, rtol=0, atol=0)
    assert runner.index == 0


def test_corrupted_input_state_is_rejected_before_invoking_network(artifact):
    runner = RuntimeModel(artifact)
    with torch.inference_mode():
        runner.state[0] = float("nan")
    spy = Mock(wraps=runner.executor.forward)
    runner.executor.forward = spy
    with pytest.raises(ValueError, match="non-finite"):
        runner.step()
    assert spy.call_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("export_device", ["cpu", "cuda"])
def test_compiled_gpu_preserves_double_and_half_network_and_full_split_parity(tmp_path, export_device):
    artifact = make_artifact(tmp_path, TinyNetwork(mixed=True), device=export_device)
    cpu, gpu = RuntimeModel(artifact), RuntimeModel(artifact, device="cuda")
    controls = torch.tensor([.7, -.3, .4, 1., 0., 0., -1., 0.])
    dt = torch.tensor([MAX_DT])
    with torch.inference_mode():
        expected_state, expected_frame = cpu.executor.forward(cpu.state, controls, dt)
        state, frame = gpu.executor.forward(gpu.state, controls.cuda(), dt.cuda())
        split_state = gpu.executor.transition(gpu.state, controls.cuda(), dt.cuda())
        split_frame = gpu.executor.render(split_state)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(frame.cpu(), expected_frame, rtol=0, atol=0)
    torch.testing.assert_close(split_state, state, rtol=0, atol=0)
    torch.testing.assert_close(split_frame, frame, rtol=0, atol=0)
    assert gpu.info["executor"] == "inductor_fullgraph"
    assert cpu.manifest["parameters"]["tensors"]["motion_weight"]["dtype"] == "torch.float64"
    assert cpu.manifest["parameters"]["tensors"]["pixel_weight"]["dtype"] == "torch.float16"
    assert torch.load(artifact / "initial_state.pt", weights_only=True).device.type == "cpu"
    assert torch.load(artifact / "state_bounds.pt", weights_only=True).device.type == "cpu"
    assert cpu.manifest["export"]["execution"]["device"] == export_device


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA or ROCm"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_saved_scalar_trigonometry_and_atan2_keep_typed_affine_inputs(tmp_path, device, dtype):
    network = ScalarMathNetwork(dtype).eval().to(device)
    artifact = make_artifact(tmp_path, network, torch.tensor([[-4., -4.], [4., 4.]]), device=device)
    runner = RuntimeModel(artifact, device=device)
    normalizations = runner.manifest["export"]["target_scalar_literal_normalizations"]
    assert normalizations > 0 if dtype == torch.float32 else normalizations == 0
    cases = [([-0.3, -1.], [.7, -.3, .4, 0., 0., 0., -1., 0.]),
             ([0., 0.], list(NO_CONTROLS)),
             ([3.5, 3.], [-.2, .5, -.7, 0., 0., 0., -1., 0.])]
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for state, controls in cases:
            inputs = (torch.tensor(state, device=device), torch.tensor(controls, device=device),
                      torch.tensor([MAX_DT], device=device))
            expected_state, expected_frame = network(*inputs)
            state, frame = runner.model(*inputs)
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
            torch.testing.assert_close(frame, expected_frame, rtol=0, atol=0)
            torch.testing.assert_close(runner.model.transition(*inputs), state, rtol=0, atol=0)
            torch.testing.assert_close(runner.model.render(state), frame, rtol=0, atol=0)
    if dtype == torch.float64:
        assert runner.manifest["parameters"]["tensors"]["trigonometry.kernel.output_weight"]["dtype"] == "torch.float64"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA or ROCm"))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_saved_shared_scalar_literals_preserve_double_and_reduced_precision(tmp_path, device, dtype):
    network = ScalarPrecisionNetwork(dtype).eval().to(device)
    artifact = make_artifact(tmp_path, network, torch.tensor([[-2., -2.], [2., 2.]]), device=device)
    runner = RuntimeModel(artifact, device=device)
    assert runner.manifest["export"]["target_scalar_literal_normalizations"] > 0
    inputs = (runner.state, torch.tensor(NO_CONTROLS, device=device), torch.tensor([MAX_DT], device=device))
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        expected_state, expected_frame = network(*inputs)
        assert expected_state.tolist() == [0.0625, 1.9375]
        state, frame = runner.executor.forward(*inputs)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(frame, expected_frame, rtol=0, atol=0)
        torch.testing.assert_close(runner.executor.transition(*inputs), state, rtol=0, atol=0)
        torch.testing.assert_close(runner.executor.render(state), frame, rtol=0, atol=0)


def test_shared_constant_is_typed_per_consumer_before_freezing(tmp_path):
    network = ScalarPrecisionNetwork(torch.float16).eval()
    inputs = (torch.zeros(2), torch.tensor(NO_CONTROLS), torch.tensor([MAX_DT]))
    traced = torch.jit.trace_module(network, {"forward": inputs, "transition": inputs,
                                           "render": inputs[:1]}, check_trace=False)
    for method in traced._c._method_names():
        torch._C._jit_pass_constant_pooling(traced._c._get_method(method).graph)
    shared = next(node for node in traced.transition.graph.nodes()
                  if node.kind() == "prim::Constant" and node.hasAttribute("value")
                  and node.kindOf("value") == "t" and node.t("value").ndim == 0
                  and node.t("value").item() == 1 + 2 ** -30)
    assert {use.user.output().type().dtype() for use in shared.output().uses()} == {torch.float32, torch.float64}
    assert _materialize_scalar_coefficients(traced) > 0
    frozen = torch.jit.freeze(traced, preserved_attrs=["transition", "render"])
    torch.jit.save(frozen, str(tmp_path / "shared.pt"))
    deployed = torch.jit.load(str(tmp_path / "shared.pt"))
    audit_model(deployed, artifact_contract((2,), 3, 2))
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        expected_state, expected_frame = network(*inputs)
        state, frame = deployed(*inputs)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(frame, expected_frame, rtol=0, atol=0)


@pytest.mark.parametrize("controls,dt", [
    ([0] * 7, MAX_DT), ([0] * 9, MAX_DT), ([float("nan")] + [0] * 7, MAX_DT),
    ([1.1, 0, 0, 0, 0, 0, -1, 0], MAX_DT),
    ([0, 0, 0, -0.1, 0, 0, -1, 0], MAX_DT),
    ([0, 0, 0, 0, 1.1, 0, -1, 0], MAX_DT),
    ([0, 0, 0, 0, 0, 0, 0.5, 0], MAX_DT),
    ([0, 0, 0, 0, 0, 0, 0.999999999, 0], MAX_DT),
    ([1.000000001, 0, 0, 0, 0, 0, -1, 0], MAX_DT),
    (NO_CONTROLS, -0.01), (NO_CONTROLS, MAX_DT + 0.00001),
    (NO_CONTROLS, float("nan")), (NO_CONTROLS, True),
])
def test_invalid_external_inputs_do_not_change_state(artifact, controls, dt):
    runner = RuntimeModel(artifact)
    initial = runner.state.clone()
    with pytest.raises(ValueError):
        runner.step(controls, dt)
    assert torch.equal(runner.state, initial)
    assert runner.index == 0


def test_artifact_checksum_is_required(artifact):
    with (artifact / "initial_state.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        RuntimeModel(artifact)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA or ROCm"))])
def test_runtime_executes_without_any_construction_modules(artifact, device):
    code = """
import importlib.abc, sys
class DenyBuild(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {'doomnet.assets', 'doomnet.geometry', 'doomnet.circuits', 'doomnet.simulation', 'doomnet.rendering', 'doomnet.state', 'doomnet.model', 'doomnet.neural_lowering', 'doomnet.neural_math', 'doomnet.neural_memory'}:
            raise AssertionError('Runtime imported construction code: ' + fullname)
sys.meta_path.insert(0, DenyBuild())
from doomnet.runtime import RuntimeModel
runner = RuntimeModel(sys.argv[1], device=sys.argv[2])
assert tuple(runner.step().shape) == (2, 3, 3)
"""
    result = subprocess.run([sys.executable, "-c", code, str(artifact), device], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_http_transports_completed_rgb_and_rejects_invalid_input(artifact):
    runner = RuntimeModel(artifact)
    server = FrameServer(("127.0.0.1", 0), runner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/status") as response:
            assert json.load(response)["width"] == 3
        payload = {"controls": list(NO_CONTROLS), "dt": MAX_DT, "steps": 2}
        request = Request(base + "/step", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            assert response.read() == torch.arange(18, dtype=torch.uint8).numpy().tobytes()
        assert runner.index == 2
        payload["dt"] = 1
        with pytest.raises(HTTPError) as raised:
            urlopen(Request(base + "/step", data=json.dumps(payload).encode()))
        assert raised.value.code == 400
        assert runner.index == 2
        with pytest.raises(HTTPError) as raised:
            urlopen(base + "/../model.pt")
        assert raised.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_startup_prepares_live_executor_before_binding(artifact, monkeypatch, capsys):
    from doomnet.runtime import serve
    directory = artifact / "startup-recording"
    runner = RuntimeModel(artifact, record_dir=directory)
    initial = runner.state.clone()
    runner.executor.forward = Mock(wraps=runner.executor.forward)
    runner.advance = Mock(wraps=runner.advance)
    runner.synchronize = Mock(wraps=runner.synchronize)

    def bind(address, prepared):
        assert prepared is runner
        runner.advance.assert_called_once_with(dt=0)
        runner.synchronize.assert_called_once_with()
        torch.testing.assert_close(runner.state, initial, rtol=0, atol=0)
        assert runner.index == 1
        return FrameServer(address, runner)

    monkeypatch.setattr("doomnet.runtime.FrameServer", bind)
    monkeypatch.setattr(FrameServer, "serve_forever", lambda server: None)
    serve(runner, port=0)
    runner.executor.forward.assert_not_called()
    output = capsys.readouterr().out
    assert output.index("Preparing neural transition and renderer") < output.index("Open http://")
    row = json.loads((directory / "inputs.jsonl").read_text())
    assert row["index"] == 1 and row["dt"] == 0
    assert row["controls"] == list(NO_CONTROLS)
    assert (directory / row["image"]).is_file()


def test_benchmark_reports_actual_full_graph_timings(artifact):
    result = benchmark(RuntimeModel(artifact), frames=3, warmup=1)
    assert result["fps"] > 0
    assert 0 < result["p50_ms"] <= result["p95_ms"]
    assert result["rss_peak_bytes"] >= result["rss_before_bytes"] > 0
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert result["model_sha256"] == manifest["files"]["model.pt"]["sha256"]
    assert result["initial_state_sha256"] == manifest["files"]["initial_state.pt"]["sha256"]


def test_recording_contains_only_inputs_and_completed_frames(artifact, tmp_path):
    directory = tmp_path / "capture"
    runner = RuntimeModel(artifact, record_dir=directory)
    runner.step()
    runner.close()
    row = json.loads((directory / "inputs.jsonl").read_text())
    assert set(row) == {"index", "wall_seconds", "controls", "dt", "image"}
    assert (directory / row["image"]).is_file()
    assert "state" not in json.loads((directory / "recording.json").read_text())


def test_recording_limits_images_but_keeps_every_external_input(artifact, tmp_path, monkeypatch):
    times = iter((0.0, 0.0, 0.02, 0.11))
    monkeypatch.setattr("doomnet.runtime.time.perf_counter", lambda: next(times))
    directory = tmp_path / "limited"
    runner = RuntimeModel(artifact, record_dir=directory)
    for _ in range(3):
        runner.step()
    runner.close()
    rows = [json.loads(line) for line in (directory / "inputs.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert [row["index"] for row in rows if "image" in row] == [1, 3]
    assert len(list(directory.glob("*.png"))) == 2


def test_native_event_inputs_survive_a_pass_without_a_model_tick(artifact, monkeypatch):
    from doomnet.runtime import play
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame
    runner = RuntimeModel(artifact)
    initial = runner.state.clone()
    captured = []
    original_advance = runner.advance
    runner.executor.forward = Mock(wraps=runner.executor.forward)
    runner.synchronize = Mock(wraps=runner.synchronize)
    def capture_advance(controls=NO_CONTROLS, dt=MAX_DT, steps=1):
        if not captured:
            assert dt == 0
            assert pygame.display.get_surface() is None
        captured.append(tuple(controls))
        return original_advance(controls, dt, steps)
    runner.advance = capture_advance
    original_window = pygame.display.set_mode
    def create_window(size):
        assert captured == [NO_CONTROLS]
        runner.synchronize.assert_called_once_with()
        torch.testing.assert_close(runner.state, initial, rtol=0, atol=0)
        return original_window(size)
    monkeypatch.setattr(pygame.display, "set_mode", create_window)
    times = iter((0.0, 0.001, 0.04, 0.041))
    monkeypatch.setattr("doomnet.runtime.time.perf_counter", lambda: next(times))
    batches = iter(([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3),
                     pygame.event.Event(pygame.MOUSEMOTION, rel=(4, 0))],
                    [], [pygame.event.Event(pygame.QUIT)]))
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: defaultdict(bool))
    monkeypatch.setattr(pygame.mouse, "get_pressed", lambda: (False, False, False))
    monkeypatch.setattr(pygame.time, "Clock", lambda: type("Clock", (), {"tick": lambda self, fps: None})())
    play(runner, mouse=True)
    runner.executor.forward.assert_not_called()
    assert len(captured) == 2
    assert captured[1][6] == 2
    assert captured[1][2] == pytest.approx(-0.12)
