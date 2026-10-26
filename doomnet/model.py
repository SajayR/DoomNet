"""Compile the authoring program into a bounded, fully admitted neural artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import warnings

import torch
from torch import nn

from .runtime import (ARTIFACT_VERSION, MAX_DT, MODEL_METHODS, NO_CONTROLS,
                      artifact_contract, audit_model, execution_device, execution_info,
                      sha256_file, validate_state, validate_state_bounds)


class ConstructedDoom(nn.Module):
    """Construction/compiler input; this source program is never a runtime artifact."""
    def __init__(self, data: dict[str, torch.Tensor], width: int = 160, height: int = 120):
        super().__init__()
        from .simulation import Simulation
        from .rendering import Renderer
        self.simulation = Simulation(data)
        self.renderer = Renderer(data, width=width, height=height)

    def forward(self, state: torch.Tensor, controls: torch.Tensor, dt: torch.Tensor):
        state = self.transition(state, controls, dt)
        return state, self.render(state)

    def transition(self, state: torch.Tensor, controls: torch.Tensor, dt: torch.Tensor):
        return self.simulation(state, controls, dt)

    def render(self, state: torch.Tensor):
        return self.renderer(state)


def _materialize_scalar_coefficients(model: torch.jit.ScriptModule) -> int:
    """Encode wrapped scalar coefficients in each affine consumer's actual dtype.

    Tracing records Python-number promotion, but serialization loses that flag
    on scalar tensor constants. Use the traced execution dtype before freezing
    erases it. Shared constants receive separate typed uses; explicit double
    branches and nonscalar weights retain their constructed precision.
    """
    converted = 0

    def visit(graph, block):
        nonlocal converted
        for node in list(block.nodes()):
            if node.kind() in {"aten::add", "aten::sub", "aten::rsub", "aten::mul", "aten::div"}:
                output_type = node.output().type()
                dtype = output_type.dtype() if isinstance(output_type, torch.TensorType) else None
                if dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
                    for index, value in enumerate(list(node.inputs())):
                        constant = value.node()
                        if constant.kind() != "prim::Constant" or not constant.hasAttribute("value") or constant.kindOf("value") != "t":
                            continue
                        coefficient = constant.t("value")
                        if coefficient.ndim != 0 or not coefficient.is_floating_point() or coefficient.dtype == dtype:
                            continue
                        coefficient = coefficient.to(dtype)
                        if not torch.isfinite(coefficient).item():
                            raise ValueError("Typed scalar coefficient must remain finite in its execution dtype")
                        replacement = graph.create("prim::Constant")
                        replacement.t_("value", coefficient)
                        replacement.output().setType(torch._C.TensorType.create_from_tensor(coefficient))
                        replacement.insertBefore(node)
                        node.replaceInput(index, replacement.output())
                        converted += 1
            for child in node.blocks():
                visit(graph, child)

    for module in model.modules():
        for method in module._c._method_names():
            graph = module._c._get_method(method).graph
            visit(graph, graph)
    return converted


def _tensor_manifest(data: dict[str, torch.Tensor]) -> dict:
    digest = hashlib.sha256()
    tensors = {}
    for name in sorted(data):
        tensor = data[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Construction field {name} must be a tensor")
        tensor = tensor.detach().cpu().contiguous()
        content = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        descriptor = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                      "bytes": tensor.numel() * tensor.element_size(),
                      "sha256": hashlib.sha256(content).hexdigest()}
        digest.update(json.dumps([name, descriptor], sort_keys=True).encode())
        digest.update(content)
        tensors[name] = descriptor
    return {"sha256": digest.hexdigest(), "tensor_count": len(tensors),
            "total_bytes": sum(value["bytes"] for value in tensors.values()), "tensors": tensors}


def neural_component_catalog(model: nn.Module) -> list[dict]:
    """Record components declared by their constructors, without certification.

    The inventory comes from actual instantiated submodules and their fixed
    tensors. It does not infer neural architecture from class inheritance,
    TorchScript conversion, or counts of tensor operators.
    """
    modules = dict(model.named_modules())
    declared = {path: module for path, module in modules.items()
                if callable(getattr(module, "construction_spec", None))}
    repository = Path(__file__).resolve().parents[1]
    catalog = []
    for path, module in declared.items():
        spec = module.construction_spec()
        if not isinstance(spec, dict):
            raise TypeError(f"{path}.construction_spec() must return a JSON object")
        json.dumps(spec, allow_nan=False)
        source = inspect.getsourcefile(type(module))
        assert source is not None, f"Declared circuit {path} requires inspectable construction source"
        prefix = f"{path}." if path else ""
        direct = _tensor_manifest({prefix + name: tensor for name, tensor
                                   in module.named_buffers(recurse=False)})
        ancestors = [name for name in declared if name != path
                     and (not name or path.startswith(name + "."))]
        parent = max(ancestors, key=len) if ancestors else None
        catalog.append({"module_path": path,
                        "type": f"{type(module).__module__}.{type(module).__qualname__}",
                        "source": {"file": str(Path(source).resolve().relative_to(repository)),
                                   "line": inspect.getsourcelines(type(module))[1]},
                        "construction_spec": spec, "parent_component": parent,
                        "fixed_tensor_prefix": prefix,
                        "direct_fixed_tensors": direct["tensors"],
                        "direct_fixed_sha256": direct["sha256"]})
    return catalog


class NeuralDoom(nn.Module):
    """Compose only lowered neural methods, sharing their constructed weights."""

    def __init__(self, transition: nn.Module, renderer: nn.Module):
        super().__init__()
        self.transition_network, self.render_network = transition, renderer

    def forward(self, state, controls, dt):
        state = self.transition(state, controls, dt)
        return state, self.render(state)

    def transition(self, state, controls, dt):
        return self.transition_network(state, controls, dt)

    def render(self, state):
        return self.render_network(state)


def _trace_methods(module: nn.Module, initial: torch.Tensor) -> tuple[torch.jit.ScriptModule, list[str], int]:
    inputs = (initial, torch.tensor(NO_CONTROLS, dtype=torch.float32, device=initial.device),
              torch.tensor([MAX_DT], dtype=torch.float32, device=initial.device))
    with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
        warnings.simplefilter("always", torch.jit.TracerWarning)
        traced = torch.jit.trace_module(module.eval(), {"forward": inputs, "transition": inputs,
                                        "render": (initial,)}, check_trace=False, strict=True)
        normalized = _materialize_scalar_coefficients(traced)
        frozen = torch.jit.freeze(traced.eval(), preserved_attrs=["transition", "render"])
    messages = sorted({str(warning.message) for warning in caught})
    for message in messages:
        if "Converting a tensor to a Python" in message or "Iterating over a tensor" in message:
            raise ValueError(f"Trace contains data-dependent Python execution: {message}")
    return frozen, messages, normalized


def _samples(device="cpu"):
    cases = [("zero_dt", NO_CONTROLS, 0.0),
             ("live_controls", [.7, -.3, .4, 1., 1., 0., 2., 1.], MAX_DT),
             ("restart", [-.2, .5, -.7, 0., 0., 1., -1., 0.], MAX_DT / 2)]
    return [(name, torch.tensor(controls, dtype=torch.float32, device=device),
             torch.tensor([dt], dtype=torch.float32, device=device)) for name, controls, dt in cases]


def _delta(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    difference = (actual.detach().cpu().double() - expected.detach().cpu().double()).abs()
    return {"maximum_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "different_values": int((difference != 0).sum()), "values": difference.numel()}


def export_neural_model(network: nn.Module, initial: torch.Tensor, bounds: torch.Tensor,
                        output_dir: str | Path, *, width: int, height: int,
                        device: str | torch.device = "cpu", construction: dict | None = None,
                        intermediates: dict[str, Path] | None = None) -> dict:
    """Export only a whole-graph admitted target, with exact serialization checks."""
    initial, bounds = initial.detach().cpu().contiguous(), bounds.detach().cpu().contiguous()
    if initial.ndim != 1 or initial.numel() == 0:
        raise ValueError("Initial neural state must be a nonempty flat tensor")
    contract = artifact_contract(initial.shape, width, height)
    validate_state_bounds(bounds, initial.shape)
    validate_state(initial, initial.shape, bounds[0], bounds[1])
    device = execution_device(device)
    torch.set_float32_matmul_precision("highest")
    network = network.to(device).eval()
    device_initial, device_bounds = initial.to(device), bounds.to(device)
    frozen, trace_warnings, normalized = _trace_methods(network, device_initial)
    audit_model(frozen, contract)
    manifest = {"artifact_version": ARTIFACT_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "contract": contract, "parameters": _tensor_manifest(dict(network.state_dict())),
                "neural_components": neural_component_catalog(network), "construction": construction or {},
                "computation_contract": {"instruction_set": "Fixed affine transformations, static wiring, ReLU, and H(x)>0; terminal uint8 output codec only.",
                    "admission": "Every reloaded method is checked for constant/activation provenance and static shapes. Unknown operations, activation products, dynamic weights, and dynamic indices are rejected.",
                    "domains": "Recurrent activations must remain within the serialized finite closed bounds; host rejects violations and never repairs state.",
                    "execution": "CPU evaluates the authoritative frozen neural graph. GPU compiles only its admitted operations to fused kernels with fullgraph=True and no fallback.",
                    "external_boundary": "Host supplies controls/timing, validates generic domains, carries returned activations, and presents completed RGB; source/compiler programs are never executed during play."},
                "export": {"format": "frozen neural TorchScript", "training": False, "trace_warnings": trace_warnings,
                           "target_scalar_literal_normalizations": normalized,
                           "execution": execution_info(device, torch.get_num_threads())},
                "files": {}, "intermediates": {}}
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".doomnet-neural-", dir=output.parent) as temporary:
        stage = Path(temporary)
        torch.jit.save(frozen, str(stage / "model.pt"))
        torch.save(initial, stage / "initial_state.pt")
        torch.save(bounds, stage / "state_bounds.pt")
        deployed = torch.jit.load(str(stage / "model.pt"), map_location=device).eval()
        manifest["methods"], texts = audit_model(deployed, contract)
        manifest["neural_ir_version"] = manifest["methods"]["forward"]["neural_ir_version"]
        state, verified = device_initial.clone(), []
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            for name, controls, dt in _samples(device):
                expected_state, expected_frame = network(state, controls, dt)
                actual_state, actual_frame = deployed(state, controls, dt)
                split_state = deployed.transition(state, controls, dt)
                split_frame = deployed.render(split_state)
                torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
                torch.testing.assert_close(actual_frame, expected_frame, rtol=0, atol=0)
                torch.testing.assert_close(split_state, actual_state, rtol=0, atol=0)
                torch.testing.assert_close(split_frame, actual_frame, rtol=0, atol=0)
                validate_state(actual_state, initial.shape, device_bounds[0], device_bounds[1])
                verified.append(name)
                state = actual_state
        manifest["export"]["exact_lowered_eager_and_full_split_checks"] = verified
        method_texts = []
        for method, audit in manifest["methods"].items():
            operators = "\n".join(f"{count:7d}  {name}" for name, count in audit["operators"].items())
            method_texts.append(f"Method: {method}\n\n{operators}\n\n{texts[method]}")
        (stage / "operators.txt").write_text("Exhaustive neural IR admission: fixed affine maps, static wires, ReLU, H>0, terminal uint8 codec.\n\n" + "\n\n".join(method_texts) + "\n")
        for name in ("model.pt", "initial_state.pt", "state_bounds.pt"):
            manifest["files"][name] = {"sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
        for name, source_path in (intermediates or {}).items():
            if Path(name).name != name or name in {"model.pt", "initial_state.pt", "state_bounds.pt", "manifest.json", "operators.txt"}:
                raise ValueError(f"Invalid construction intermediate filename: {name}")
            (stage / name).write_bytes(Path(source_path).read_bytes())
            manifest["intermediates"][name] = {"sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size,
                                               "execution": "construction/debug only; not loaded by runtime"}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        output.mkdir(parents=True, exist_ok=True)
        for artifact_path in stage.iterdir():
            if artifact_path.name != "manifest.json":
                artifact_path.replace(output / artifact_path.name)
        (stage / "manifest.json").replace(output / "manifest.json")
    return manifest


def build_model(wad_path: str | Path, output_dir: str | Path, skill: int = 2,
                width: int = 160, height: int = 120, threads: int = 1,
                device: str = "cpu") -> dict:
    """Lower source gameplay and rendering; export no unlowered runtime path."""
    from .assets import build_assets
    from .neural_lowering import lower_neural_method
    from .state import initial_state, state_bounds
    if not 0 <= skill <= 4 or threads < 1:
        raise ValueError("skill must be from 0 to 4 and threads must be positive")
    target_device = execution_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(threads)
    print("Constructing WAD content and finite recurrent domains…", flush=True)
    data, metadata = build_assets(wad_path, skill=skill)
    initial, bounds = initial_state(data), torch.stack(state_bounds(data))
    validate_state_bounds(bounds, initial.shape)
    validate_state(initial, initial.shape, bounds[0], bounds[1])
    source_eager = ConstructedDoom(data, width=width, height=height).eval()
    source, source_warnings, normalized = _trace_methods(source_eager, initial)
    control_low = torch.tensor([-1., -1., -1., 0., 0., 0., -1., 0.])
    control_high = torch.tensor([1., 1., 1., 1., 1., 1., 2., 1.])
    input_domains = {"state": (bounds[0], bounds[1]), "controls": (control_low, control_high),
                     "dt": (torch.zeros(1), torch.tensor([MAX_DT]))}
    print("Lowering complete transition and renderer to neural primitives…", flush=True)
    transition, transition_report = lower_neural_method(source, "transition", input_domains, source_modules=source_eager)
    renderer, render_report = lower_neural_method(source, "render", {"state": input_domains["state"]}, source_modules=source_eager)
    target = NeuralDoom(transition, renderer).to(target_device).eval()
    target_initial = initial.to(target_device)
    comparisons = []
    with torch.inference_mode():
        for name, controls, dt in _samples():
            source_state, source_frame = source_eager(initial, controls, dt)
            target_state, target_frame = target(target_initial, controls.to(target_device), dt.to(target_device))
            comparisons.append({"case": name, "state": _delta(source_state, target_state),
                                "frame": _delta(source_frame, target_frame)})
    construction = {"source": metadata, "asset_tensors": _tensor_manifest(data),
                    "source_parameters": _tensor_manifest(dict(source_eager.state_dict())),
                    "source_trace_warnings": source_warnings, "source_scalar_literal_normalizations": normalized,
                    "lowering": {"transition": transition_report, "render": render_report},
                    "source_accuracy_probes": {"measurements": comparisons,
                                               "status": "Diagnostic measurements; numerical/gameplay acceptance requires explicit limits and full-level validation."}}
    print("Freezing, reloading, and exhaustively auditing the neural target…", flush=True)
    with tempfile.TemporaryDirectory(prefix="doomnet-source-") as temporary:
        source_path = Path(temporary) / "source.pt"
        # Fresh scope information was used above; saved source is debug data.
        torch.jit.save(source, str(source_path))
        manifest = export_neural_model(target, initial, bounds, output_dir, width=width, height=height,
                                        device=target_device, construction=construction,
                                        intermediates={"source.pt": source_path})
    print(f"Admitted neural artifact written to {Path(output_dir).resolve()}", flush=True)
    return manifest
