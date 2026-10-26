"""The whole executable graph must consist of admitted neural primitives."""

from io import BytesIO
import operator

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from doomnet.circuits import BooleanCircuit
from doomnet.neural_ir import audit_neural_graph, neural_method_fx
from doomnet.neural_memory import PaletteROM


def frozen(module, *inputs):
    return torch.jit.freeze(torch.jit.trace(module.eval(), inputs, check_trace=False))


class AffineCarry(nn.Module):
    """A constructed bounded branch carry, with actual neuron computation."""

    def __init__(self):
        super().__init__()
        self.register_buffer("wires", torch.tensor([1, 0]))
        self.register_buffer("weight", torch.tensor([[1., -1.]]))

    def forward(self, inputs):
        pair = inputs.index_select(-1, self.wires)
        off = (F.linear(pair, self.weight) > 0).float()
        branches = torch.stack((off, 1 - off), dim=-1)
        carry = F.relu(pair.unsqueeze(-2) - branches * 10)
        return carry.sum(-1)


def test_admits_real_affine_threshold_relu_circuit_and_static_wires():
    model = frozen(AffineCarry(), torch.tensor([3., 7.]))
    audit, graph = audit_neural_graph(model, input_shapes=[(2,)])
    assert {"affine", "hard_threshold", "relu", "wire"} <= audit["primitive_families"].keys()
    assert audit["dynamic_weights"] == audit["dynamic_indices"] == 0
    assert audit["inputs"] == [{"name": "inputs", "shape": [2], "dtype": "torch.float32"}]
    assert audit["outputs"] == [{"shape": [1], "dtype": "torch.float32", "codec": None}]
    assert "aten::linear" in graph
    torch.testing.assert_close(model(torch.tensor([3., 7.])), torch.tensor([3.]))


def test_admits_existing_dnf_circuit():
    circuit = BooleanCircuit(3, [[(1, -2)], [(2,), (3,)]])
    model = frozen(circuit, torch.zeros(4, 3))
    audit, _ = audit_neural_graph(model, input_shapes={"inputs": [4, 3]})
    assert audit["primitive_families"]["hard_threshold"] == 2
    inputs = torch.tensor([[1., 0., 0.], [0., 1., 1.], [0., 0., 1.], [1., 1., 0.]])
    torch.testing.assert_close(model(inputs), torch.tensor([[1., 0.], [0., 1.], [0., 1.], [0., 1.]]))


def test_half_precision_recursive_rom_passes_whole_graph_audit_and_returns_exact_bytes():
    table = torch.randint(256, (257, 4), dtype=torch.uint8,
                          generator=torch.Generator().manual_seed(13))
    address = torch.tensor([[0., 256., 89.], [32., 1., 196.]])
    circuit = PaletteROM(table, query_chunk=4, matrix_dtype=torch.float16)
    model = frozen(circuit, address)
    stream = BytesIO()
    torch.jit.save(model, stream)
    stream.seek(0)
    deployed = torch.jit.load(stream)
    audit, _ = audit_neural_graph(deployed, input_shapes=[address.shape])
    constant_values = [node.output().toIValue() for node in deployed.inlined_graph.nodes()
                       if node.kind() == "prim::Constant"]
    assert any(isinstance(value, torch.Tensor) and value.dtype == torch.float16 for value in constant_values)
    assert {"affine", "hard_threshold", "relu"} <= audit["primitive_families"].keys()
    assert audit["outputs"] == [{"shape": [2, 3, 4], "dtype": "torch.float32", "codec": None}]
    assert audit["dynamic_weights"] == audit["dynamic_indices"] == 0
    torch.testing.assert_close(deployed(address.flip(-1)), table[address.flip(-1).long()].float(), rtol=0, atol=0)


class MixedPrecisionBytes(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("weight", torch.tensor([[255., 0., 128.], [0., 255., 129.]], dtype=dtype))

    def forward(self, inputs):
        return F.relu(F.linear((inputs > 0).to(self.weight.dtype), self.weight)).float()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_precision_internal_affine_layers_are_neural(dtype):
    model = frozen(MixedPrecisionBytes(dtype), torch.eye(3))
    audit, _ = audit_neural_graph(model, input_shapes=[(3, 3)])
    assert audit["outputs"][0]["dtype"] == "torch.float32"
    torch.testing.assert_close(model(torch.eye(3)), torch.tensor([[255., 0.], [0., 255.], [128., 129.]]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_external_input_dtype_contract_remains_float32_or_float64(dtype):
    model = frozen(NonNeuralOperation(torch.relu), torch.zeros(2))
    with pytest.raises(ValueError, match="input inputs must contain float32 or float64"):
        audit_neural_graph(model, input_shapes=[(2,)], input_dtypes=[dtype])


class DynamicMultiply(nn.Module):
    def forward(self, x, y):
        return x * y


def test_rejects_dynamic_multiplication_even_though_it_is_builtin_aten():
    model = frozen(DynamicMultiply(), torch.ones(2), torch.ones(2))
    with pytest.raises(ValueError, match="dynamic multiplication"):
        audit_neural_graph(model, input_shapes=[(2,), (2,)])


def test_observed_constant_values_do_not_establish_constant_provenance():
    model = frozen(NonNeuralOperation(lambda x: x * (x * 0 + 1)), torch.zeros(2))
    with pytest.raises(ValueError, match="dynamic multiplication"):
        audit_neural_graph(model, input_shapes=[(2,)])


class DynamicWeights(nn.Module):
    def forward(self, x, weight):
        return F.linear(x, weight)


def test_rejects_activation_weights():
    model = frozen(DynamicWeights(), torch.ones(2), torch.eye(2))
    with pytest.raises(ValueError, match="linear weights must be a construction constant"):
        audit_neural_graph(model, input_shapes=[(2,), (2, 2)])


def test_fixed_division_is_affine_but_dynamic_or_rounded_division_is_rejected():
    model = frozen(NonNeuralOperation(lambda x: x / 2), torch.zeros(2))
    audit, _ = audit_neural_graph(model, input_shapes=[(2,)])
    assert audit["primitive_families"]["affine"] == 1
    model = frozen(NonNeuralOperation(lambda x: x / (x + 1)), torch.zeros(2))
    with pytest.raises(ValueError, match="division denominator must be a construction constant"):
        audit_neural_graph(model, input_shapes=[(2,)])
    model = frozen(NonNeuralOperation(lambda x: torch.div(x, 2, rounding_mode="floor")), torch.zeros(2))
    with pytest.raises(ValueError, match="rounded division"):
        audit_neural_graph(model, input_shapes=[(2,)])


def test_fixed_division_requires_finite_affine_weights():
    model = frozen(NonNeuralOperation(lambda x: x / 1e-40), torch.zeros(2))
    with pytest.raises(ValueError, match="finite affine weights"):
        audit_neural_graph(model, input_shapes=[(2,)])


class DynamicIndex(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer("table", torch.tensor([10., 20., 30.]))

    def forward(self, scores):
        index = (scores > 0).long()
        if self.kind == "index_select":
            return self.table.index_select(0, index)
        if self.kind == "gather":
            return torch.gather(self.table, 0, index)
        return self.table[index]


@pytest.mark.parametrize("kind", ["index_select", "gather", "index"])
def test_rejects_dynamic_indices(kind):
    model = frozen(DynamicIndex(kind), torch.zeros(2))
    with pytest.raises(ValueError, match="integer/Boolean casts|construction constant"):
        audit_neural_graph(model, input_shapes=[(2,)])


class NonNeuralOperation(nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, inputs):
        return self.operation(inputs)


@pytest.mark.parametrize("operation,kind", [
    (torch.sin, "sin"), (torch.cos, "cos"), (torch.sqrt, "sqrt"),
    (torch.floor, "floor"), (torch.square, "square"), (torch.abs, "abs"),
    (torch.exp, "exp"),
    (lambda x: torch.sort(x).values, "sort"),
    (lambda x: (x > 0) & (x > 1), "gt"),
])
def test_rejects_unlowered_nonlinear_operations(operation, kind):
    model = frozen(NonNeuralOperation(operation), torch.tensor([1., 2.]))
    with pytest.raises(ValueError, match=f"aten::{kind}"):
        audit_neural_graph(model, input_shapes=[(2,)])


def test_rejects_nonzero_comparison_reference():
    model = frozen(NonNeuralOperation(lambda x: x > 1), torch.zeros(2))
    with pytest.raises(ValueError, match="compare an affine activation with zero"):
        audit_neural_graph(model, input_shapes=[(2,)])


class FixedWires(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("index", torch.tensor([[2, 0], [1, 1]]))

    def forward(self, inputs):
        routed = inputs.gather(1, self.index)
        return routed[:, 0].reshape(2, 1).expand_as(routed).transpose(0, 1)


def test_static_gather_is_affine_wiring():
    model = frozen(FixedWires(), torch.zeros(2, 3))
    audit, _ = audit_neural_graph(model, input_shapes=[(2, 3)])
    assert audit["outputs"][0]["shape"] == [2, 2]
    torch.testing.assert_close(model(torch.arange(6.).reshape(2, 3)), torch.tensor([[2., 4.], [2., 4.]]))


def test_shape_metadata_can_construct_affine_wires_without_reading_activations():
    model = frozen(NonNeuralOperation(lambda x: (x + torch.ones_like(x)).reshape(x.size(0), -1)), torch.zeros(2, 3))
    audit, _ = audit_neural_graph(model, input_shapes=[(4, 3)])
    assert audit["outputs"][0]["shape"] == [4, 3]
    assert audit["primitive_families"]["metadata"] >= 1
    torch.testing.assert_close(model(torch.zeros(4, 3)), torch.ones(4, 3))


class Codec(nn.Module):
    def __init__(self, terminal=True):
        super().__init__()
        self.terminal = terminal

    def forward(self, inputs):
        frame = F.relu(inputs).to(torch.uint8)
        return (inputs, frame) if self.terminal else frame.float() + inputs


def test_only_terminal_uint8_codec_is_admitted():
    model = frozen(Codec(), torch.zeros(2, 3))
    audit, _ = audit_neural_graph(model, input_shapes=[(2, 3)])
    assert audit["primitive_families"]["output_codec"] == 1
    assert audit["outputs"][1] == {"shape": [2, 3], "dtype": "torch.uint8", "codec": "uint8"}
    model = frozen(Codec(terminal=False), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="codecs must be terminal"):
        audit_neural_graph(model, input_shapes=[(2, 3)])


def test_saved_graph_requires_shapes_and_repeats_the_same_audit():
    model = frozen(AffineCarry(), torch.zeros(2))
    stream = BytesIO()
    torch.jit.save(model, stream)
    stream.seek(0)
    deployed = torch.jit.load(stream)
    with pytest.raises(ValueError, match="declared static"):
        audit_neural_graph(deployed)
    original = audit_neural_graph(model, input_shapes=[(2,)])[0]
    serialized = audit_neural_graph(deployed, input_shapes=[(2,)])[0]
    assert serialized["inputs"] == original["inputs"]
    assert serialized["outputs"] == original["outputs"]
    # Serialization canonicalizes constants/list nodes. The artifact manifest
    # must record the reloaded graph, as the actual runtime does.
    assert audit_neural_graph(deployed, input_shapes=[(2,)])[0] == serialized


@pytest.mark.parametrize("shape", [[True, 3], [-1, 3], [None, 3]])
def test_contract_shapes_must_be_static_nonnegative_integers(shape):
    model = frozen(NonNeuralOperation(torch.relu), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="static nonnegative shape"):
        audit_neural_graph(model, input_shapes=[shape])


def test_shape_contract_must_cover_inputs_exactly_once():
    model = frozen(DynamicMultiply(), torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match="every tensor input exactly once"):
        audit_neural_graph(model, input_shapes=[(2,)])
    with pytest.raises(ValueError, match="missing tensor input y"):
        audit_neural_graph(model, input_shapes={"x": (2,), "wrong": (2,)})


def test_mutable_parameters_are_rejected_before_operator_audit():
    model = torch.jit.trace(nn.Linear(2, 1), (torch.zeros(2),))
    with pytest.raises(ValueError, match="freeze every parameter"):
        audit_neural_graph(model, input_shapes=[(2,)])


@pytest.mark.parametrize("operation", [lambda x: x.add_(1), lambda x: x.mul_(2)])
def test_inplace_operations_must_be_functionalized(operation):
    model = frozen(NonNeuralOperation(operation), torch.zeros(2))
    with pytest.raises(ValueError, match="outside the neural IR"):
        audit_neural_graph(model, input_shapes=[(2,)])


def test_audit_checks_shape_correctness_instead_of_observed_trace_values():
    model = frozen(AffineCarry(), torch.zeros(2))
    with pytest.raises(ValueError, match="neural IR rejected"):
        audit_neural_graph(model, input_shapes=[(1,)])


class BadConstant(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor([float("inf")]))

    def forward(self, inputs):
        return inputs * self.weight


def test_nonfinite_weights_fail_at_construction_constant():
    model = frozen(BadConstant(), torch.zeros(1))
    with pytest.raises(ValueError, match="construction tensors must be finite"):
        audit_neural_graph(model, input_shapes=[(1,)])


class MixedRecurrentCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("double_weight", torch.diag(torch.tensor([0.5, 2., 1.], dtype=torch.float64)))
        self.register_buffer("double_bias", torch.tensor([0.25, -0.5, 0.125], dtype=torch.float64))
        self.register_buffer("half_weight", torch.tensor([[255., 0., 128.], [0., 255., 129.],
                                                        [16., 128., 255.]], dtype=torch.float16))

    def transition(self, state):
        changed = F.linear(state.double(), self.double_weight, self.double_bias)
        # This wrapped Python literal exceeds float32's exact integer range.
        # Preserving its deliberate double precision changes the returned state.
        return torch.stack((changed[..., 0] + 16777217., changed[..., 1], changed[..., 2]), -1).float()

    def render(self, state):
        neurons = (state > 0).to(torch.float16)
        return F.relu(F.linear(neurons, self.half_weight).float()).to(torch.uint8)

    def forward(self, state):
        state = self.transition(state)
        return state, self.render(state)


@pytest.mark.parametrize("device,jit_device", [("cpu", "cpu"), ("cuda", "cpu"), ("cuda", "cuda")])
def test_fx_conversion_and_fullgraph_compile_preserve_mixed_precision_recurrence(device, jit_device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No Torch GPU executor available")
    state = torch.tensor([[-3., 1., -0.25], [1., 0., 2.]])
    source = MixedRecurrentCodec().eval()
    traced = torch.jit.trace_module(source, {"forward": (state,), "transition": (state,),
                                           "render": (state,)}, check_trace=False)
    model = torch.jit.freeze(traced, preserved_attrs=["transition", "render"])
    artifact = BytesIO()
    torch.jit.save(model, artifact)
    artifact.seek(0)
    deployed = torch.jit.load(artifact, map_location=jit_device)
    converted = neural_method_fx(deployed, input_shapes=[state.shape]).to(device)
    assert {torch.float16, torch.float64} <= {buffer.dtype for buffer in converted.buffers()}
    assert any(buffer.ndim == 0 and buffer.dtype == torch.float64 and buffer.item() == 16777217.
               for buffer in converted.buffers())
    assert converted.meta["neural_ir_audit"] == audit_neural_graph(deployed, input_shapes=[state.shape])[0]
    compiled = torch.compile(converted, fullgraph=True)
    for changed in (state, state.flip(-1), state + 0.5):
        expected_state, expected_frame = deployed(changed.to(jit_device))
        actual_state, actual_frame = compiled(changed.to(device))
        torch.testing.assert_close(actual_state.cpu(), expected_state.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(actual_frame.cpu(), expected_frame.cpu(), rtol=0, atol=0)
        assert actual_state.dtype == torch.float32 and actual_frame.dtype == torch.uint8
    transition = neural_method_fx(deployed, "transition", [state.shape]).to(device)
    render = neural_method_fx(deployed, "render", [state.shape]).to(device)
    split_state = transition(state.to(device))
    expected_split = deployed.transition(state.to(jit_device))
    torch.testing.assert_close(split_state.cpu(), expected_split.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(render(split_state).cpu(), deployed.render(expected_split).cpu(), rtol=0, atol=0)


class ConstantIndexContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("rows", torch.tensor([0, 1, 1, 0]))
        self.register_buffer("columns", torch.tensor([2, 0, 1, 2]))

    def forward(self, inputs):
        return inputs[self.rows, self.columns]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fx_relocates_tensor_constants_inside_frozen_containers_as_static_wires(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No Torch GPU executor available")
    inputs = torch.arange(6.).reshape(2, 3)
    model = frozen(ConstantIndexContainer(), inputs)
    artifact = BytesIO()
    torch.jit.save(model, artifact)
    artifact.seek(0)
    deployed = torch.jit.load(artifact, map_location=device)
    converted = neural_method_fx(deployed, input_shapes=[inputs.shape]).to(device)
    audit = converted.meta["neural_ir_audit"]
    assert audit["constant_tensor_count"] == 2 and audit["constant_tensor_bytes"] == 64
    assert len(list(converted.buffers())) == 2
    assert all(buffer.device.type == device for buffer in converted.buffers())
    compiled = torch.compile(converted, fullgraph=True)
    torch.testing.assert_close(compiled(inputs.to(device)).cpu(), torch.tensor([2., 3., 4., 2.]), rtol=0, atol=0)


class ShapeAndChunks(nn.Module):
    def forward(self, inputs):
        rows = torch.tensor(inputs.size(0), dtype=torch.int64)
        flat = (inputs + torch.ones_like(inputs)).reshape(rows, -1)
        return torch.stack(flat.split(2, dim=0), dim=0).transpose(0, 1)


def test_fx_folds_shape_integer_and_constant_neurons_and_retains_static_chunk_wires():
    inputs = torch.arange(12.).reshape(4, 3)
    model = frozen(ShapeAndChunks(), inputs)
    audit, original = audit_neural_graph(model, input_shapes=[inputs.shape])
    assert "aten::Int" in original and audit["primitive_families"]["metadata"] >= 1
    converted = neural_method_fx(model, input_shapes=[inputs.shape])
    targets = {str(node.target) for node in converted.graph.nodes if node.op == "call_function"}
    assert not any("aten.size" in target or "aten.Int" in target or "NumToTensor" in target for target in targets)
    assert str(operator.getitem) in targets
    compiled = torch.compile(converted, fullgraph=True)
    torch.testing.assert_close(compiled(inputs + 0.5), model(inputs + 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("operation", [lambda x: x * x, torch.exp, lambda x: x[(x > 0).long()]])
def test_fx_converter_rejects_non_neural_graph_before_conversion(operation):
    model = frozen(NonNeuralOperation(operation), torch.tensor([1., 2.]))
    with pytest.raises(ValueError, match="neural IR rejected"):
        neural_method_fx(model, input_shapes=[(2,)])
