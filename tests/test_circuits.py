"""Constructed rules, stable visibility, exact score carries, and portable graphs."""

import io
import itertools
import json

import pytest
import torch

from doomnet.circuits import BooleanCircuit, MinimumCircuit, NearestCircuit


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(threads)


def assert_selection(circuit, scores):
    ordered, indices = torch.sort(scores, stable=True)
    actual, index = circuit(scores)
    if isinstance(circuit, MinimumCircuit):
        ordered, indices = ordered[..., 0], indices[..., 0]
    else:
        ordered, indices = ordered[..., :circuit.output_count], indices[..., :circuit.output_count]
    torch.testing.assert_close(actual, ordered, rtol=0, atol=0)
    assert torch.equal(index, indices)


@pytest.mark.parametrize("count", range(1, 8))
def test_every_small_configuration_is_stable_for_every_k(count):
    scores = torch.tensor(list(itertools.product((0., 0.125, 1e6), repeat=count)))
    for k in range(1, count + 1):
        assert_selection(NearestCircuit(count, k), scores)
    assert_selection(MinimumCircuit(count), scores)


@pytest.mark.parametrize("count,k", [(512, 64), (475, 64), (273, 24)])
def test_realistic_columns_match_stable_sort_exactly(count, k):
    generator = torch.Generator().manual_seed(41)
    scores = torch.rand((160, count), generator=generator) * 4000
    scores[torch.rand(scores.shape, generator=generator) > 0.2] = 1e6
    scores[:, :12] = 100.125  # Frequent equal geometric depths.
    scores[0] = 1e6
    assert_selection(NearestCircuit(count, k), scores)
    assert_selection(MinimumCircuit(count), scores)


def test_relu_carries_preserve_small_scores_beside_far_sentinels():
    values = torch.tensor([0., 1e-4, 0.015625, 0.1, 1.0001, 31.0001,
                           999999.75, 999999.875, 999999.9375, 1e6])
    scores = values.repeat(3).roll(13).reshape(2, 3, 5)
    for k in (1, 3, 5):
        assert_selection(NearestCircuit(5, k), scores)
    assert_selection(MinimumCircuit(5), scores)
    for count in (24, 64):
        pixels = torch.full((3, 4, count), 1e6)
        pixels[0, :, -1] = 0.1
        pixels[1, :, :2] = 999999.9375
        assert_selection(MinimumCircuit(count), pixels)


@pytest.mark.parametrize("circuit", [NearestCircuit(7, 3), MinimumCircuit(7)])
def test_selection_preserves_empty_batch_shapes_without_inferred_wiring(circuit):
    scores = torch.empty((2, 0, circuit.input_count))
    assert_selection(circuit, scores)
    frozen = torch.jit.freeze(torch.jit.trace(circuit.eval(), (torch.zeros(2, 1, circuit.input_count),)))
    actual, index = frozen(scores)
    expected, expected_index = circuit(scores)
    assert torch.equal(actual, expected) and torch.equal(index, expected_index)
    assert actual.shape == ((2, 0) if isinstance(circuit, MinimumCircuit) else (2, 0, 3))


@pytest.mark.parametrize("circuit", [NearestCircuit(9, 5), MinimumCircuit(9)])
def test_frozen_graph_uses_affine_neurons_and_static_wires(circuit):
    scores = torch.tensor([[1e6, 0.1, 9., 9., 0.1, 7., 6., 1e6, 4.]])
    with torch.inference_mode():
        frozen = torch.jit.freeze(torch.jit.trace(circuit.eval(), (scores,), check_trace=False))
        expected, index = circuit(scores.flip(-1))
        actual, actual_index = frozen(scores.flip(-1))
    assert torch.equal(actual, expected)
    assert torch.equal(actual_index, index)
    graph = str(frozen.inlined_graph)
    for forbidden in ("aten::sort", "aten::topk", "aten::min", "aten::argmin",
                      "aten::where", "aten::gather", "PythonOp", "prim::If", "prim::Loop"):
        assert forbidden not in graph
    for operation in ("aten::relu", "aten::gt", "aten::index_select", "aten::mul", "aten::add"):
        assert operation in graph
    if isinstance(circuit, NearestCircuit):
        assert "aten::linear" in graph
    json.dumps(circuit.construction_spec())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No Torch GPU executor available")
@pytest.mark.parametrize("circuit", [NearestCircuit(475, 64), MinimumCircuit(64)])
def test_cpu_frozen_selection_relocates_to_gpu_without_changing_winners(circuit):
    count = circuit.input_count
    generator = torch.Generator().manual_seed(8)
    scores = torch.rand((160, count), generator=generator) * 2500
    scores[torch.rand(scores.shape, generator=generator) > 0.1] = 1e6
    scores[0] = 1e6
    scores[1, -3:] = torch.tensor([0.1, 0.1, 0.1])
    with torch.inference_mode():
        frozen = torch.jit.freeze(torch.jit.trace(circuit.eval(), (scores,), check_trace=False))
        artifact = io.BytesIO()
        torch.jit.save(frozen, artifact)
        artifact.seek(0)
        gpu = torch.jit.load(artifact, map_location="cuda")
        actual, index = gpu(scores.cuda())
        expected, expected_index = frozen(scores)
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(index.cpu(), expected_index)


def test_boolean_circuit_compiles_signed_clauses_and_shared_rules():
    rules = [[(1, -2), (2, 3)], [(-1, -3), (1, -2)], [], [()]]
    circuit = BooleanCircuit(3, rules)
    inputs = torch.tensor(list(itertools.product((0., 1.), repeat=3)))
    expected = torch.tensor([[float(any(all(bool(row[abs(literal) - 1]) == (literal > 0)
                                          for literal in clause) for clause in output))
                              for output in rules] for row in inputs])
    assert torch.equal(circuit(inputs), expected)
    assert len(circuit.clause_bias) == 4
    assert torch.equal(circuit(inputs.reshape(2, 4, 3)), expected.reshape(2, 4, 4))
    circuit.validate_input(inputs)
    json.dumps(circuit.construction_spec())
    with torch.inference_mode():
        frozen = torch.jit.freeze(torch.jit.trace(circuit.eval(), (inputs,), check_trace=False))
    assert torch.equal(frozen(inputs.flip(0)), expected.flip(0))
    graph = str(frozen.inlined_graph)
    assert graph.count("aten::linear") == 2
    assert graph.count("aten::gt") == 2


def test_boolean_outputs_without_clauses_are_false():
    circuit = BooleanCircuit(2, [[], []])
    inputs = torch.tensor([[0., 0.], [1., 1.]])
    assert torch.equal(circuit(inputs), torch.zeros(2, 2))
    with torch.inference_mode():
        frozen = torch.jit.freeze(torch.jit.trace(circuit.eval(), (inputs,), check_trace=False))
    assert torch.equal(frozen(inputs), torch.zeros(2, 2))


@pytest.mark.parametrize("rules", [[], [1], [[1]], [[(0,)]], [[(4,)]], [[(True,)]],
                                 [[(1, 1)]], [[(1, -1)]], [[(1.0,)]], [[(-4,)]]])
def test_boolean_constructor_rejects_invalid_rules(rules):
    with pytest.raises(ValueError):
        BooleanCircuit(3, rules)


@pytest.mark.parametrize("count,k,bound", [(0, 1, 1e6), (True, 1, 1e6), (2.5, 1, 1e6),
                                         (2, 0, 1e6), (2, 3, 1e6), (2, True, 1e6),
                                         (2, 1, 0), (2, 1, -1), (2, 1, float("nan")),
                                         (2, 1, float("inf")), (2, 1, True),
                                         (2 ** 24 + 1, 1, 1e6)])
def test_selection_constructor_rejects_invalid_dimensions_and_bounds(count, k, bound):
    with pytest.raises(ValueError):
        NearestCircuit(count, k, bound)


@pytest.mark.parametrize("scores", [torch.tensor(1.), torch.tensor([1.]), torch.tensor([1, 2]),
                                  torch.tensor([1., float("nan")]), torch.tensor([1., float("inf")]),
                                  torch.tensor([-1., 1.]), torch.tensor([1., 1e6 + 1])])
def test_external_scores_are_validated_on_host(scores):
    with pytest.raises(ValueError):
        MinimumCircuit(2).validate_input(scores)


@pytest.mark.parametrize("inputs", [torch.tensor([1, 0]), torch.tensor([0., 0.5]),
                                  torch.tensor([0., float("nan")]), torch.tensor([0.])])
def test_external_boolean_activations_are_validated_on_host(inputs):
    with pytest.raises(ValueError):
        BooleanCircuit(2, [[(1,)]]).validate_input(inputs)
