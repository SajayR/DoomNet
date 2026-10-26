"""Executed arithmetic circuits, domain accuracy, and strict frozen graphs."""

import io
import json
import math

import pytest
import torch

from doomnet.neural_ir import audit_neural_graph
from doomnet.neural_math import (Atan2, BinaryMultiply, BinarySquare, Cos, Divide, Floor, Frac,
                                PiecewiseLinear, Remainder, Round, Sin, SinCos,
                                Sqrt, Trunc, bounded_gate, clamp,
                                hard_step)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(threads)


def freeze(module, *inputs):
    with torch.inference_mode():
        return torch.jit.freeze(torch.jit.trace(module.eval(), inputs, check_trace=False))


def test_step_and_bounded_signed_carries_preserve_small_values():
    values = torch.tensor([-1., -1e-7, -0., 0., 1e-7, 1.])
    assert torch.equal(hard_step(values), torch.tensor([0., 0., 0., 0., 1., 1.]))
    for gate in (0., 1.):
        actual = bounded_gate(values, torch.full_like(values, gate), 1)
        assert torch.equal(actual, values if gate else torch.zeros_like(values))
    assert torch.equal(clamp(values * 2, -1, 1), values.mul(2).clamp(-1, 1))
    extreme = torch.tensor([-torch.finfo(torch.float32).max, -1e6, -.0001,
                            0, .0001, 1e6, torch.finfo(torch.float32).max])
    assert torch.equal(clamp(extreme, -.1, .1), extreme.clamp(-.1, .1))


def test_geometric_product_accuracy_in_float64():
    circuit = BinaryMultiply(4000, 1)
    generator = torch.Generator().manual_seed(41)
    left = (torch.rand(20001, generator=generator, dtype=torch.float64) * 2 - 1) * 4000
    right = torch.rand(20001, generator=generator, dtype=torch.float64) * 2 - 1
    left[-7:] = torch.tensor([-0.1, -0.0001, -1e-7, 0, 1e-7, 0.0001, 0.1])
    actual = circuit(left, right)
    assert (actual - left * right).abs().max() <= circuit.error_bound + 2e-12
    assert torch.equal(circuit(left, torch.zeros_like(right)), torch.zeros_like(left))
    assert torch.equal(circuit(torch.zeros_like(left), right), torch.zeros_like(right))


def test_binary_multiplier_handles_large_sentinels_and_exact_integer_products():
    left = torch.arange(-255, 256).reshape(-1, 1).float()
    right = torch.arange(-127, 128).reshape(1, -1).float()
    assert torch.equal(BinaryMultiply(256, 128)(left, right), left * right)
    values = torch.tensor([-1e6, -1., -1e-4, 0, 1e-4, 1., 1e6], dtype=torch.float64)
    coefficients = torch.tensor([0., 1., -1., 0., 0.5, -0.5, 1.], dtype=torch.float64)
    circuit = BinaryMultiply(1e6, 1, bits=40)
    assert (circuit(values, coefficients) - values * coefficients).abs().max() < 1e-6


def test_float32_product_roundoff_is_small_across_map_coordinates():
    left = torch.linspace(-4000, 4000, 20001)
    right = torch.sin(torch.arange(20001) * 0.1234)
    circuit = BinaryMultiply(4000, 1)
    assert (circuit(left, right) - left * right).abs().max() <= 0.0015
    assert (circuit(left, right) - left * right).abs().max() <= circuit.construction_spec()["float32_absolute_error_bound"]
    tiny = torch.tensor([-0.1, -0.0001, -1e-7, 0, 1e-7, 0.0001, 0.1])
    assert (BinaryMultiply(4000, 1)(tiny, torch.ones_like(tiny)) - tiny).abs().max() < 4e-6


def test_correlated_binary_square_preserves_small_collision_distances():
    values = torch.tensor([-8192, -1, -.1, -.001, -.0001, 0, .0001, .001, .1, 1, 8192], dtype=torch.float64)
    circuit = BinarySquare(8192)
    error = (circuit(values) - values.square()).abs()
    assert bool((error <= values.abs() * circuit.pointwise_error_coefficient + 1e-15).all())
    assert circuit(values)[3] <= 1e-6
    assert bool((circuit(values) >= 0).all())


def test_declared_output_ranges_hold_at_dense_float32_extrema():
    values = torch.linspace(-1, 1, 20001)
    for circuit in (BinarySquare(1), BinaryMultiply(1, 1)):
        actual = circuit(values) if isinstance(circuit, BinarySquare) else circuit(values, values.flip(0))
        low, high = circuit.construction_spec()["output_activation_bounds"]
        assert bool(((actual >= low) & (actual <= high)).all())
    angles = torch.linspace(-4 * math.pi, 4 * math.pi, 20001)
    trig = SinCos(4 * math.pi)(angles)
    assert bool(((trig >= -1) & (trig <= 1)).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_binary_division_accuracy_across_denominator_octaves(dtype):
    denominator = torch.logspace(-16, 13, 4001, base=2, dtype=dtype)
    ratios = torch.linspace(-4096, 4096, 4001, dtype=dtype)
    numerator = denominator * ratios
    circuit = Divide(8192 * 4096, 8192, 4096, fractional_bits=24)
    actual = circuit(numerator, denominator)
    expected = numerator / denominator
    tolerance = 0.0005 if dtype == torch.float32 else circuit.error_bound + 2e-12
    assert (actual - expected).abs().max() <= tolerance
    assert torch.equal(circuit(numerator, -denominator), -actual)
    assert circuit(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype)) == 0
    assert circuit(torch.tensor(4096., dtype=dtype), torch.tensor(1., dtype=dtype)) == 4096


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sqrt_has_bounded_error_and_defined_zero(dtype):
    values = torch.logspace(-20, 26, 16001, base=2, dtype=dtype)
    square_root = Sqrt(2 ** 26, dtype=dtype)
    root = values.sqrt()
    roundoff = 4e-7 if dtype == torch.float32 else 2e-15
    assert ((square_root(values) - root) / root).abs().max() <= square_root.relative_error_bound + roundoff
    tiny = torch.linspace(0, square_root.minimum, 2001, dtype=dtype)
    assert (square_root(tiny) - tiny.sqrt()).abs().max() <= square_root.zero_interval_error_bound + 1e-12
    assert square_root(torch.tensor(0., dtype=dtype)) == 0


class DamageDistance(torch.nn.Module):
    """The constructed square/sum/root path with E1M1 coordinate headroom."""

    def __init__(self):
        super().__init__()
        self.square = BinarySquare(8192)
        self.root = Sqrt(2 ** 26)

    def forward(self, delta):
        return self.root(self.square(delta).sum(-1))


@pytest.mark.parametrize("executor", ["cpu", "cuda_compiled"])
def test_default_damage_distance_precision_covers_blast_witness_and_near_zero(executor):
    if executor == "cuda_compiled" and not torch.cuda.is_available():
        pytest.skip("No Torch GPU executor available")
    # Artifact 9c8b1dc9's green_blast:barrel_blast event failed the unchanged
    # health budget: source 68.6474456787 versus neural 68.6472167969. Its
    # captured delta isolates the square-root interpolation error; these are
    # arithmetic diagnostic inputs, never authored gameplay states.
    witness = torch.tensor([[26.024658203125, 76.3330078125]])
    angles = torch.tensor([0., math.pi / 8, math.pi / 4, math.pi / 3, math.pi / 2])
    direction = torch.stack((angles.cos(), angles.sin()), -1)
    radii = torch.linspace(0, 128, 8193)
    dense = (radii[:, None, None] * direction[None, :, :]).reshape(-1, 2)
    tiny_radii = torch.cat((torch.logspace(-10, -1, 513), torch.linspace(0, 2 ** -9, 2049)))
    tiny = (tiny_radii[:, None, None] * direction[None, :, :]).reshape(-1, 2)
    values = torch.cat((witness, dense, tiny))
    expected = values.square().sum(-1).sqrt()
    model = DamageDistance().eval()
    if executor == "cuda_compiled":
        model = torch.compile(model.cuda(), fullgraph=True)
        values = values.cuda()
    # Use a constant small batch for compiled guards and bounded test memory.
    size = 1024
    padded = torch.cat((values, values[-1:].expand((-len(values)) % size, -1)))
    with torch.inference_mode():
        actual = torch.cat([model(batch).cpu() for batch in padded.split(size)])[:len(values)]
    error = (actual - expected).abs()
    assert error[0] <= 1e-5, "captured barrel-blast distance must have margin below the health budget"
    assert error[1:1 + len(dense)].max() <= 4e-5
    assert error[1 + len(dense):].max() <= 3e-5
    assert actual[1] == 0
    # This continuous accuracy check does not promise identical integer armor
    # absorption at every float32 rounding threshold.


@pytest.mark.parametrize("cls,native", [(Floor, torch.floor), (Trunc, torch.trunc),
                                     (Frac, torch.frac), (Round, torch.round)])
def test_signed_quantization_preserves_tiny_negative_values_and_even_ties(cls, native):
    generator = torch.Generator().manual_seed(17)
    values = (torch.rand(20001, generator=generator) * 2 - 1) * 2 ** 24
    boundaries = torch.arange(-100, 101).float()
    close = torch.tensor([-1e-7, 0, 1e-7])
    values = torch.cat((values, boundaries, boundaries - 0.5, boundaries + 0.5,
                        close, close - 1, close + 1, torch.tensor([-2 ** 24, 2 ** 24]).float()))
    assert torch.equal(cls(2 ** 24)(values), native(values))
    assert torch.equal(cls(0.5)(close), native(close))


@pytest.mark.parametrize("modulus,bound", [(64, 2 ** 24), (32749, 32749 * 75 + 74), (10, 999), (8, 4096)])
def test_integer_remainder_exactly_matches_rng_and_texel_domains(modulus, bound):
    generator = torch.Generator().manual_seed(7)
    values = torch.randint(-bound, bound + 1, (20001,), generator=generator).float()
    multiples = torch.arange(-bound // modulus + 1, bound // modulus + 1).float() * modulus
    values = torch.cat((values, multiples, torch.tensor([-1., 0., 1., -bound, bound])))
    assert torch.equal(Remainder(modulus, bound)(values), torch.remainder(values, modulus))
    fractions = torch.tensor([-1e-7, -0.5, 0, 0.5, 1e-7])
    addressed = Remainder(modulus, bound)(Floor(bound)(fractions))
    assert torch.equal(addressed, torch.remainder(fractions.floor(), modulus))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_periodic_trig_matches_native_across_wraps(dtype):
    values = torch.linspace(-4 * math.pi, 4 * math.pi, 20001, dtype=dtype)
    circuit = SinCos(4 * math.pi, dtype=dtype)
    expected = torch.stack((values.sin(), values.cos()), -1)
    tolerance = circuit.error_bound + (4e-6 if dtype == torch.float32 else 3e-15)
    assert (circuit(values) - expected).abs().max() <= tolerance
    assert torch.equal(Sin(4 * math.pi, dtype=dtype)(values), circuit(values)[..., 0])
    assert torch.equal(Cos(4 * math.pi, dtype=dtype)(values), circuit(values)[..., 1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_atan2_covers_octants_axes_and_origin(dtype):
    coordinates = torch.tensor([-4096, -1, -1e-4, 0, 1e-4, 1, 4096], dtype=dtype)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    circuit = Atan2(4096, dtype=dtype)
    assert (circuit(y, x) - torch.atan2(y, x)).abs().max() <= circuit.error_bound + 5e-7
    assert circuit(torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype)) == 0
    tiny = torch.tensor([1e-20, 2e-20, 1e-10, 1], dtype=dtype)
    assert (circuit(tiny, tiny * 0.5) - torch.atan2(tiny, tiny * 0.5)).abs().max() < 2e-6


def test_piecewise_linear_handles_nonuniform_knots_and_vector_outputs():
    circuit = PiecewiseLinear([0, 1, 3, 8], [[0, 8], [1, 6], [5, 2], [0, -3]], dtype=torch.float64)
    values = torch.tensor([0., .5, 1., 2., 3., 5.5, 8.], dtype=torch.float64)
    expected = torch.tensor([[0, 8], [.5, 7], [1, 6], [3, 4], [5, 2], [2.5, -.5], [0, -3]], dtype=torch.float64)
    torch.testing.assert_close(circuit(values), expected, rtol=0, atol=0)
    assert torch.equal(circuit(values.reshape(1, -1)), expected.reshape(1, -1, 2))


GRAPH_CASES = [(BinarySquare(4096), (torch.tensor([-.7, .2]),)),
               (BinaryMultiply(4096, 1), (torch.tensor([-.7, .2]), torch.tensor([.3, -.4]))),
               (Divide(4096, 8192, 4096), (torch.tensor([-.7, .2]), torch.tensor([.3, -.4]))),
               (Sqrt(2 ** 26), (torch.tensor([.7, .2]),)),
               (Floor(2 ** 24), (torch.tensor([-.7, .2]),)),
               (Trunc(2 ** 24), (torch.tensor([-.7, .2]),)),
               (Frac(2 ** 24), (torch.tensor([-.7, .2]),)),
               (Round(2 ** 24), (torch.tensor([-.7, .2]),)),
               (Remainder(32749, 32749 * 75 + 74), (torch.tensor([-32750., 32749.]),)),
               (SinCos(4 * math.pi), (torch.tensor([-.7, .2]),)),
               (Atan2(4096), (torch.tensor([-.7, .2]), torch.tensor([.3, -.4])))]


@pytest.mark.parametrize("circuit,inputs", GRAPH_CASES, ids=[type(case[0]).__name__ for case in GRAPH_CASES])
def test_every_frozen_circuit_has_only_fixed_affine_maps_and_activations(circuit, inputs):
    frozen = freeze(circuit, *inputs)
    audit, graph = audit_neural_graph(frozen, input_shapes=[tuple(value.shape) for value in inputs],
                                    input_dtypes=[value.dtype for value in inputs])
    assert audit["opaque_operators"] == audit["dynamic_weights"] == audit["dynamic_indices"] == 0
    assert audit["primitive_families"]["relu"] > 0
    assert all(operation not in graph for operation in ("aten::div", "aten::sqrt", "aten::floor", "aten::remainder"))
    json.dumps(circuit.construction_spec())
    assert not list(circuit.parameters())
    changed = tuple(value.flip(0).repeat(3) for value in inputs)
    assert torch.equal(frozen(*changed), circuit(*changed))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No Torch GPU executor available")
@pytest.mark.parametrize("circuit,inputs", GRAPH_CASES, ids=[type(case[0]).__name__ for case in GRAPH_CASES])
def test_frozen_cpu_circuits_relocate_to_gpu_as_executed_neural_graphs(circuit, inputs):
    frozen = freeze(circuit, *inputs)
    artifact = io.BytesIO()
    torch.jit.save(frozen, artifact)
    artifact.seek(0)
    gpu = torch.jit.load(artifact, map_location="cuda")
    actual = gpu(*(value.cuda() for value in inputs)).cpu()
    expected = frozen(*inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("factory", [lambda: BinarySquare(0),
                                    lambda: BinaryMultiply(float("inf"), 1), lambda: BinaryMultiply(1, 1, bits=True),
                                    lambda: Divide(1, 1, 1, fractional_bits=0), lambda: Sqrt(1, minimum=2),
                                    lambda: Sqrt(-1), lambda: Sqrt(1, minimum=0), lambda: Floor(2 ** 24 + 1),
                                    lambda: Remainder(0, 1), lambda: SinCos(1, segments=0),
                                    lambda: PiecewiseLinear([0, 0], [0, 1]),
                                    lambda: PiecewiseLinear([0, 1], [0]),
                                    lambda: PiecewiseLinear([0, 1], [0, float("nan")])])
def test_invalid_construction_fails_at_the_source(factory):
    with pytest.raises(ValueError):
        factory()


def test_external_domain_validation_is_explicit_and_outside_the_network():
    for values in (torch.tensor([1]), torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.tensor([2.])):
        with pytest.raises(ValueError):
            BinarySquare(1).validate_input(values)
    with pytest.raises(ValueError, match="nonzero"):
        Divide(1, 1, 1).validate_input(torch.tensor(0.), torch.tensor(0.))
    with pytest.raises(ValueError, match="quotient"):
        Divide(2, 1, 1).validate_input(torch.tensor(2.), torch.tensor(1.))
    with pytest.raises(ValueError, match="too small"):
        Divide(1, 1, 1).validate_input(torch.tensor(0.), torch.tensor(1e-40))
    with pytest.raises(ValueError, match="nonnegative"):
        Sqrt(1).validate_input(torch.tensor(-1.))
    with pytest.raises(ValueError, match="knot interval"):
        PiecewiseLinear([1, 2], [1, 4]).validate_input(torch.tensor(0.))
    with pytest.raises(ValueError, match="too small"):
        Atan2(1).validate_input(torch.tensor(1e-40), torch.tensor(1e-40))
