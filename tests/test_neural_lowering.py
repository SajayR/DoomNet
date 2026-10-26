import pytest
import torch
from torch import nn

from doomnet.neural_ir import audit_neural_graph
from doomnet.neural_lowering import lower_neural_method


def lower(module, examples, domains):
    source = torch.jit.freeze(torch.jit.trace(module.eval(), examples, check_trace=False))
    compiled, report = lower_neural_method(source, "forward", domains, source_modules=module)
    neural = torch.jit.freeze(torch.jit.trace(compiled, examples, check_trace=False))
    audit_neural_graph(neural, input_shapes=[x.shape for x in examples])
    return compiled, report


class Rules(nn.Module):
    def forward(self, x, y):
        flags = torch.stack((x > y, x >= y, x < y, x <= y, x == y, x != y))
        boolean = torch.stack(((x > 0) & (y < 0), (x > 0) | (y < 0), ~(x > 0)))
        routed = torch.where(x > 0, y, x)
        extrema = torch.stack((torch.minimum(x, y), torch.maximum(x, y), x.clamp(-1, 1),
                               x.clamp_min(-1), x.clamp_max(1), x.abs()))
        counts = torch.stack((flags.any(0), flags.all(0))).float()
        return torch.cat((flags.float(), boolean.float(), routed[None], extrema, counts), 0)


def test_boolean_comparisons_signed_routing_and_extrema_are_neural():
    x = torch.tensor([-2., -1., -0., 0., 1e-7, .5, 1., 2.])
    y = torch.tensor([1., -1., 0., 1., -1e-7, 1., 2., -2.])
    bounds = [(torch.full_like(x, -2.), torch.full_like(x, 2.))] * 2
    compiled, report = lower(Rules(), (x, y), bounds)
    assert torch.equal(compiled(x, y), Rules()(x, y))
    assert report["compiler"] == "JIT-to-neural-FX"
    assert "aten::where" in report["source_operators"]


class ReadMemory(nn.Module):
    def forward(self, values, choose):
        index = (choose > 0).long()
        return values.index_select(0, index)


def test_dynamic_integer_read_uses_constructed_addressing():
    values = torch.tensor([[1e-7, -1e-7], [-1e6, 1e6]])
    choose = torch.tensor([-1., 1., -1.])
    compiled, report = lower(ReadMemory(), (values, choose),
        [(torch.full_like(values, -1e6), torch.full_like(values, 1e6)),
         (torch.full_like(choose, -1.), torch.full_like(choose, 1.))])
    assert torch.equal(compiled(values, choose), ReadMemory()(values, choose))
    assert any(x["construction"]["type"] == "NeuralRead" for x in report["circuits"])


class GatherMemory(nn.Module):
    def forward(self, values, choose):
        return values.gather(1, (choose > 0).long())


def test_batched_dynamic_gather_uses_per_batch_memory():
    values = torch.tensor([[-3., 1.], [2., -4.]])
    choose = torch.tensor([[-1., 1., -1.], [1., 1., -1.]])
    compiled, _ = lower(GatherMemory(), (values, choose),
        [(torch.full_like(values, -4.), torch.full_like(values, 4.)),
         (torch.full_like(choose, -1.), torch.full_like(choose, 1.))])
    assert torch.equal(compiled(values, choose), GatherMemory()(values, choose))


class Artwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.tensor([[0, 17, 255], [31, 0, 98]], dtype=torch.uint8))

    def forward(self, choose):
        return self.table[(choose > 0).long()].float()


def test_fixed_byte_artwork_executes_a_neural_rom():
    choose = torch.tensor([-1., 1., 0., 1.])
    compiled, report = lower(Artwork(), (choose,),
                             [(torch.full_like(choose, -1.), torch.full_like(choose, 1.))])
    assert torch.equal(compiled(choose), Artwork()(choose))
    assert any(x["construction"]["type"] == "PaletteROM" for x in report["circuits"])


class AdvancedMemory(nn.Module):
    def forward(self, values, a, b):
        return values[(a > 0).long(), (b > 0).long()]


def test_advanced_addresses_have_fixed_affine_coordinate_wiring():
    values = torch.arange(12.).reshape(2, 2, 3)
    a = torch.tensor([[-1.], [1.]])
    b = torch.tensor([[-1., 1., -1.]])
    compiled, _ = lower(AdvancedMemory(), (values, a, b),
        [(torch.zeros_like(values), torch.full_like(values, 11.)),
         (torch.full_like(a, -1.), torch.full_like(a, 1.)),
         (torch.full_like(b, -1.), torch.full_like(b, 1.))])
    assert torch.equal(compiled(values, a, b), AdvancedMemory()(values, a, b))


class StaticUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("index", torch.tensor([2, 0]))

    def forward(self, original, updated):
        return original.index_copy(0, self.index, updated)


def test_fixed_index_copy_becomes_static_wiring():
    original = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    updated = torch.tensor([[7., 8.], [9., 10.]])
    compiled, _ = lower(StaticUpdate(), (original, updated),
        [(torch.zeros_like(original), torch.full_like(original, 6.)),
         (torch.zeros_like(updated), torch.full_like(updated, 10.))])
    assert torch.equal(compiled(original, updated), StaticUpdate()(original, updated))


class Codec(nn.Module):
    def forward(self, x):
        return x.clamp(0, 255).to(torch.uint8)


def test_final_byte_conversion_remains_an_output_codec():
    x = torch.tensor([-1., 1.5, 254.5, 256.])
    compiled, _ = lower(Codec(), (x,), [(torch.full_like(x, -1.), torch.full_like(x, 256.))])
    assert torch.equal(compiled(x), Codec()(x))


class Unsupported(nn.Module):
    def forward(self, x):
        return x.pow(3)


def test_unknown_source_operator_fails_instead_of_remaining_in_graph():
    x = torch.zeros(2)
    source = torch.jit.freeze(torch.jit.trace(Unsupported().eval(), x))
    with pytest.raises(ValueError, match="unsupported source operator aten::pow"):
        lower_neural_method(source, "forward", [(torch.full_like(x, -1.), torch.ones_like(x))])


def test_input_domain_shape_and_finiteness_are_essential_validation():
    x = torch.zeros(2)
    source = torch.jit.freeze(torch.jit.trace(Codec().eval(), x))
    with pytest.raises(ValueError, match="finite, ordered"):
        lower_neural_method(source, "forward", [(torch.zeros(2), torch.tensor([1., float('inf')]))])
    with pytest.raises(ValueError, match="every tensor input"):
        lower_neural_method(source, "forward", [])


class CarryPairs(nn.Module):
    def __init__(self, binary, crossed=False):
        super().__init__()
        self.binary, self.crossed = binary, crossed

    def forward(self, x, g):
        gate = (g > 0).float() if self.binary else g
        values = torch.stack((x, x), -1)
        off = torch.stack((gate, 1 - gate), -1)
        carried = torch.relu(values - off * 10)
        if self.crossed:
            return carried[0, 0] + carried[1, 1]
        return carried[:, 0] + carried[:, 1]


def test_complementary_carry_proof_requires_binary_gates():
    x, g = torch.ones(2), torch.full((2,), .5)
    compiled, report = lower(CarryPairs(False), (x, g),
        [(torch.ones(2), torch.ones(2)), (torch.zeros(2), torch.ones(2))])
    assert torch.equal(compiled(x, g), torch.zeros(2))
    assert report["outputs"][0]["minimum"] == 0


def test_complementary_carry_proof_does_not_mix_batch_coordinates():
    x, g = torch.full((2,), 10.), torch.tensor([-1., 1.])
    compiled, report = lower(CarryPairs(True, crossed=True), (x, g),
        [(torch.zeros(2), torch.full((2,), 10.)), (torch.full((2,), -1.), torch.ones(2))])
    assert compiled(x, g) == 20
    assert report["outputs"][0]["maximum"] >= 20


class ClauseRouting(nn.Module):
    def __init__(self):
        super().__init__()
        from doomnet.circuits import BooleanCircuit
        self.rule = BooleanCircuit(2, [[(1, 2)]])

    def forward(self, x):
        gate = self.rule(torch.stack((x >= 0, x <= 64), -1).float()).squeeze(-1) > 0
        return torch.where(gate, x, torch.zeros_like(x))


def test_encoded_boolean_clauses_refine_routed_output_domains():
    x = torch.tensor([-1e6, 0., 32., 64., 1e6])
    compiled, report = lower(ClauseRouting(), (x,),
        [(torch.full_like(x, -1e6), torch.full_like(x, 1e6))])
    assert torch.equal(compiled(x), ClauseRouting()(x))
    assert report["outputs"][0]["minimum"] == 0
    assert report["outputs"][0]["maximum"] == 64


class PartitionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("paths", torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]))
        self.register_buffer("bias", torch.tensor([1., 0., 0., -1.]))
        self.register_buffer("codes", torch.tensor([[0., 1., 2., 3.]]))
        self.register_buffer("memory", torch.tensor([7., 11., 13., 17.]))

    def forward(self, x):
        leaves = torch.relu(torch.nn.functional.linear((x > 0).float(), self.paths, self.bias))
        index = torch.nn.functional.linear(leaves, self.codes).squeeze(-1).long()
        return self.memory[index]


def test_pairwise_exclusive_relu_clauses_certify_decoder_addresses():
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    compiled, _ = lower(PartitionDecoder(), (x,),
        [(torch.full_like(x, -1.), torch.ones_like(x))])
    assert torch.equal(compiled(x), PartitionDecoder()(x))


class SelectorRead(nn.Module):
    def __init__(self, selector_type=None):
        super().__init__()
        from doomnet.circuits import NearestCircuit
        self.selector = (selector_type or NearestCircuit)(3, 2)
        self.register_buffer("memory", torch.tensor([11., 13., 17.]))

    def forward(self, x):
        _, index = self.selector(x)
        return self.memory[index]


def test_verified_selector_contract_excludes_fixed_padding_addresses():
    x = torch.full((2, 3), 1e6)
    compiled, report = lower(SelectorRead(), (x,), [(torch.zeros_like(x), torch.full_like(x, 1e6))])
    assert torch.equal(compiled(x), torch.tensor([[11., 13.], [11., 13.]]))
    contract = report["verified_subcircuits"][0]
    assert contract["module_path"] == "selector"
    assert len(contract["constructor_sha256"]) == 64
    assert contract["input_checks"] == contract["index_checks"] == 1


def test_selector_contract_rejects_tampered_constructor_buffers():
    module = SelectorRead()
    module.selector.carry_weight[0] = -999999
    x = torch.zeros(2, 3)
    source = torch.jit.freeze(torch.jit.trace(module.eval(), x))
    with pytest.raises(ValueError, match="modified construction weights or topology"):
        lower_neural_method(source, "forward", [(torch.zeros_like(x), torch.full_like(x, 1e6))], source_modules=module)


def test_selector_contract_rejects_unproven_input_domains():
    module = SelectorRead()
    x = torch.zeros(2, 3)
    source = torch.jit.freeze(torch.jit.trace(module.eval(), x))
    with pytest.raises(ValueError, match="requires proven scores"):
        lower_neural_method(source, "forward", [(torch.zeros_like(x), torch.full_like(x, 1e6 + 1))], source_modules=module)


def test_selector_contract_rejects_unsupported_subclasses():
    from doomnet.circuits import NearestCircuit

    class UnsupportedSelector(NearestCircuit):
        pass

    module = SelectorRead(UnsupportedSelector)
    x = torch.zeros(2, 3)
    source = torch.jit.freeze(torch.jit.trace(module.eval(), x))
    with pytest.raises(ValueError, match="exact supported class"):
        lower_neural_method(source, "forward", [(torch.zeros_like(x), torch.full_like(x, 1e6))], source_modules=module)


class ProjectedRouting(nn.Module):
    def __init__(self, crossed):
        super().__init__()
        self.crossed = crossed

    def forward(self, x):
        gate = ((x >= 0) & (x <= 1))[0]
        return torch.where(gate, x[1 if self.crossed else 0], torch.zeros_like(x[0]))


@pytest.mark.parametrize("crossed", [False, True])
def test_branch_domains_project_through_matching_static_coordinates(crossed):
    x = torch.tensor([[.5, .5], [2., 2.]])
    compiled, report = lower(ProjectedRouting(crossed), (x,),
        [(torch.full_like(x, -2.), torch.full_like(x, 2.))])
    assert torch.equal(compiled(x), ProjectedRouting(crossed)(x))
    assert report["outputs"][0]["maximum"] == (2 if crossed else 1)


class Products(nn.Module):
    def forward(self, wide, narrow):
        return wide * narrow


def test_binary_product_quantizes_the_smaller_domain_first():
    wide, narrow = torch.tensor([.25]), torch.tensor([128.])
    compiled, report = lower(Products(), (wide, narrow),
        [(torch.tensor([-2e12]), torch.tensor([2e12])),
         (torch.tensor([-512.]), torch.tensor([512.]))])
    assert compiled(wide, narrow) == 32
    assert report["circuits"][0]["construction"]["input_bounds"][0] == 512


class PositiveArithmetic(nn.Module):
    def forward(self, x):
        return x.square(), x.sqrt()


def test_positive_square_and_sqrt_domains_allow_approximation_undershoot():
    x = torch.tensor([1.3333334])
    compiled, report = lower(PositiveArithmetic(), (x,), [(x, x)])
    for output, exact, domain in zip(compiled(x), PositiveArithmetic()(x), report["outputs"], strict=True):
        assert domain["minimum"] < exact.item()
        assert domain["minimum"] <= output.item() <= domain["maximum"]


class SafeDivision(nn.Module):
    def forward(self, denominator):
        safe = torch.where(denominator.abs() < .1, torch.ones_like(denominator), denominator)
        return torch.ones_like(denominator) / safe


def test_safe_denominator_condition_proves_a_signed_nonzero_union():
    denominator = torch.tensor([-2., -.1, 0., .1, 2.])
    compiled, _ = lower(SafeDivision(), (denominator,),
        [(torch.full_like(denominator, -2.), torch.full_like(denominator, 2.))])
    assert torch.allclose(compiled(denominator), SafeDivision()(denominator), atol=2e-6, rtol=0)


class IntegerMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("flags", torch.tensor([0, 24]))

    def forward(self, x):
        flags = self.flags[(x > 0).long()]
        return flags & 16, flags | 8


def test_fixed_integer_bit_masks_lower_to_digit_and_boolean_neurons():
    x = torch.tensor([-1., 1.])
    compiled, _ = lower(IntegerMask(), (x,), [(torch.full_like(x, -1.), torch.ones_like(x))])
    for result, expected in zip(compiled(x), IntegerMask()(x), strict=True):
        assert torch.equal(result, expected)


def test_temporary_construction_constants_keep_distinct_values():
    from doomnet.neural_lowering import _Compiler

    identity = torch.jit.freeze(torch.jit.trace(nn.Identity().eval(), torch.zeros(1)))
    compiler = _Compiler(identity, "forward", [(torch.zeros(1), torch.ones(1))], None)
    references = [compiler.constant(torch.tensor(float(index))) for index in range(128)]
    compiler.graph.output(compiler.emit(torch.stack, references))
    compiled = torch.fx.GraphModule(compiler.root, compiler.graph)
    assert torch.equal(compiled(torch.zeros(1)), torch.arange(128.))
