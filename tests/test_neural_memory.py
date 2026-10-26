"""Exact neural addressing, signed carries, recursive page weights and graphs."""

import io
import json
from pathlib import Path

import pytest
import torch

from doomnet.neural_memory import AddressDecoder, FixedROM, NeuralRead, PaletteROM


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("count", [1, 16, 85, 91, 475])
def test_integer_addresses_decode_to_exact_one_hot(count):
    address = torch.arange(count, dtype=torch.float32)
    decoder = AddressDecoder(count)
    assert torch.equal(decoder(address), torch.eye(count))
    decoder.validate_input(address)


@pytest.mark.parametrize("count", [85, 91, 475])
def test_recurrent_reads_preserve_signed_values_exactly(count):
    generator = torch.Generator().manual_seed(count)
    values = (torch.rand(count, generator=generator) - 0.5) * 8000
    values[:6] = torch.tensor([-4000., -0.0001, -0., 0.0001, 0.1, 4000.])
    address = torch.randperm(count, generator=generator).float().reshape(1, -1)
    circuit = NeuralRead(count, 4000)
    circuit.validate_input(address, values)
    assert torch.equal(circuit(address, values), values[address.long()])
    changed = values.flip(0)
    assert torch.equal(circuit(address, changed), changed[address.long()])
    json.dumps(circuit.construction_spec())


def test_memory_batches_and_value_channels_use_fixed_broadcast_wiring():
    generator = torch.Generator().manual_seed(9)
    values = (torch.rand(2, 3, 85, 4, generator=generator) - 0.5) * 10
    address = torch.randint(85, (2, 3, 7, 5), generator=generator).float()
    circuit = NeuralRead(85, 5, (4,))
    circuit.validate_input(address, values)
    expected = torch.stack([torch.stack([values[i, j][address[i, j].long()]
                                        for j in range(3)]) for i in range(2)])
    assert torch.equal(circuit(address, values), expected)
    paired = address[:, :, 0, 0]
    assert torch.equal(circuit(paired, values), values[torch.arange(2)[:, None],
                                                    torch.arange(3)[None, :], paired.long()])


def test_fixed_table_keeps_its_constructed_weights_and_tensor_values():
    table = torch.arange(85 * 6, dtype=torch.float32).reshape(85, 2, 3) * -0.1
    expected = table.clone()
    circuit = FixedROM(table)
    table.fill_(0)
    address = torch.tensor([[84., 0.], [21., 12.]])
    assert torch.equal(circuit(address), expected[address.long()])
    assert FixedROM(torch.tensor([3., -0.1]))(torch.tensor(1.)).item() == torch.tensor(-0.1).item()
    json.dumps(circuit.construction_spec())


@pytest.mark.parametrize("count", [1, 2, 15, 16, 17, 255, 256, 257, 5000])
@pytest.mark.parametrize("matrix_dtype", [torch.float32, torch.float16])
def test_recursive_palette_pages_return_every_byte_exactly(count, matrix_dtype):
    generator = torch.Generator().manual_seed(count)
    table = torch.randint(256, (count, 4), generator=generator, dtype=torch.uint8)
    circuit = PaletteROM(table, query_chunk=127, matrix_dtype=matrix_dtype)
    address = torch.randperm(count, generator=generator).float()
    circuit.validate_input(address)
    assert torch.equal(circuit(address), table[address.long()].float())
    assert torch.equal(circuit(torch.tensor(float(count - 1))), table[-1].float())
    assert circuit(torch.empty(0)).shape == (0, 4)
    json.dumps(circuit.construction_spec())


def test_repeated_pages_eliminate_unused_upper_address_bits():
    page = torch.tensor([[255, 0], [0, 255], [128, 129], [3, 8]], dtype=torch.uint8)
    table = page.repeat(4096, 1)
    circuit = PaletteROM(table, page_size=4)
    assert len(circuit.pages) == 1
    address = torch.tensor([0., 3., 16., 4095., 16383.])
    assert torch.equal(circuit(address), table[address.long()].float())


def test_float16_byte_digits_preserve_page_ids_beyond_half_precision_integers():
    table = torch.randint(256, (70000, 4), generator=torch.Generator().manual_seed(10), dtype=torch.uint8)
    circuit = PaletteROM(table, matrix_dtype=torch.float16)
    assert circuit.pages[0].decoder.count > 4096
    assert circuit.pages[1].digit_count == 2
    address = torch.arange(0, 70000, 113, dtype=torch.float32)
    assert torch.equal(circuit(address), table[address.long()].float())


@pytest.mark.parametrize("kind", ["variable", "fixed", "palette"])
def test_frozen_memory_graph_contains_only_neural_addressing(kind):
    address = torch.tensor([[0., 16., 84.], [83., 5., 43.]])
    if kind == "variable":
        circuit = NeuralRead(85, 100)
        inputs = (address, torch.linspace(-100, 100, 85))
    elif kind == "fixed":
        circuit = FixedROM(torch.arange(255, dtype=torch.float32).reshape(85, 3))
        inputs = (address,)
    else:
        circuit = PaletteROM(torch.randint(256, (85, 4), dtype=torch.uint8), query_chunk=4)
        inputs = (address,)
    frozen = torch.jit.freeze(torch.jit.trace(circuit.eval(), inputs, check_trace=False))
    changed = (address.flip(-1),) + inputs[1:]
    assert torch.equal(frozen(*changed), circuit(*changed))
    graph = str(frozen.inlined_graph)
    for forbidden in ("aten::gather", "aten::index", "aten::embedding", "aten::floor",
                      "aten::round", "aten::remainder", "aten::sort", "aten::where",
                      "prim::If", "prim::Loop", "PythonOp"):
        assert forbidden not in graph
    assert "aten::gt" in graph
    if kind != "fixed":
        assert "aten::relu" in graph
    if kind != "variable":
        assert "aten::linear" in graph


def test_original_artwork_is_exact_at_random_addresses_and_padding():
    path = Path(__file__).resolve().parents[1] / "assets" / "doom1.wad"
    if not path.exists():
        pytest.skip("Original Doom shareware WAD unavailable")
    from doomnet.assets import build_assets

    data, _ = build_assets(path)
    generator = torch.Generator().manual_seed(22)
    for name in ("texture_rgba", "sprite_rgba", "flat_rgba"):
        table = data[name].reshape(-1, 4)
        circuit = PaletteROM(table, matrix_dtype=torch.float16)
        address = torch.cat((torch.tensor([0., float(len(table) - 1)]),
                             torch.randint(len(table), (2048,), generator=generator).float()))
        assert torch.equal(circuit(address), table[address.long()].float())
        assert circuit.construction_spec()["page_coefficients"] < table.numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No Torch GPU executor available")
def test_frozen_palette_byte_layers_relocate_to_gpu_exactly():
    table = torch.randint(256, (5000, 4), generator=torch.Generator().manual_seed(5), dtype=torch.uint8)
    address = torch.randint(5000, (128,), generator=torch.Generator().manual_seed(6)).float()
    circuit = PaletteROM(table, matrix_dtype=torch.float16).eval()
    frozen = torch.jit.freeze(torch.jit.trace(circuit, (address,), check_trace=False))
    artifact = io.BytesIO()
    torch.jit.save(frozen, artifact)
    artifact.seek(0)
    gpu = torch.jit.load(artifact, map_location="cuda")
    assert torch.equal(gpu(address.cuda()).cpu(), table[address.long()].float())


@pytest.mark.parametrize("address", [torch.tensor([-1.]), torch.tensor([85.]), torch.tensor([0.5]),
                                    torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                    torch.tensor([0], dtype=torch.int64)])
def test_invalid_external_addresses_fail(address):
    with pytest.raises(ValueError):
        AddressDecoder(85).validate_input(address)


@pytest.mark.parametrize("values", [torch.tensor([0, 1]), torch.tensor([0., float("nan")]),
                                   torch.tensor([0., float("inf")]), torch.tensor([-2., 0.]),
                                   torch.zeros(3), torch.zeros(3, 2)])
def test_invalid_external_variable_memory_fails(values):
    with pytest.raises(ValueError):
        NeuralRead(2, 1).validate_input(torch.tensor([0., 1.]), values)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 2 ** 24 + 1])
def test_invalid_decoder_counts_fail(count):
    with pytest.raises(ValueError):
        AddressDecoder(count)


@pytest.mark.parametrize("bound", [0, -1, True, float("nan"), float("inf"), 1e40])
def test_invalid_variable_bounds_fail(bound):
    with pytest.raises(ValueError):
        NeuralRead(85, bound)


@pytest.mark.parametrize("table", [torch.zeros(4), torch.zeros(0, 4, dtype=torch.uint8),
                                  torch.zeros(4, 0, dtype=torch.uint8), torch.zeros(2, 3, 4, dtype=torch.uint8)])
def test_invalid_palette_tables_fail(table):
    with pytest.raises(ValueError):
        PaletteROM(table)


@pytest.mark.parametrize("kwargs", [{"page_size": 1}, {"page_size": 3}, {"query_chunk": 0},
                                   {"matrix_dtype": torch.bfloat16}])
def test_invalid_palette_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        PaletteROM(torch.zeros(4, 3, dtype=torch.uint8), **kwargs)
