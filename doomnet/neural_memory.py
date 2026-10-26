"""Exact constructed neural addressing for recurrent memory and fixed artwork.

Addresses are integer-valued float32 activations. A decoder contains fixed
integer-center threshold neurons. Recurrent values are carried through signed,
bounded ReLU gates; artwork is compiled into a recursively deduplicated page
network. Runtime evaluation never indexes a tensor using an activation.

The page matrices contain byte digits, so their one-hot affine evaluations are
exact in float16 as well as float32. Address arithmetic and byte recombination
remain float32. Query chunks are fixed wiring after graph construction.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _count(value, name):
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 0 < value <= 2 ** 24):
        raise ValueError(f"{name} must be a positive integer at most 2**24")
    return value


def _validate_addresses(address, count):
    if (address.dtype != torch.float32 or not bool(torch.isfinite(address).all())
            or not bool(((address >= 0) & (address < count)
                         & (address == address.round())).all())):
        raise ValueError("addresses must be integer-valued float32 within memory bounds")


class AddressDecoder(nn.Module):
    """Integer-center threshold pairs produce one-hot address neurons."""

    def __init__(self, count):
        super().__init__()
        self.count = _count(count, "count")
        self.register_buffer("centers", torch.arange(count, dtype=torch.float32))

    def forward(self, address):
        delta = address[..., None] - self.centers
        return (delta + 0.5 > 0).float() - (delta - 0.5 > 0).float()

    def validate_input(self, address):
        _validate_addresses(address, self.count)


class NeuralRead(nn.Module):
    """Read bounded variable values through fixed decoder and signed ReLU gates.

    ``values`` has shape ``batch + (count,) + value_shape``. ``address`` starts
    with the same batch dimensions, followed by any number of query dimensions.
    The result has shape ``address.shape + value_shape``. A global memory has no
    batch dimensions. The signed gates preserve the selected float32 value exactly.
    """

    def __init__(self, count, bound, value_shape=()):
        super().__init__()
        self.decoder = AddressDecoder(count)
        if (isinstance(bound, bool) or not isinstance(bound, (int, float))
                or not math.isfinite(bound) or not 0 < bound <= torch.finfo(torch.float32).max):
            raise ValueError("bound must be positive, finite, and representable in float32")
        self.value_shape = tuple(_count(size, "value dimension") for size in value_shape)
        self.bound = float(bound)
        self.register_buffer("off_weight", torch.tensor(-self.bound, dtype=torch.float32))

    def forward(self, address, values):
        value_rank = len(self.value_shape)
        batch = values.shape[:values.ndim - value_rank - 1]
        query_rank = address.ndim - len(batch)
        values = values.reshape(batch + (1,) * query_rank + (self.decoder.count,) + self.value_shape)
        off = 1 - self.decoder(address)
        off = off.reshape(off.shape + (1,) * value_rank) * self.off_weight
        carried = F.relu(values + off) - F.relu(-values + off)
        return carried.sum(dim=-value_rank - 1)

    def validate_input(self, address, values):
        self.decoder.validate_input(address)
        tail = (self.decoder.count,) + self.value_shape
        if (values.dtype != torch.float32 or values.ndim < len(tail)
                or values.shape[-len(tail):] != tail):
            raise ValueError("values must be float32 with the constructed memory shape")
        batch = values.shape[:-len(tail)]
        if address.ndim < len(batch) or address.shape[:len(batch)] != batch:
            raise ValueError("address and memory batch dimensions must match")
        if not bool(torch.isfinite(values).all()) or not bool((values.abs() <= self.bound).all()):
            raise ValueError("memory values must be finite and within the constructed signed bound")

    def construction_spec(self):
        return {"type": "NeuralRead", "count": self.decoder.count,
                "value_shape": list(self.value_shape), "signed_bound": self.bound,
                "activations": ["hard_threshold", "relu"],
                "connectivity": "fixed address centers, bounded signed carries, unit sum"}


class FixedROM(nn.Module):
    """Small fixed tables are a threshold decoder followed by one affine layer."""

    def __init__(self, table):
        super().__init__()
        if table.ndim == 0 or table.numel() == 0 or not bool(torch.isfinite(table).all()):
            raise ValueError("fixed memory must be a nonempty finite tensor with an entry axis")
        self.decoder = AddressDecoder(table.shape[0])
        self.value_shape = tuple(table.shape[1:])
        weight = table.detach().float().reshape(table.shape[0], -1).T.contiguous().clone()
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("fixed memory values must be representable in float32")
        self.register_buffer("weight", weight)

    def forward(self, address):
        value = F.linear(self.decoder(address), self.weight)
        return value.reshape(address.shape + self.value_shape)

    def validate_input(self, address):
        self.decoder.validate_input(address)

    def construction_spec(self):
        return {"type": "FixedROM", "count": self.decoder.count,
                "value_shape": list(self.value_shape), "affine_layers": 1,
                "activations": ["hard_threshold"], "weights": list(self.weight.shape)}


class _Page(nn.Module):
    """Evaluate all constant page patterns, then carry the neurally named page."""

    def __init__(self, patterns, matrix_dtype):
        super().__init__()
        self.decoder = AddressDecoder(len(patterns))
        digit_count = max(1, (int(patterns.max()).bit_length() + 7) // 8)
        places = 256 ** np.arange(digit_count)
        digits = (patterns[:, None, :] // places[None, :, None]) % 256
        self.register_buffer("weight", torch.from_numpy(digits.reshape(-1, patterns.shape[1])).to(matrix_dtype))
        self.register_buffer("off_weight", torch.tensor(-256., dtype=matrix_dtype))
        self.register_buffer("place_weight", torch.tensor(places, dtype=torch.float32).reshape(1, -1))
        self.digit_count = digit_count

    def forward(self, offset, page):
        candidates = F.linear(offset.to(self.weight.dtype), self.weight)
        candidates = candidates.reshape(offset.shape[:-1] + (self.decoder.count, self.digit_count))
        if self.decoder.count > 1:
            off = (1 - self.decoder(page)).to(self.weight.dtype) * self.off_weight
            candidates = F.relu(candidates + off[..., None])
        digits = candidates.sum(-2).float()
        return F.linear(digits, self.place_weight).squeeze(-1)


class PaletteROM(nn.Module):
    """Compile a uint8 table into exact recursive neural page/palette layers.

The table is ``[entry, channel]``. Constructor-time deduplication gives a fixed
palette and constant page matrices; the page IDs are themselves read by further
page layers. Binary address extraction uses only affine threshold neurons. A
float16 matrix option accelerates GPU affine layers without rounding byte data.
    """

    def __init__(self, table, page_size=16, query_chunk=512, matrix_dtype=torch.float32):
        super().__init__()
        if table.dtype != torch.uint8 or table.ndim != 2 or table.numel() == 0:
            raise ValueError("palette memory requires a nonempty uint8 [entry, channel] tensor")
        self.count = _count(table.shape[0], "entry count")
        self.channels = table.shape[1]
        self.page_size = _count(page_size, "page_size")
        if page_size < 2 or page_size & (page_size - 1):
            raise ValueError("page_size must be a power of two at least two")
        self.query_chunk = _count(query_chunk, "query_chunk")
        if matrix_dtype not in (torch.float32, torch.float16):
            raise ValueError("page matrices require float32 or float16")
        raw = np.ascontiguousarray(table.detach().cpu().numpy())
        _, positions, codes = np.unique(raw.view(np.dtype((np.void, self.channels))).reshape(-1),
                                        return_index=True, return_inverse=True)
        self.palette = FixedROM(table.detach().cpu()[torch.from_numpy(positions)])
        self.pages = nn.ModuleList()
        while True:
            padded = np.pad(codes, (0, (-len(codes)) % page_size)).reshape(-1, page_size)
            patterns, codes = np.unique(padded, axis=0, return_inverse=True)
            self.pages.append(_Page(patterns, matrix_dtype))
            if len(patterns) == 1:
                break
        self.address_bits = max(1, (self.count - 1).bit_length())
        self.page_bits = (page_size - 1).bit_length()
        # All binary extraction constants remain float32, including on a GPU.
        self.register_buffer("bit_weights", 2. ** torch.arange(self.address_bits - 1, -1, -1))
        self.offset_decoder = AddressDecoder(page_size)
        self.register_buffer("digit_weight", 2. ** torch.arange(self.page_bits - 1, -1, -1))

    def _read(self, address):
        remainder = address
        bits = []
        for weight in self.bit_weights.unbind():
            bit = (remainder - weight + 0.5 > 0).float()
            remainder = remainder - bit * weight
            bits.append(bit)
        binary = torch.stack(bits, -1)
        # Fixed zero padding handles a partial most-significant page; negative
        # padding discards upper bits when all remaining pages are identical.
        binary = F.pad(binary, (len(self.pages) * self.page_bits - self.address_bits, 0))
        page = torch.zeros_like(address)
        for level in range(len(self.pages) - 1, -1, -1):
            start = (len(self.pages) - level - 1) * self.page_bits
            digit = F.linear(binary[..., start:start + self.page_bits], self.digit_weight[None, :]).squeeze(-1)
            page = self.pages[level](self.offset_decoder(digit), page)
        return self.palette(page)

    def forward(self, address):
        values = [self._read(chunk) for chunk in address.reshape(-1).split(self.query_chunk)]
        return torch.cat(values, 0).reshape(address.shape + (self.channels,))

    def validate_input(self, address):
        _validate_addresses(address, self.count)

    def construction_spec(self):
        return {"type": "PaletteROM", "count": self.count, "channels": self.channels,
                "palette_colors": self.palette.decoder.count, "page_size": self.page_size,
                "page_patterns": [page.decoder.count for page in self.pages],
                "page_coefficients": sum(page.weight.numel() for page in self.pages),
                "matrix_dtype": str(self.pages[0].weight.dtype), "query_chunk": self.query_chunk,
                "address_bits": self.address_bits, "activations": ["hard_threshold", "relu"],
                "connectivity": "binary address neurons, fixed page matrices, bounded carries, palette affine"}
