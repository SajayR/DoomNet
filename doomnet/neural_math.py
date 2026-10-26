"""Fixed arithmetic networks: affine maps, ReLU, and H(x) = float(x > 0).

Every numerical forward path is an executed network, not a promise to lower a
native arithmetic operation later. Python math and native tensor calculations
only construct constants or validate external inputs. Tensor reshaping and
constant slices are static wires. All gates carry bounded activations through
ReLU; none multiply two activations or select with an activation as an index.

Error specifications distinguish exact real arithmetic from conservative
floating-point executor roundoff estimates and empirical checks. Select
``dtype=torch.float64`` when constructing interpolation constants for float64.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def hard_step(value):
    """The sole hard activation, H(x)=1 for x>0 and 0 for x<=0."""
    return (value > 0).to(value.dtype)


def bounded_gate(value, gate, bound):
    """Carry value in [-bound,bound] when a zero/one gate is one."""
    off = (1 - gate) * bound
    return F.relu(value - off) - F.relu(-value - off)


def clamp(value, lower, upper):
    """Clamp finite activations, preserving values and endpoints exactly.

    The largest finite execution value bounds the carry. Unlike cancelling a
    shifted ReLU, this retains tiny interior values beside large endpoints.
    """
    below, above = hard_step(lower - value), hard_step(value - upper)
    carried = bounded_gate(value, 1 - below - above, torch.finfo(value.dtype).max)
    return carried + below * lower + above * upper


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _count(value, name, maximum=52):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _finite_scale(value, name):
    if not math.isfinite(value) or value > torch.finfo(torch.float32).max:
        raise ValueError(f"{name} must be representable in float32")
    return value


def _power_bound(value):
    return _finite_scale(2.0 ** math.ceil(math.log2(value)), "normalized bound")


def _execution_errors(real_error, scale, stages, factor):
    """Conservative normal-float gamma_n bounds, separate from measurements.

    Dyadic digit extraction is exact on normal activations.
    The remaining bounded sums obey |fl(sum)-sum| <= gamma_n*sum(abs(terms)).
    Factors include input-sum perturbation and final affine carries. These
    estimates assume full execution-dtype matrix precision and round-to-nearest.
    """
    result = {"execution_error_model": "IEEE round-to-nearest, normal affine terms, full-dtype matrices"}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        unit = torch.finfo(dtype).eps / 2
        gamma = stages * unit / (1 - stages * unit)
        roundoff = factor * gamma * scale
        result[f"{name}_roundoff_bound"] = roundoff
        result[f"{name}_absolute_error_bound"] = real_error + roundoff
    return result


def _validate(value, bound, name="input"):
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64 activations")
    if not bool(torch.isfinite(value).all()) or not bool((value.abs() <= bound).all()):
        raise ValueError(f"{name} must be finite and within [-{bound}, {bound}]")


def _validate_pair(left, right, left_bound, right_bound):
    _validate(left, left_bound, "left input")
    _validate(right, right_bound, "right input")
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError("arithmetic inputs must have the same dtype and device")
    try:
        torch.broadcast_shapes(left.shape, right.shape)
    except RuntimeError as error:
        raise ValueError("arithmetic input shapes must broadcast") from error


class BinaryMultiply(nn.Module):
    """Quantize |left| in binary and carry right through bounded ReLU gates.

    With exact arithmetic, the product's absolute error is at most
    x_scale*y_scale*2**-bits. Unlike polarization, small left values are not
    obtained by cancellation of two squares. Power-of-two normalization is
    exact on ordinary floating-point inputs.
    """

    def __init__(self, x_bound, y_bound, bits=30):
        super().__init__()
        self.x_bound, self.y_bound = _positive(x_bound, "x_bound"), _positive(y_bound, "y_bound")
        self.bits = _count(bits, "bits")
        self.x_scale, self.y_scale = _power_bound(self.x_bound), _power_bound(self.y_bound)
        self.output_scale = _finite_scale(self.x_scale * self.y_scale, "product output bound")
        self.output_activation_bound = self.output_scale
        self.error_bound = self.output_scale * 2.0 ** -self.bits

    def forward(self, left, right):
        left, right = left * (1 / self.x_scale), right * (1 / self.y_scale)
        residual = F.relu(left) + F.relu(-left)
        result = residual * 0
        for stage in range(self.bits + 1):
            place = 2.0 ** -stage
            digit = 1 - hard_step(place - residual)
            residual = residual - digit * place
            result = result + bounded_gate(right, digit, 1) * place
        negative = hard_step(-left)
        result = clamp(result, -1, 1)
        return (result - 2 * bounded_gate(result, negative, 1)) * self.output_scale

    def validate_input(self, left, right):
        _validate_pair(left, right, self.x_bound, self.y_bound)

    def construction_spec(self):
        return {"type": "BinaryMultiply", "input_bounds": [self.x_bound, self.y_bound],
                "bits": self.bits, "absolute_error_bound": self.error_bound,
                "normalization_bounds": [self.x_scale, self.y_scale],
                "output_activation_bound": self.output_activation_bound,
                "output_activation_bounds": [-self.output_activation_bound, self.output_activation_bound],
                "construction": "binary digit extraction, bounded ReLU carries, bounded normalized output",
                "activations": ["hard_threshold", "relu"],
                **_execution_errors(self.error_bound, self.output_scale, self.bits + 3, 2)}


class BinarySquare(nn.Module):
    """A correlated binary product with a certified nonnegative activation."""

    def __init__(self, bound, bits=30):
        super().__init__()
        self.product = BinaryMultiply(bound, bound, bits)
        self.bound = self.product.x_bound
        self.output_activation_bound = self.product.output_activation_bound
        self.error_bound = self.product.error_bound
        self.pointwise_error_coefficient = self.product.x_scale * 2.0 ** -self.product.bits

    def forward(self, value):
        return F.relu(self.product(value, value))

    def validate_input(self, value):
        _validate(value, self.bound)

    def construction_spec(self):
        return dict(self.product.construction_spec(), type="BinarySquare", domain=[-self.bound, self.bound],
                    output_activation_bounds=[0, self.output_activation_bound],
                    pointwise_absolute_error_coefficient=self.pointwise_error_coefficient,
                    pointwise_error="abs(input) times pointwise_absolute_error_coefficient",
                    construction="correlated binary product and nonnegative output activation")


class Divide(nn.Module):
    """Bounded binary long division, truncating the quotient toward zero.

    The denominator must be nonzero and |numerator/denominator| must not exceed
    quotient_bound. Each quotient bit compares residual against denominator
    times a fixed bit place and subtracts a bounded carried denominator. There
    is no reciprocal, variable multiplication, or native division in forward.
    """

    def __init__(self, numerator_bound, denominator_bound, quotient_bound, fractional_bits=24):
        super().__init__()
        self.numerator_bound = _positive(numerator_bound, "numerator_bound")
        self.denominator_bound = _positive(denominator_bound, "denominator_bound")
        self.quotient_bound = _positive(quotient_bound, "quotient_bound")
        self.fractional_bits = _count(fractional_bits, "fractional_bits")
        self.top_bit = math.floor(math.log2(self.quotient_bound))
        if self.top_bit < -self.fractional_bits:
            raise ValueError("quotient_bound must be at least one quotient quantum")
        _finite_scale(self.denominator_bound * 2.0 ** self.top_bit, "division comparison bound")
        self.carry_bound = _power_bound(self.quotient_bound)
        self.output_activation_bound = 2.0 ** (self.top_bit + 1)
        self.error_bound = 2.0 ** -self.fractional_bits

    def forward(self, numerator, denominator):
        residual = F.relu(numerator) + F.relu(-numerator)
        magnitude = F.relu(denominator) + F.relu(-denominator)
        result = residual * 0
        for exponent in range(self.top_bit, -self.fractional_bits - 1, -1):
            place = 2.0 ** exponent
            digit = 1 - hard_step(magnitude * place - residual)
            carried = F.relu(magnitude - (1 - digit) * self.denominator_bound)
            residual = residual - carried * place
            result = result + digit * place
        signs = hard_step(-numerator) + hard_step(-denominator)
        negative = hard_step(signs - 0.5) - hard_step(signs - 1.5)
        return result - 2 * F.relu(result - (1 - negative) * self.carry_bound)

    def validate_input(self, numerator, denominator):
        _validate_pair(numerator, denominator, self.numerator_bound, self.denominator_bound)
        if not bool((denominator != 0).all()):
            raise ValueError("division denominator must be nonzero")
        minimum = torch.finfo(denominator.dtype).tiny * 2.0 ** self.fractional_bits
        if not bool((denominator.abs() >= minimum).all()):
            raise ValueError("division denominator is too small for normal quotient-bit comparisons")
        if not bool(((numerator / denominator).abs() <= self.quotient_bound).all()):
            raise ValueError("division quotient exceeds the constructed quotient bound")

    def construction_spec(self):
        return {"type": "Divide", "input_bounds": [self.numerator_bound, self.denominator_bound],
                "quotient_bound": self.quotient_bound, "denominator": "nonzero",
                "fractional_bits": self.fractional_bits, "absolute_error_bound": self.error_bound,
                "stages": self.top_bit + self.fractional_bits + 1,
                "output_activation_bound": self.output_activation_bound,
                "output_activation_bounds": [-self.output_activation_bound, self.output_activation_bound],
                "minimum_float32_denominator": torch.finfo(torch.float32).tiny * 2.0 ** self.fractional_bits,
                "minimum_float64_denominator": torch.finfo(torch.float64).tiny * 2.0 ** self.fractional_bits,
                "construction": "binary quotient comparisons and bounded denominator carries",
                "activations": ["hard_threshold", "relu"],
                **_execution_errors(self.error_bound, self.quotient_bound, self.top_bit + self.fractional_bits + 5, 6)}


class PiecewiseLinear(nn.Module):
    """Local ReLU hats with fixed knots and scalar or vector sample values.

    Local hats avoid subtracting many large global hinges. Only the two hats
    beside a valid input contribute; their weighted sum interpolates samples.
    """

    def __init__(self, knots, values, *, dtype=torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("interpolation dtype must be float32 or float64")
        points, samples = torch.as_tensor(knots, dtype=torch.float64), torch.as_tensor(values, dtype=torch.float64)
        if (points.ndim != 1 or len(points) < 2 or not bool(torch.isfinite(points).all())
                or not bool((points[1:] > points[:-1]).all())):
            raise ValueError("knots must contain at least two finite strictly increasing values")
        if (samples.ndim not in (1, 2) or samples.shape[0] != len(points)
                or not bool(torch.isfinite(samples).all())):
            raise ValueError("values must contain one finite scalar or vector per knot")
        self.scalar_output = samples.ndim == 1
        self.register_buffer("knots", points.to(dtype))
        self.lower, self.upper = float(self.knots[0]), float(self.knots[-1])
        intervals = self.knots[1:] - self.knots[:-1]
        self.register_buffer("inverse_left", torch.cat((1 / intervals[:1], 1 / intervals)).to(dtype))
        self.register_buffer("inverse_right", torch.cat((1 / intervals, 1 / intervals[-1:])).to(dtype))
        self.register_buffer("output_weight", (samples[None] if self.scalar_output else samples.T).to(dtype))
        self.register_buffer("output_lower", self.output_weight.amin(-1))
        self.register_buffer("output_upper", self.output_weight.amax(-1))
        if (not bool((self.knots[1:] > self.knots[:-1]).all())
                or not all(bool(torch.isfinite(buffer).all()) for buffer in self.buffers())):
            raise ValueError("knots and affine weights must remain finite and distinct in the execution dtype")
        if float(self.output_weight.abs().max()) > torch.finfo(dtype).max / 4:
            raise ValueError("sample magnitudes need headroom for finite interpolation sums")

    def forward(self, value):
        coordinate = value[..., None]
        hats = F.relu(1 - F.relu((self.knots - coordinate) * self.inverse_left)
                      - F.relu((coordinate - self.knots) * self.inverse_right))
        result = clamp(F.linear(hats, self.output_weight), self.output_lower, self.output_upper)
        return result.squeeze(-1) if self.scalar_output else result

    def validate_input(self, value):
        _validate(value, max(abs(self.lower), abs(self.upper)))
        if not bool(((value >= self.lower) & (value <= self.upper)).all()):
            raise ValueError("interpolation input is outside the knot interval")

    def construction_spec(self):
        return {"type": "PiecewiseLinear", "domain": [self.lower, self.upper],
                "knots": len(self.knots), "outputs": self.output_weight.shape[0],
                "output_activation_bounds": [self.output_lower.tolist(), self.output_upper.tolist()],
                "construction": "local ReLU hats, fixed affine samples, bounded output activation",
                "activations": ["hard_threshold", "relu"]}


class Sqrt(nn.Module):
    """Square root through threshold octaves and a shared local-hat network.

    Every positive octave normalizes into [1,2]; the lowest also represents
    zero with the secant on [0,1]. Its worst absolute error is sqrt(power)/4.
    """

    def __init__(self, maximum, minimum=2.0 ** -28, segments=512, *, dtype=torch.float32):
        super().__init__()
        self.minimum, self.maximum = _positive(minimum, "minimum"), _positive(maximum, "maximum")
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        self.segments = _count(segments, "segments", maximum=4096)
        first, last = math.floor(math.log2(self.minimum)), math.floor(math.log2(self.maximum))
        powers = [2.0 ** exponent for exponent in range(first, last + 1)]
        scales = [math.sqrt(power) for power in powers]
        for scale in (self.maximum, 1 / powers[0], *scales):
            _finite_scale(scale, "dyadic affine scale")
        self.register_buffer("thresholds", torch.tensor(powers[1:], dtype=dtype))
        self.register_buffer("normalize_weight", torch.tensor([1 / power for power in powers], dtype=dtype))
        self.register_buffer("rescale_weight", torch.tensor(scales, dtype=dtype))
        knots = [0.] + [1 + index / self.segments for index in range(self.segments + 1)]
        self.kernel = PiecewiseLinear(knots, [math.sqrt(value) for value in knots], dtype=dtype)
        self.output_activation_bound = _power_bound(float(self.rescale_weight.max()) * float(self.kernel.output_upper.max()))
        self.zero_interval_error_bound = math.sqrt(powers[0]) / 4
        self.relative_error_bound = 1 / (32 * self.segments ** 2)

    def forward(self, value):
        entered = 1 - hard_step(self.thresholds - value[..., None])
        edges = torch.cat((value[..., None] * 0 + 1, entered, value[..., None] * 0), -1)
        gates = edges[..., :-1] - edges[..., 1:]
        carried = F.relu(value[..., None] - (1 - gates) * self.maximum)
        normalized = F.linear(carried, self.normalize_weight)
        result = self.kernel(normalized)
        return F.linear(F.relu(result[..., None] - (1 - gates) * 2), self.rescale_weight)

    def validate_input(self, value):
        _validate(value, self.maximum)
        if not bool((value >= 0).all()):
            raise ValueError("square-root inputs must be nonnegative")

    def construction_spec(self):
        value_bound = math.sqrt(self.maximum)
        real_error = max(self.relative_error_bound * value_bound, self.zero_interval_error_bound)
        return {"type": "Sqrt", "domain": [0, self.maximum],
                "octaves": len(self.normalize_weight), "normalized_segments": self.segments,
                "normalized_domain": [1, 2], "zero_interval_absolute_error_bound": self.zero_interval_error_bound,
                "relative_error_bound": self.relative_error_bound, "absolute_error_bound": real_error,
                "relative_error_bound_above_minimum": self.relative_error_bound,
                "output_activation_bound": self.output_activation_bound,
                "output_activation_bounds": [0, self.output_activation_bound],
                "construction": "threshold octaves, bounded ReLU carries, shared local-hat interpolant",
                "activations": ["hard_threshold", "relu"],
                **_execution_errors(real_error, value_bound, 12, 16)}


class Trunc(nn.Module):
    """Exact signed integer truncation on a bounded float32 domain."""

    def __init__(self, bound):
        super().__init__()
        self.bound = _positive(bound, "bound")
        if self.bound > 2 ** 24:
            raise ValueError("integer quantization bound cannot exceed 2**24")
        self.top_bit = math.floor(math.log2(self.bound))
        self.carry_bound = _power_bound(self.bound)

    def _digits(self, value):
        residual = F.relu(value) + F.relu(-value)
        result = residual * 0
        for exponent in range(self.top_bit, -1, -1):
            place = 2.0 ** exponent
            digit = 1 - hard_step(place - residual)
            carried = digit * place
            residual, result = residual - carried, result + carried
        return result, residual

    def forward(self, value):
        result, _ = self._digits(value)
        return result - 2 * F.relu(result - (1 - hard_step(-value)) * self.carry_bound)

    def validate_input(self, value):
        _validate(value, self.bound)

    def construction_spec(self):
        return {"type": type(self).__name__, "domain": [-self.bound, self.bound], "exact": True,
                "stages": max(0, self.top_bit + 1), "construction": "signed binary integer digit extraction",
                "activations": ["hard_threshold", "relu"]}


class Floor(Trunc):
    """Exact floor, retaining negative fractions even beside large bounds."""

    def forward(self, value):
        result, residual = self._digits(value)
        negative = hard_step(-value)
        fraction = hard_step(negative + hard_step(residual) - 1.5)
        return result - 2 * F.relu(result - (1 - negative) * self.carry_bound) - fraction


class Frac(nn.Module):
    """Signed fractional part, matching x-trunc(x)."""

    def __init__(self, bound):
        super().__init__()
        self.trunc = Trunc(bound)

    def forward(self, value):
        return value - self.trunc(value)

    def validate_input(self, value):
        self.trunc.validate_input(value)

    def construction_spec(self):
        return dict(self.trunc.construction_spec(), type="Frac", construction="input minus binary truncation")


class Remainder(nn.Module):
    """Remainder with a fixed positive divisor and bounded signed input.

    Integer divisors and integer inputs are exact within the float32 integer
    range. Continuous remainders retain ordinary floating-point roundoff.
    Texture addressing should apply Floor before this circuit.
    """

    def __init__(self, modulus, bound):
        super().__init__()
        self.modulus, self.bound = _positive(modulus, "modulus"), _positive(bound, "bound")
        _finite_scale(max(self.modulus, self.bound), "remainder bound")
        self.top_bit = math.floor(math.log2(self.bound / self.modulus))
        self.integer_exact = self.modulus.is_integer() and self.bound <= 2 ** 24

    def forward(self, value):
        residual = F.relu(value) + F.relu(-value)
        for exponent in range(self.top_bit, -1, -1):
            trial = self.modulus * 2.0 ** exponent
            digit = 1 - hard_step(trial - residual)
            residual = residual - digit * trial
        negative_fraction = hard_step(hard_step(-value) + hard_step(residual) - 1.5)
        return residual + bounded_gate(self.modulus - 2 * residual, negative_fraction, self.modulus)

    def validate_input(self, value):
        _validate(value, self.bound)

    def construction_spec(self):
        return {"type": "Remainder", "domain": [-self.bound, self.bound], "modulus": self.modulus,
                "stages": max(0, self.top_bit + 1), "integer_inputs_exact": self.integer_exact,
                "construction": "binary constant-divisor residual extraction and signed carry",
                "activations": ["hard_threshold", "relu"]}


class Round(nn.Module):
    """Nearest integer with even ties, composed from exact floor and parity."""

    def __init__(self, bound):
        super().__init__()
        self.floor = Floor(bound)
        self.parity = Remainder(2, math.ceil(bound))

    def forward(self, value):
        base = self.floor(value)
        fraction = value - base
        above = hard_step(fraction - 0.5)
        distance = F.relu(fraction - 0.5) + F.relu(0.5 - fraction)
        tie = 1 - hard_step(distance)
        even_tie = hard_step(tie + self.parity(base) - 1.5)
        return base + above + even_tie

    def validate_input(self, value):
        self.floor.validate_input(value)

    def construction_spec(self):
        return dict(self.floor.construction_spec(), type="Round", ties="even", construction="floor, fractional threshold, integer parity")


class SinCos(nn.Module):
    """Bounded periodic sine/cosine through a fixed local-hat interpolant."""

    def __init__(self, bound=2 * math.pi, segments=256, *, dtype=torch.float32):
        super().__init__()
        self.bound = _positive(bound, "bound")
        self.segments = _count(segments, "segments", maximum=4096)
        self.wrap = Remainder(2 * math.pi, self.bound)
        knots = [2 * math.pi * index / self.segments for index in range(self.segments + 1)]
        self.kernel = PiecewiseLinear(knots, [[math.sin(value), math.cos(value)] for value in knots], dtype=dtype)
        self.error_bound = (2 * math.pi / self.segments) ** 2 / 8

    def forward(self, value):
        return self.kernel(self.wrap(value))

    def validate_input(self, value):
        _validate(value, self.bound)

    def construction_spec(self):
        return {"type": type(self).__name__, "domain": [-self.bound, self.bound], "segments": self.segments,
                "absolute_error_bound": self.error_bound, "construction": "bounded period residual and local ReLU hats",
                "output_activation_bound": 1, "output_activation_bounds": [-1, 1],
                "activations": ["hard_threshold", "relu"],
                **_execution_errors(self.error_bound, self.bound + 1, max(0, self.wrap.top_bit + 1) + 12, 16)}


class Sin(SinCos):
    def forward(self, value):
        return super().forward(value)[..., 0]


class Cos(SinCos):
    def forward(self, value):
        return super().forward(value)[..., 1]


class Atan2(nn.Module):
    """Quadrant-aware atan2(y,x); the origin has the defined output zero."""

    def __init__(self, bound, segments=64, fractional_bits=20, *, dtype=torch.float32):
        super().__init__()
        self.bound = _positive(bound, "bound")
        self.segments = _count(segments, "segments", maximum=4096)
        self.divide = Divide(self.bound, max(self.bound, 1), 1, fractional_bits)
        knots = [index / self.segments for index in range(self.segments + 1)]
        self.kernel = PiecewiseLinear(knots, [math.atan(value) for value in knots], dtype=dtype)
        self.error_bound = 1 / (8 * self.segments ** 2) + self.divide.error_bound

    def forward(self, y, x):
        ax, ay = F.relu(x) + F.relu(-x), F.relu(y) + F.relu(-y)
        y_larger = hard_step(ay - ax)
        small = F.relu(ax - (1 - y_larger) * self.bound) + F.relu(ay - y_larger * self.bound)
        large = F.relu(ay - (1 - y_larger) * self.bound) + F.relu(ax - y_larger * self.bound)
        # atan2(0,0)=0 is defined behavior; its ratio is explicitly 0/1.
        denominator = large + (1 - hard_step(large))
        angle = self.kernel(self.divide(small, denominator))
        angle = angle + bounded_gate(math.pi / 2 - 2 * angle, y_larger, math.pi / 2)
        angle = angle + bounded_gate(math.pi - 2 * angle, hard_step(-x), math.pi)
        return clamp(angle - 2 * F.relu(angle - (1 - hard_step(-y)) * math.pi), -math.pi, math.pi)

    def validate_input(self, y, x):
        _validate_pair(y, x, self.bound, self.bound)
        magnitude = torch.maximum(y.abs(), x.abs())
        lower = torch.finfo(y.dtype).tiny * 2.0 ** self.divide.fractional_bits
        if not bool(((magnitude == 0) | (magnitude >= lower)).all()):
            raise ValueError("atan2 magnitude is too small for the constructed quotient bits")

    def construction_spec(self):
        return {"type": "Atan2", "input_bounds": [self.bound, self.bound], "origin": 0,
                "segments": self.segments, "absolute_error_bound": self.error_bound,
                "output_activation_bound": 4,
                "float32_output_activation_bounds": [-float(torch.tensor(math.pi)), float(torch.tensor(math.pi))],
                "float64_output_activation_bounds": [-math.pi, math.pi],
                "fractional_bits": self.divide.fractional_bits,
                "construction": "bounded octant quotient, local ReLU hats, quadrant carries",
                "activations": ["hard_threshold", "relu"],
                **_execution_errors(self.error_bound, math.pi, self.divide.fractional_bits + 16, 16)}
