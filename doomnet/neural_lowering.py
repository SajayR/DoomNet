"""Compile a frozen, straight-line tensor program into constructed neurons.

This is a construction-time compiler. Its output executes fixed affine maps,
ReLU, strict zero-threshold neurons, and static wires. Source integer/Boolean
activations become float32 activations; source dtype is retained only for shape
inference and integer-cast semantics. An unsupported source node is an error.
The resulting module must still pass ``audit_neural_graph`` after freezing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
import operator
from typing import Mapping

import torch
from torch import fx, nn
from torch.nn import functional as F


_DTYPES = {0: torch.uint8, 1: torch.int8, 2: torch.int16, 3: torch.int32,
           4: torch.int64, 5: torch.float16, 6: torch.float32,
           7: torch.float64, 11: torch.bool, 15: torch.bfloat16}
_WIRES = {"slice", "select", "reshape", "view", "flatten", "unsqueeze",
          "squeeze", "transpose", "permute", "t", "expand", "repeat",
          "cat", "stack", "pad", "constant_pad_nd", "clone", "contiguous",
          "detach", "alias", "split", "split_with_sizes", "chunk", "unbind"}
_SOURCE = _WIRES | {"add", "sub", "rsub", "mul", "div", "neg", "linear",
    "mm", "bmm", "matmul", "addmm", "sum", "mean", "relu", "gt", "ge",
    "lt", "le", "eq", "ne", "__and__", "__iand__", "__or__", "bitwise_not",
    "any", "all", "abs", "minimum", "maximum", "clamp", "clamp_min",
    "clamp_max", "where", "to", "size", "Int", "expand_as", "zeros_like",
    "ones_like", "full_like", "index", "index_select", "gather", "index_copy",
    "floor", "round", "frac", "remainder", "sin", "cos", "atan2", "sqrt",
    "square"}


@dataclass
class _Fact:
    sample: object
    ref: fx.Node | None = None
    low: torch.Tensor | None = None
    high: torch.Tensor | None = None
    integer: torch.Tensor | bool = False
    items: tuple[_Fact, ...] | None = None
    codec: bool = False
    predicate: tuple | None = None
    complement_of: _Fact | None = None
    partition: tuple | None = None
    offset_partition: tuple | None = None
    candidate_partition: tuple | None = None
    partition_piece: tuple | None = None
    minimum_abs: float = 0.
    condition: tuple | None = None
    condition_columns: tuple | None = None
    threshold_columns: tuple | None = None
    relu_clauses: tuple | None = None
    exclusive: bool = False
    semantic: object = field(default_factory=object)

    @property
    def fixed(self):
        return self.ref is None and (self.items is None or all(x.fixed for x in self.items))

    @property
    def shape(self):
        return tuple(self.sample.shape)

    def interval(self):
        if self.fixed:
            value = torch.as_tensor(self.sample, dtype=torch.float64)
            return value, value
        assert self.low is not None and self.high is not None, "activations need finite domains"
        return self.low, self.high


def _compact(value):
    value = torch.as_tensor(value)
    if value.numel() > 1 and bool((value == value.reshape(-1)[0]).all()):
        return value.reshape(-1)[0]
    return value


def _magnitude(fact):
    low, high = fact.interval()
    return max(float(low.abs().max()), float(high.abs().max()))


def _integer(fact):
    if fact.fixed:
        value = torch.as_tensor(fact.sample)
        return not value.is_floating_point() or bool((value == value.trunc()).all())
    return bool(torch.as_tensor(fact.integer).all())


def _integer_mask(fact):
    if fact.fixed:
        value = torch.as_tensor(fact.sample)
        return value == value.trunc() if value.is_floating_point() else torch.tensor(True)
    return torch.as_tensor(fact.integer)


def _constraints(condition, truth=True):
    """Closed domains implied by an exact Boolean expression being true/false."""
    if condition is None:
        return {}
    kind, *args = condition
    if kind == "not":
        return _constraints(args[0], not truth)
    if kind in {"and", "or"}:
        conjunction = (kind == "and") == truth
        children = [_constraints(child, truth) for child in args]
        if conjunction:
            result = {}
            for child in children:
                for key, (fact, lo, hi) in child.items():
                    previous = result.get(key, (fact, -math.inf, math.inf))
                    result[key] = (fact, max(lo, previous[1]), min(hi, previous[2]))
            return result
        keys = set.intersection(*(set(child) for child in children)) if children else set()
        return {key: (children[0][key][0], min(child[key][1] for child in children),
                      max(child[key][2] for child in children)) for key in keys}
    if kind != "cmp":
        return {}
    name, a, b = args
    if not truth:
        name = {"gt": "le", "ge": "lt", "lt": "ge", "le": "gt", "eq": "ne", "ne": "eq"}[name]
    al, ah = a.interval(); bl, bh = b.interval()
    result = {}
    if name in {"gt", "ge", "eq"}:
        result[a.semantic] = (a, float(bl.min()), math.inf)
        result[b.semantic] = (b, -math.inf, float(ah.max()))
    if name in {"lt", "le", "eq"}:
        previous = result.get(a.semantic, (a, -math.inf, math.inf))
        result[a.semantic] = (a, previous[1], float(bh.max()))
        previous = result.get(b.semantic, (b, -math.inf, math.inf))
        result[b.semantic] = (b, float(al.min()), previous[2])
    return result


def _refined(fact, condition, truth):
    low, high = fact.interval()
    constraint = _constraints(condition, truth).get(fact.semantic)
    if constraint is None:
        return low, high
    return torch.clamp_min(low, constraint[1]), torch.clamp_max(high, constraint[2])


def _invoke(schema, facts, *, metadata=False):
    def meta(value):
        if isinstance(value, torch.Tensor):
            return torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="meta")
        if isinstance(value, (tuple, list)):
            return type(value)(meta(item) for item in value)
        return value

    positional, keywords = [], {}
    for argument, fact in zip(schema.arguments, facts, strict=True):
        value = fact.sample
        if metadata:
            value = meta(value)
        if argument.kwarg_only:
            keywords[argument.name] = value
        else:
            positional.append(value)
    name = schema.name.removeprefix("aten::")
    return getattr(getattr(torch.ops.aten, name), schema.overload_name or "default")(*positional, **keywords)


class _Compiler:
    def __init__(self, source, method, input_bounds, source_modules):
        self.graph = fx.Graph()
        self.root = nn.Module()
        self.environment = {}
        self.constants = {}
        self.roms = {}
        self.circuits = []
        self.operators = Counter()
        self.selectors = self.selector_contracts(source_modules)
        self.method = method
        self.source = getattr(source, method).inlined_graph.copy()
        torch._C._jit_pass_inline(self.source)
        torch._C._jit_pass_remove_inplace_ops(self.source)
        torch._C._jit_pass_cse(self.source)
        inputs = [x for x in self.source.inputs() if isinstance(x.type(), torch._C.TensorType)]
        if isinstance(input_bounds, Mapping):
            names = {x.debugName().split(".")[0] for x in inputs}
            if names != set(input_bounds):
                raise ValueError(f"input bounds must describe exactly {sorted(names)}")
            bounds = [input_bounds[x.debugName().split(".")[0]] for x in inputs]
        else:
            bounds = list(input_bounds)
            if len(bounds) != len(inputs):
                raise ValueError("input bounds must describe every tensor input")
        self.input_contracts = []
        for value, (low, high) in zip(inputs, bounds, strict=True):
            low, high = torch.broadcast_tensors(torch.as_tensor(low, dtype=torch.float64),
                                                torch.as_tensor(high, dtype=torch.float64))
            if not bool((torch.isfinite(low) & torch.isfinite(high) & (low <= high)).all()):
                raise ValueError("input bounds must be finite, ordered tensors")
            name = value.debugName().split(".")[0]
            known_shape = value.type().sizes()
            if known_shape is not None and tuple(known_shape) != tuple(low.shape):
                raise ValueError(f"input {name} bounds have shape {tuple(low.shape)}, expected {tuple(known_shape)}")
            self.environment[value] = _Fact(torch.empty(low.shape, device="meta"),
                self.graph.placeholder(name), _compact(low), _compact(high))
            self.input_contracts.append({"name": name, "shape": list(low.shape),
                                         "minimum": float(low.min()), "maximum": float(high.max())})

    def selector_contracts(self, source_modules):
        from .circuits import MinimumCircuit, NearestCircuit, _SelectionCircuit
        contracts = {}
        if source_modules is None:
            return contracts
        for path, module in source_modules.named_modules():
            if not isinstance(module, _SelectionCircuit):
                continue
            if type(module) not in {MinimumCircuit, NearestCircuit}:
                raise ValueError(f"selector contract {path} requires an exact supported class")
            if module.forward.__func__ is not type(module).forward:
                raise ValueError(f"selector contract {path} requires the original forward method")
            rebuilt = (MinimumCircuit(module.input_count, module.bound) if type(module) is MinimumCircuit
                       else NearestCircuit(module.input_count, module.output_count, module.bound))
            spec = rebuilt.construction_spec()
            topology = lambda obj: [(stage.input_count, stage.output_count, stage.comparator_count,
                                      stage.stable_compare) for stage in obj.stages]
            actual, expected = dict(module.named_buffers()), dict(rebuilt.named_buffers())
            if module.construction_spec() != spec or topology(module) != topology(rebuilt) or actual.keys() != expected.keys() or any(actual[name].dtype != value.dtype or not torch.equal(actual[name].cpu(), value) for name, value in expected.items()):
                raise ValueError(f"selector contract {path} has modified construction weights or topology")
            digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
            for name, value in expected.items():
                digest.update(name.encode())
                digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            lines, first = inspect.getsourcelines(_SelectionCircuit.forward)
            contracts[path] = {"module_path": path, "construction": spec,
                "constructor_sha256": digest.hexdigest(), "input_checks": 0, "index_checks": 0,
                "source_file": inspect.getsourcefile(_SelectionCircuit.forward),
                "return_lines": set(range(first, first + len(lines))),
                "theorem": "stable increasing score/index selection places fixed bound-score padding after every real candidate; the first k<=input_count indices are real"}
        return contracts

    def argument(self, fact, *, floating=False):
        if fact.items is not None:
            constructor = list if isinstance(fact.sample, list) else tuple
            return constructor(self.argument(x, floating=floating) for x in fact.items)
        if not fact.fixed:
            return fact.ref
        if isinstance(fact.sample, (tuple, list)):
            return type(fact.sample)(self.argument(_Fact(x), floating=floating) for x in fact.sample)
        if not isinstance(fact.sample, torch.Tensor):
            return fact.sample
        value = fact.sample.detach()
        if floating and not value.is_floating_point():
            value = value.float()
        key = (id(fact.sample), value.dtype)
        if key not in self.constants:
            name = f"constant_{len(self.constants)}"
            self.root.register_buffer(name, value.contiguous())
            self.constants[key] = (self.graph.get_attr(name), fact.sample)
        return self.constants[key][0]

    def constant(self, value):
        return self.argument(_Fact(torch.as_tensor(value, dtype=torch.float32)))

    def emit(self, operation, *arguments, **keywords):
        return self.graph.call_function(operation, arguments, keywords)

    def step(self, value):
        return self.emit(torch.ops.aten.to.dtype, self.emit(torch.gt, value, 0), 6, False, False, None)

    def gate(self, value, gate, fact):
        bound = _magnitude(fact)
        rounded = torch.tensor(bound, dtype=torch.float32)
        if float(rounded) < bound:
            rounded = torch.nextafter(rounded, torch.tensor(math.inf))
        bound = float(rounded)
        if not math.isfinite(bound):
            raise ValueError("bounded gate needs a finite float32 carry weight")
        if not bound:
            return self.emit(torch.mul, value, 0.)
        off = self.emit(torch.mul, self.emit(torch.sub, gate, 1.), bound)
        positive = self.emit(torch.relu, self.emit(torch.add, value, off))
        if float(fact.interval()[0].min()) >= 0:
            return positive
        negative_value = self.emit(torch.neg, value) if isinstance(value, fx.Node) else -value
        negative = self.emit(torch.relu, self.emit(torch.add, negative_value, off))
        return self.emit(torch.sub, positive, negative)

    def choose(self, gate, left, right):
        return self.emit(torch.add, self.gate(self.argument(left, floating=True), gate, left),
                         self.gate(self.argument(right, floating=True), self.emit(torch.sub, 1., gate), right))

    def module(self, module, arguments):
        name = f"circuit_{len(self.circuits)}"
        self.root.add_module(name, module.eval())
        self.circuits.append({"module": name, "construction": module.construction_spec()})
        return self.graph.call_module(name, tuple(arguments))

    def fact(self, sample, ref, low, high, integer=False, **extra):
        low, high = (_compact(torch.as_tensor(x, dtype=torch.float64)) for x in (low, high))
        if not bool(torch.isfinite(low).all()) or not bool(torch.isfinite(high).all()):
            raise ValueError("inferred activation domain is not finite")
        return _Fact(sample, ref, low, high, _compact(integer), **extra)

    def wire(self, schema, facts, sample):
        name = schema.name.removeprefix("aten::")
        if name in {"cat", "stack"}:
            if not facts[1].fixed:
                raise ValueError("wire axes must be construction constants")
        elif any(not x.fixed for x in facts[1:]):
            raise ValueError("wire axes, shapes, and indices must be construction constants")
        partition, piece = None, None
        condition, columns = None, None
        signature = (name,) + tuple(id(x.sample) if isinstance(x.sample, torch.Tensor) else tuple(x.sample) if isinstance(x.sample, list) else x.sample for x in facts[1:])
        def projected(condition):
            if condition is None:
                return None
            kind, *arguments = condition
            if kind in {"and", "or", "not"}:
                return (kind, *(projected(x) for x in arguments))
            if kind != "cmp":
                return condition
            comparison, a, b = arguments
            def target(value):
                if value.fixed or value.shape != facts[0].shape:
                    return value
                domain = []
                for bound in value.interval():
                    domain.append(bound if bound.ndim == 0 else _invoke(schema, [_Fact(torch.broadcast_to(bound, value.shape))] + facts[1:]))
                return self.fact(sample, value.ref, *domain, value.integer,
                                 semantic=(value.semantic, signature))
            return (kind, comparison, target(a), target(b))
        if name in {"cat", "stack"}:
            tensors = facts[0].items
            assert tensors is not None, "tensor wires require explicit lists"
            dimension = facts[1].sample
            op = torch.cat if name == "cat" else torch.stack
            bounds = [op([torch.broadcast_to(x.interval()[side], x.shape) for x in tensors], dimension)
                      for side in (0, 1)]
            integer = op([torch.broadcast_to(_integer_mask(x), x.shape) for x in tensors], dimension)
            if name == "stack" and dimension % len(sample.shape) == len(sample.shape) - 1:
                columns = tuple(x.condition for x in tensors)
            if name == "stack" and len(tensors) == 2 and all(_integer(x) and float(x.interval()[0].min()) >= 0 and float(x.interval()[1].max()) <= 1 for x in tensors) and (tensors[0].complement_of is tensors[1] or tensors[1].complement_of is tensors[0]):
                axis = dimension % len(sample.shape)
                partition = (axis, object(), None)
        else:
            bounds = []
            for side in (0, 1):
                bound = facts[0].interval()[side]
                if bound.ndim == 0 and name not in {"pad", "constant_pad_nd"}:
                    bounds.append(bound)
                else:
                    values = [_Fact(torch.broadcast_to(bound, facts[0].shape))] + facts[1:]
                    bounds.append(_invoke(schema, values))
            integer = _integer_mask(facts[0])
            if integer.ndim:
                integer = _invoke(schema, [_Fact(torch.broadcast_to(integer, facts[0].shape))] + facts[1:])
            original = facts[0]
            if original.partition is not None:
                axis, group, carried_bounds = original.partition
                dimension = facts[1].sample % (len(original.shape) + int(name == "unsqueeze")) if name in {"select", "slice", "squeeze", "unsqueeze"} else None
                if name == "select" and dimension == axis and carried_bounds is not None:
                    index = facts[2].sample
                    domain = [_invoke(schema, [_Fact(torch.broadcast_to(x, original.shape))] + facts[1:]) for x in carried_bounds]
                    piece = (group, index, tuple(domain))
                elif name == "slice" and dimension == axis:
                    if facts[2].sample == 0 and facts[3].sample >= 2 and facts[4].sample == 1:
                        partition = original.partition
                elif name in {"slice", "select", "unsqueeze", "squeeze", "clone", "contiguous", "detach", "alias"}:
                    new_axis = axis + int(name == "unsqueeze" and dimension <= axis) - int(name in {"select", "squeeze"} and dimension < axis)
                    domain = None if carried_bounds is None else tuple(_invoke(schema, [_Fact(torch.broadcast_to(x, original.shape))] + facts[1:]) for x in carried_bounds)
                    partition = (new_axis, (group, signature), domain)
            if original.partition_piece is not None and name in {"slice", "select", "unsqueeze", "squeeze", "clone", "contiguous", "detach", "alias"}:
                group, index, domain = original.partition_piece
                domain = tuple(_invoke(schema, [_Fact(torch.broadcast_to(x, original.shape))] + facts[1:]) for x in domain)
                piece = ((group, signature), index, domain)
            if original.condition_columns is not None:
                old_axis = len(original.shape) - 1
                if name in {"squeeze", "select"} and facts[1].sample % len(original.shape) == old_axis:
                    index = facts[2].sample if name == "select" else 0
                    condition = original.condition_columns[index]
                elif name == "slice" and facts[1].sample % len(original.shape) == old_axis:
                    columns = original.condition_columns[facts[2].sample:facts[3].sample:facts[4].sample]
                elif name in {"slice", "unsqueeze", "squeeze", "select", "clone", "contiguous", "detach", "alias"}:
                    columns = original.condition_columns
            if original.condition is not None and condition is None and columns is None:
                condition = projected(original.condition)
        ref = self.call(schema, facts, floating=True)
        if isinstance(sample, (list, tuple)):
            items = []
            for index, item in enumerate(sample):
                lo = bounds[0][index] if isinstance(bounds[0], (list, tuple)) else bounds[0]
                hi = bounds[1][index] if isinstance(bounds[1], (list, tuple)) else bounds[1]
                items.append(self.fact(item, self.emit(operator.getitem, ref, index), lo, hi, integer))
            return _Fact(sample, ref, items=tuple(items))
        return self.fact(sample, ref, *bounds, integer, partition=partition, partition_piece=piece,
                         condition=condition, condition_columns=columns, minimum_abs=facts[0].minimum_abs,
                         semantic=(facts[0].semantic, signature) if name not in {"cat", "stack"} else object())

    def call(self, schema, facts, *, floating=False):
        positional, keywords = [], {}
        for argument, fact in zip(schema.arguments, facts, strict=True):
            value = self.argument(fact, floating=floating and argument.name in {"self", "other", "mat1", "mat2", "input", "weight", "bias", "tensors"})
            if argument.name == "dtype" and floating and value is not None:
                value = 6
            if argument.kwarg_only:
                keywords[argument.name] = value
            else:
                positional.append(value)
        name = schema.name.removeprefix("aten::")
        operation = getattr(getattr(torch.ops.aten, name), schema.overload_name or "default")
        return self.emit(operation, *positional, **keywords)

    def comparison(self, name, a, b):
        left, right = self.argument(a, floating=True), self.argument(b, floating=True)
        difference = self.emit(torch.sub, left, right)
        if name in {"gt", "le"}:
            result = self.step(difference)
        else:
            result = self.step(self.emit(torch.neg, difference))
        if name in {"le", "ge"}:
            result = self.emit(torch.sub, 1., result)
        if name in {"eq", "ne"}:
            result = self.emit(torch.add, result, self.step(difference))
            if name == "eq":
                result = self.emit(torch.sub, 1., result)
        return result

    def address(self, fact, count, *, negative=False):
        low, high = fact.interval()
        if not _integer(fact):
            raise ValueError("memory addresses must have a proven integer-valued domain")
        if float(low.min()) < (-count if negative else 0) or float(high.max()) >= count:
            raise ValueError(f"memory address domain [{float(low.min())}, {float(high.max())}] exceeds {count} entries")
        ref = self.argument(fact, floating=True)
        if negative and float(low.min()) < 0:
            if float(high.max()) < 0:
                ref = self.emit(torch.add, ref, count)
            else:
                ref = self.emit(torch.add, ref, self.emit(torch.mul, self.step(self.emit(torch.neg, ref)), count))
        return ref

    def read(self, values, address, *, negative=False):
        from .neural_memory import FixedROM, NeuralRead, PaletteROM
        count, value_shape = values.shape[0], values.shape[1:]
        ref = self.address(address, count, negative=negative)
        if values.fixed:
            key = (values.sample.data_ptr(), values.shape, values.sample.stride(), values.sample.dtype)
            if key not in self.roms:
                table = values.sample.detach().cpu()
                if table.dtype == torch.uint8:
                    channels = math.prod(value_shape)
                    query_count = max(1, math.prod(address.shape))
                    page_size = 512 if channels >= 3 else 16 if count <= 2 ** 21 else 64
                    query_chunk = min(query_count, 16384) if channels < 3 and count > 2 ** 21 else query_count
                    rom = PaletteROM(table.reshape(count, -1), page_size=page_size,
                                     query_chunk=query_chunk, matrix_dtype=torch.float16)
                else:
                    if table.is_floating_point() or table.dtype == torch.bool or bool((table.abs() <= 2 ** 24).all()):
                        rom = FixedROM(table)
                    else:
                        raise ValueError("fixed integer memory must be exactly representable in float32")
                name = f"circuit_{len(self.circuits)}"
                self.root.add_module(name, rom.eval())
                self.circuits.append({"module": name, "construction": rom.construction_spec()})
                self.roms[key] = name
            result = self.graph.call_module(self.roms[key], (ref,))
            if values.sample.dtype == torch.uint8:
                result = self.emit(torch.reshape, result, address.shape + value_shape)
        else:
            result = self.module(NeuralRead(count, max(_magnitude(values), 1.), value_shape),
                                 (ref, self.argument(values)))
        low, high = values.interval()
        low = torch.broadcast_to(low, values.shape).amin(0)
        high = torch.broadcast_to(high, values.shape).amax(0)
        return result, low, high

    def indexed(self, name, facts, sample):
        source = facts[0]
        if name == "index_copy":
            dimension, index, update = facts[1].sample, facts[2].sample, facts[3]
            if not facts[1].fixed or not facts[2].fixed:
                raise ValueError("index_copy requires fixed destination wires")
            dimension %= len(source.shape)
            route = torch.arange(source.shape[dimension])
            for new, destination in enumerate(index.tolist()):
                route[destination] = source.shape[dimension] + new
            tensors = _Fact([source.sample, update.sample], items=(source, update))
            cat_schema = torch._C.parse_schema("aten::cat(Tensor[] tensors, int dim=0) -> Tensor")
            cat_sample = torch.empty(tuple(sample.shape[:dimension]) +
                (source.shape[dimension] + update.shape[dimension],) + tuple(sample.shape[dimension + 1:]), device="meta")
            joined = self.wire(cat_schema, [tensors, _Fact(dimension)], cat_sample)
            index_schema = torch._C.parse_schema("aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")
            return self.wire(index_schema, [joined, _Fact(dimension), _Fact(route)], sample)
        index_facts = facts[1].items if name == "index" else None
        static = facts[1].fixed if name == "index" else facts[2].fixed
        schema = self.current_schema
        if static:
            return self.wire(schema, facts, sample)
        if name == "gather":
            dimension = facts[1].sample % len(source.shape)
            index = facts[2]
            if not source.fixed:
                from .neural_memory import NeuralRead
                order = tuple(i for i in range(len(source.shape)) if i != dimension) + (dimension,)
                values_ref = self.argument(source)
                for axis, size in enumerate(index.shape):
                    if axis != dimension and size != source.shape[axis]:
                        values_ref = self.emit(torch.ops.aten.slice.Tensor, values_ref, axis, 0, size, 1)
                values_ref = self.emit(torch.permute, values_ref, order)
                index_ref = self.emit(torch.permute, self.address(index, source.shape[dimension]), order)
                result = self.module(NeuralRead(source.shape[dimension], max(_magnitude(source), 1.)),
                                     (index_ref, values_ref))
                result = self.emit(torch.permute, result, tuple(order.index(i) for i in range(len(order))))
                return self.fact(sample, result, source.interval()[0].min(), source.interval()[1].max(), _integer(source))
            # Construction-time fixed coordinate grids turn a gather into one
            # flat neural ROM address without executing a runtime gather.
            address_ref = self.argument(index, floating=True)
            self.address(index, source.shape[dimension])
            stride = math.prod(source.shape[dimension + 1:])
            address_ref = self.emit(torch.mul, address_ref, stride)
            base = torch.zeros(index.shape)
            for axis, size in enumerate(index.shape):
                if axis != dimension:
                    coordinate = torch.arange(size).reshape((1,) * axis + (size,) + (1,) * (len(index.shape) - axis - 1))
                    base += coordinate * math.prod(source.shape[axis + 1:])
            address_ref = self.emit(torch.add, address_ref, self.constant(base))
            address = self.fact(torch.empty(index.shape, dtype=torch.int64, device="meta"), address_ref,
                                0, math.prod(source.shape) - 1, True)
            flat = _Fact(source.sample.reshape(-1))
            result, low, high = self.read(flat, address)
            return self.fact(sample, result, low, high, _integer(source))
        if name in {"index_select", "select"}:
            dimension, index = facts[1].sample % len(source.shape), facts[2]
        else:
            assert index_facts is not None, "advanced indices require an explicit list"
            indexed = [(axis, item) for axis, item in enumerate(index_facts) if item.sample is not None]
            if len(indexed) != 1:
                return self.advanced_index(source, indexed, sample)
            dimension, index = indexed[0]
        order = (dimension,) + tuple(i for i in range(len(source.shape)) if i != dimension)
        permuted_sample = source.sample.permute(order)
        if source.fixed:
            permuted = _Fact(permuted_sample)
        else:
            low, high = [torch.broadcast_to(x, source.shape).permute(order) for x in source.interval()]
            permuted = self.fact(permuted_sample, self.emit(torch.permute, self.argument(source), order), low, high, _integer(source))
        result, low, high = self.read(permuted, index, negative=name == "index")
        result_shape = index.shape + tuple(source.shape[i] for i in order[1:])
        query_rank = len(index.shape)
        output_order = (tuple(range(query_rank, query_rank + dimension)) + tuple(range(query_rank))
                        + tuple(range(query_rank + dimension, len(result_shape))))
        if output_order != tuple(range(len(result_shape))):
            result = self.emit(torch.permute, result, output_order)
        # A dynamic address selects any entry; preserve each remaining field's
        # domain, then move the static value dimensions into their output order.
        low, high = [torch.broadcast_to(x, result_shape).permute(output_order) for x in (low, high)]
        return self.fact(sample, result, low, high, _integer(source))

    def advanced_index(self, source, indexed, sample):
        indexed_axes = [axis for axis, _ in indexed]
        if any(item.sample.dtype == torch.bool for _, item in indexed):
            raise ValueError("data-dependent Boolean indexing has no fixed output shape")
        broadcast = torch.broadcast_shapes(*(item.shape for _, item in indexed))
        contiguous = indexed_axes == list(range(indexed_axes[0], indexed_axes[-1] + 1))
        free_axes = [axis for axis in range(len(source.shape)) if axis not in indexed_axes]
        before = indexed_axes[0] if contiguous else 0
        base = torch.zeros(sample.shape)
        address = None
        for axis, item in indexed:
            ref = self.address(item, source.shape[axis], negative=True)
            ref = self.emit(torch.reshape, self.emit(torch.ops.aten.expand.default, ref, broadcast, implicit=False),
                            (1,) * before + broadcast + (1,) * (len(sample.shape) - before - len(broadcast)))
            ref = self.emit(torch.mul, ref, math.prod(source.shape[axis + 1:]))
            address = ref if address is None else self.emit(torch.add, address, ref)
        for axis in free_axes:
            output_axis = (axis if contiguous and axis < before else
                           free_axes.index(axis) + len(broadcast) if not contiguous else
                           before + len(broadcast) + free_axes.index(axis) - before)
            coordinate = torch.arange(source.shape[axis]).reshape((1,) * output_axis +
                (source.shape[axis],) + (1,) * (len(sample.shape) - output_axis - 1))
            base += coordinate * math.prod(source.shape[axis + 1:])
        address = self.emit(torch.add, address, self.constant(base))
        address = self.fact(torch.empty(sample.shape, dtype=torch.int64, device="meta"), address,
                            0, math.prod(source.shape) - 1, True)
        if source.fixed:
            flat = _Fact(source.sample.reshape(-1))
        else:
            flat = self.fact(torch.empty((math.prod(source.shape),), device="meta"),
                self.emit(torch.reshape, self.argument(source), (-1,)),
                *(torch.broadcast_to(x, source.shape).reshape(-1) for x in source.interval()), _integer(source))
        result, low, high = self.read(flat, address)
        return self.fact(sample, result, low, high, _integer(source))

    def numerical(self, name, facts, sample):
        from . import neural_math as circuits
        a = facts[0]
        bound = max(_magnitude(a), 1.)
        args = [self.argument(a, floating=True)]
        integer = False
        if name == "mul":
            b = facts[1]
            b_bound = max(_magnitude(b), 1.)
            if bound <= b_bound:
                module = circuits.BinaryMultiply(bound, b_bound)
                args.append(self.argument(b, floating=True))
            else:
                module = circuits.BinaryMultiply(b_bound, bound)
                args = [self.argument(b, floating=True), args[0]]
            al, ah = a.interval(); bl, bh = b.interval()
            products = torch.stack(torch.broadcast_tensors(al * bl, al * bh, ah * bl, ah * bh))
            low, high = products.amin(0), products.amax(0)
        elif name == "div":
            b = facts[1]
            low_b, high_b = b.interval()
            if bool(((low_b <= 0) & (high_b >= 0)).any()) and not b.minimum_abs:
                raise ValueError("division needs a proven nonzero denominator domain")
            minimum = b.minimum_abs or min(float(low_b.abs().min()), float(high_b.abs().min()))
            quotient = bound / minimum
            module = circuits.Divide(bound, max(_magnitude(b), 1.), max(quotient, 1.))
            args.append(self.argument(b, floating=True))
            low, high = -quotient, quotient
        elif name in {"floor", "round", "frac", "trunc"}:
            module = getattr(circuits, {"floor": "Floor", "round": "Round", "frac": "Frac", "trunc": "Trunc"}[name])(bound)
            low, high = a.interval()
            if name == "floor": low, high = low.floor(), high.floor()
            elif name == "round": low, high = low.round(), high.round()
            elif name == "trunc": low, high = low.trunc(), high.trunc()
            else: low, high = -1., 1.
            integer = name != "frac"
        elif name == "remainder":
            if not facts[1].fixed or torch.as_tensor(facts[1].sample).numel() != 1:
                raise ValueError("remainder requires a fixed scalar modulus")
            modulus = float(torch.as_tensor(facts[1].sample))
            module = circuits.Remainder(modulus, bound)
            low, high = (0., modulus) if modulus > 0 else (modulus, 0.)
            integer = _integer(a) and modulus.is_integer()
            if integer:
                low, high = (0., modulus - 1.) if modulus > 0 else (modulus + 1., 0.)
        elif name in {"sin", "cos"}:
            module = getattr(circuits, name.title())(bound)
            low, high = -1., 1.
        elif name == "atan2":
            module = circuits.Atan2(max(bound, _magnitude(facts[1])))
            args.append(self.argument(facts[1], floating=True))
            low, high = -math.pi, math.pi
        elif name in {"sqrt", "square"}:
            low_a, high_a = a.interval()
            if name == "sqrt":
                if float(low_a.min()) < 0:
                    raise ValueError("square root needs a proven nonnegative domain")
                module = circuits.Sqrt(float(high_a.max()))
                low, high = low_a.sqrt(), high_a.sqrt()
            else:
                module = circuits.BinarySquare(bound)
                low = torch.where((low_a <= 0) & (high_a >= 0), 0., torch.minimum(low_a.square(), high_a.square()))
                high = torch.maximum(low_a.square(), high_a.square())
        else:
            raise ValueError(f"no numerical circuit for {name}")
        spec = module.construction_spec()
        error = spec.get("float32_absolute_error_bound", spec.get("absolute_error_bound", 0.))
        if name in {"square", "sqrt"}:
            # Both constructions execute nonnegative ReLU carries. Their
            # one-sided range is stronger than a signed approximation bound.
            low, high = torch.clamp_min(low - error, 0), high + error
        elif error:
            low, high = low - error, high + error
        activation_bounds = spec.get("output_activation_bounds")
        if activation_bounds is not None:
            low = torch.clamp_min(torch.as_tensor(low), activation_bounds[0])
            high = torch.clamp_max(torch.as_tensor(high), activation_bounds[1])
        return self.fact(sample, self.module(module, args), low, high, integer)

    def operation(self, node, facts):
        schema = torch._C.parse_schema(node.schema())
        self.current_schema = schema
        name = schema.name.removeprefix("aten::")
        scope = node.scopeName().split("/")[-1].removeprefix("__module.")
        selector = self.selectors.get(scope)
        if selector is not None and name in {"pad", "constant_pad_nd"}:
            spec = selector["construction"]
            low, high = facts[0].interval()
            padding = facts[1].sample
            fill = facts[3].sample if name == "pad" else facts[2].sample
            if facts[0].shape[-1] != spec["input_count"] or padding != [0, (1 << (spec["input_count"] - 1).bit_length()) - spec["input_count"]] or fill != spec["score_bound"] or float(low.min()) < 0 or float(high.max()) > spec["score_bound"]:
                raise ValueError(f"selector {scope} requires proven scores within [0,{spec['score_bound']}], inferred [{float(low.min())},{float(high.max())}]")
            selector["input_checks"] += 1
        if name not in _SOURCE:
            raise ValueError(f"unsupported source operator {schema.name}.{schema.overload_name}")
        if name == "__iand__":
            schema = torch._C.parse_schema("aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor")
            name = "__and__"
        if all(x.fixed for x in facts):
            return _Fact(_invoke(schema, facts))
        if any(x.codec for x in facts):
            raise ValueError("display byte codecs cannot feed model computation")
        if name == "size":
            return _Fact(list(facts[0].shape) if len(facts) == 1 else facts[0].shape[facts[1].sample])
        if name == "Int":
            if any(use.user.kind() != "aten::select" or use.offset != 2 for use in node.output().uses()):
                raise ValueError("activation-to-Python integer extraction is only lowerable as a scalar memory address")
            if facts[0].shape != ():
                raise ValueError("scalar memory addresses require zero-dimensional activations")
            sample = torch.empty((), dtype=torch.int64, device="meta")
            if not _integer(facts[0]):
                return self.numerical("trunc", facts, sample)
            return self.fact(sample, self.argument(facts[0]), *facts[0].interval(), True,
                             semantic=facts[0].semantic)
        if name == "select" and not facts[2].fixed:
            dimension = facts[1].sample % len(facts[0].shape)
            sample = torch.empty(facts[0].shape[:dimension] + facts[0].shape[dimension + 1:],
                                 dtype=facts[0].sample.dtype, device="meta")
            return self.indexed(name, facts, sample)
        sample = _invoke(schema, facts, metadata=True)
        if name in _WIRES:
            return self.wire(schema, facts, sample)
        if name in {"index", "index_select", "gather", "index_copy"}:
            return self.indexed(name, facts, sample)
        a = facts[0]
        al, ah = a.interval()
        ar = self.argument(a, floating=True)
        if name == "expand_as":
            if a.fixed:
                return _Fact(a.sample.expand(facts[1].shape))
            return self.fact(sample, self.emit(torch.ops.aten.expand_as.default, ar, self.argument(facts[1])), al, ah, a.integer)
        if name in {"zeros_like", "ones_like", "full_like"}:
            fill = {"zeros_like": 0., "ones_like": 1.}.get(name, facts[1].sample if name == "full_like" else 0.)
            return _Fact(torch.full(sample.shape, fill, dtype=sample.dtype))
        if name == "to":
            dtype = sample.dtype
            if dtype == torch.uint8:
                return self.fact(sample, self.emit(torch.ops.aten.to.dtype, ar, 0, False, False, None), al, ah, True, codec=True)
            if dtype == torch.bool:
                result = self.comparison("ne", a, _Fact(0.))
                return self.fact(sample, result, 0., 1., True)
            if selector is not None and dtype == torch.int64:
                selected = list(node.inputs())[0].node()
                selection_args = list(selected.inputs())
                source_range = str(node.sourceRange())
                line_prefix = f"{selector['source_file']}("
                if selected.kind() != "aten::select" or selection_args[1].toIValue() % len(self.environment[selection_args[0]].shape) != len(self.environment[selection_args[0]].shape) - 1 or selection_args[2].toIValue() != 1 or line_prefix not in source_range or not any(f"{line_prefix}{line})" in source_range for line in selector["return_lines"]):
                    raise ValueError(f"selector {scope} contract did not match its original index-slot return")
                if not selector["input_checks"]:
                    raise ValueError(f"selector {scope} has no validated score input")
                selector["index_checks"] += 1
                return self.fact(sample, ar, 0., selector["construction"]["input_count"] - 1, True)
            if not dtype.is_floating_point and not _integer(a):
                return self.numerical("trunc", facts, sample)
            return self.fact(sample, ar, al, ah, a.integer, condition=a.condition,
                             condition_columns=a.condition_columns, minimum_abs=a.minimum_abs,
                             semantic=a.semantic)
        if name in {"gt", "ge", "lt", "le", "eq", "ne"}:
            b = facts[1]
            condition = ("cmp", name, a, b)
            columns = None
            if name == "gt" and b.fixed and bool(torch.as_tensor(b.sample).eq(0).all()):
                columns = a.threshold_columns
                if a.condition is not None and _integer(a) and float(al.min()) >= 0 and float(ah.max()) <= 1:
                    condition = a.condition
            if columns is not None:
                condition = None
            return self.fact(sample, self.comparison(name, a, b), 0., 1., True,
                             predicate=(name, a, b), condition=condition, condition_columns=columns)
        if name in {"__and__", "__or__", "bitwise_not", "any", "all"}:
            if a.sample.dtype != torch.bool:
                if name not in {"__and__", "__or__"}:
                    raise ValueError("Boolean circuits require Boolean source operands")
                b = facts[1]
                variable, mask_fact = (b, a) if a.fixed else (a, b)
                if not mask_fact.fixed or torch.as_tensor(mask_fact.sample).numel() != 1:
                    raise ValueError("integer bit operations require a fixed scalar mask")
                mask = int(torch.as_tensor(mask_fact.sample))
                vl, vh = variable.interval()
                if not _integer(variable) or float(vl.min()) < 0 or _magnitude(variable) > 2 ** 24 or not 0 <= mask <= 2 ** 24:
                    raise ValueError("integer bit operations require nonnegative exact float32 integers")
                from .neural_math import Remainder, Trunc
                vr = self.argument(variable, floating=True)
                result = self.emit(torch.mul, vr, 0.)
                for bit in range(mask.bit_length()):
                    place = 2 ** bit
                    if mask & place:
                        bound = max(_magnitude(variable) / place, 1.)
                        digit = self.module(Trunc(bound), (self.emit(torch.mul, vr, 1. / place),))
                        digit = self.module(Remainder(2, math.ceil(bound)), (digit,))
                        result = self.emit(torch.add, result, self.emit(torch.mul, digit, place))
                low, high = 0., float(mask)
                if name == "__or__":
                    result = self.emit(torch.sub, self.emit(torch.add, vr, mask), result)
                    low, high = float(mask), float(vh.max()) + mask
                return self.fact(sample, result, low, high, True)
            if name == "bitwise_not": result = self.emit(torch.sub, 1., ar)
            elif name in {"any", "all"}:
                dimensions = facts[1].sample if len(facts) > 1 else None
                keepdim = facts[2].sample if len(facts) > 2 else False
                summed = self.emit(torch.sum, ar, dim=dimensions, keepdim=keepdim)
                count = math.prod(a.shape) if dimensions is None else math.prod(a.shape[d] for d in ([dimensions] if isinstance(dimensions, int) else dimensions))
                result = self.step(summed if name == "any" else self.emit(torch.sub, summed, count - .5))
            else:
                result = self.step(self.emit(torch.sub, self.emit(torch.add, ar, self.argument(facts[1], floating=True)),
                                             1.5 if name == "__and__" else .5))
            condition = (("not", a.condition) if name == "bitwise_not" else
                         ("and" if name == "__and__" else "or", a.condition, facts[1].condition)
                         if name in {"__and__", "__or__"} else None)
            return self.fact(sample, result, 0., 1., True, condition=condition)
        if name == "where":
            if a.sample.dtype != torch.bool:
                raise ValueError("conditional routing requires Boolean source predicates")
            left, right = facts[1:3]
            ll, lh = _refined(left, a.condition, True)
            rl, rh = _refined(right, a.condition, False)
            low, high = torch.minimum(ll, rl), torch.maximum(lh, rh)
            minimum_abs = 0.
            if a.predicate is not None:
                comparison, measured, limit = a.predicate
                if measured.predicate is not None and measured.predicate[0] == "abs" and limit.fixed:
                    original = measured.predicate[1]
                    other = left if comparison in {"lt", "le"} and right is original else right if comparison in {"gt", "ge"} and left is original else None
                    if other is not None and other.fixed:
                        minimum_abs = min(float(torch.as_tensor(limit.sample)),
                                          float(torch.as_tensor(other.sample).abs().min()))
            return self.fact(sample, self.choose(ar, left, right), low, high, _integer(left) and _integer(right), minimum_abs=minimum_abs)
        if name in {"minimum", "maximum", "clamp", "clamp_min", "clamp_max"}:
            if name in {"minimum", "maximum"}:
                b = facts[1]; bl, bh = b.interval()
                op = torch.minimum if name == "minimum" else torch.maximum
                result = self.choose(self.comparison("lt" if name == "minimum" else "gt", a, b), a, b)
                return self.fact(sample, result, op(al, bl), op(ah, bh), _integer(a) and _integer(b))
            result, low, high = ar, al, ah
            limits = (("min", facts[1]), ("max", facts[2])) if name == "clamp" else ((name.removeprefix("clamp_"), facts[1]),)
            for direction, limit in limits:
                if limit.sample is None: continue
                if not limit.fixed: raise ValueError("clamp limits must be construction constants")
                operation = torch.maximum if direction == "min" else torch.minimum
                current = self.fact(sample, result, low, high, _integer(a))
                result = self.choose(self.comparison("lt" if direction == "min" else "gt", current, limit), limit, current)
                low, high = operation(low, torch.as_tensor(limit.sample)), operation(high, torch.as_tensor(limit.sample))
            return self.fact(sample, result, low, high, _integer(a))
        if name in {"relu", "abs", "neg"}:
            if name == "relu": result, low, high = self.emit(torch.relu, ar), al.relu(), ah.relu()
            elif name == "neg": result, low, high = self.emit(torch.neg, ar), -ah, -al
            else:
                result = self.emit(torch.add, self.emit(torch.relu, ar), self.emit(torch.relu, self.emit(torch.neg, ar)))
                low = torch.where((al <= 0) & (ah >= 0), 0., torch.minimum(al.abs(), ah.abs()))
                high = torch.maximum(al.abs(), ah.abs())
            result_fact = self.fact(sample, result, low, high, a.integer,
                                    predicate=("abs", a) if name == "abs" else None)
            if name == "relu" and a.candidate_partition is not None:
                result_fact.partition = a.candidate_partition
            if name == "relu" and a.relu_clauses is not None:
                weight, bias = a.relu_clauses
                positive, negative = (weight > 0).float(), (weight < 0).float()
                conflict = positive @ negative.T + negative @ positive.T
                off_diagonal = ~torch.eye(len(weight), dtype=torch.bool)
                if bool((conflict[off_diagonal] > 0).all()):
                    result_fact.exclusive = True
            return result_fact
        if name in {"add", "sub", "rsub", "mul", "div"}:
            b = facts[1]; bl, bh = b.interval()
            if name in {"add", "sub", "rsub"}:
                alpha = facts[2].sample
                if not facts[2].fixed: raise ValueError("affine scale must be fixed")
                if name == "rsub": low, high = bl - alpha * ah, bh - alpha * al
                else:
                    sign = alpha if name == "add" else -alpha
                    low, high = al + sign * (bl if sign >= 0 else bh), ah + sign * (bh if sign >= 0 else bl)
                integer = (_integer_mask(a) & _integer_mask(b)) if float(alpha).is_integer() else False
                result = self.fact(sample, self.call(schema, facts, floating=True), low, high, integer)
                if name == "rsub" and b.fixed and torch.as_tensor(b.sample).numel() == 1 and float(torch.as_tensor(b.sample)) == 1 and alpha == 1:
                    result.complement_of = a
                if name == "sub" and a.fixed and torch.as_tensor(a.sample).numel() == 1 and float(torch.as_tensor(a.sample)) == 1 and alpha == 1:
                    result.complement_of = b
                if name == "add" and alpha == 1:
                    offset, values = (b, a) if b.offset_partition is not None else (a, b)
                    if offset.offset_partition is not None:
                        axis, group, weight = offset.offset_partition
                        vl, vh = values.interval()
                        if bool((vl >= 0).all()) and bool((vh <= -weight).all()):
                            result.candidate_partition = (axis, group, values.interval())
                    if a.partition_piece is not None and b.partition_piece is not None:
                        ga, ia, da = a.partition_piece; gb, ib, db = b.partition_piece
                        if ga == gb and {ia, ib} == {0, 1}:
                            result.low = torch.minimum(da[0], db[0])
                            result.high = torch.maximum(da[1], db[1])
                return result
            if name == "mul" and not (a.fixed or b.fixed):
                if a.sample.dtype == torch.bool or b.sample.dtype == torch.bool or (_integer(a) and float(al.min()) >= 0 and float(ah.max()) <= 1) or (_integer(b) and float(bl.min()) >= 0 and float(bh.max()) <= 1):
                    gate, value = (a, b) if _integer(a) and float(al.min()) >= 0 and float(ah.max()) <= 1 else (b, a)
                    low, high = value.interval()
                    return self.fact(sample, self.gate(self.argument(value), self.argument(gate), value), torch.minimum(low, torch.zeros_like(low)), torch.maximum(high, torch.zeros_like(high)), _integer(value))
                return self.numerical("square" if a is b else name, facts, sample)
            if name == "div" and not b.fixed:
                return self.numerical(name, facts, sample)
            if name == "div" and len(facts) > 2 and facts[2].sample is not None:
                rounding = facts[2].sample
                if rounding not in {"floor", "trunc"}:
                    raise ValueError("unsupported source division rounding mode")
                if not b.fixed:
                    unrounded = self.numerical("div", facts[:2], torch.empty(sample.shape, device="meta"))
                else:
                    if bool((bl == 0).any()):
                        raise ValueError("fixed division denominator must be nonzero")
                    corners = torch.stack(torch.broadcast_tensors(al / bl, al / bh, ah / bl, ah / bh))
                    unrounded = self.fact(torch.empty(sample.shape, device="meta"),
                        self.emit(torch.div, ar, self.argument(b, floating=True)), corners.amin(0), corners.amax(0))
                return self.numerical(rounding, [unrounded], sample)
            if name == "div" and bool((bl == 0).any()):
                raise ValueError("fixed division denominator must be nonzero")
            products = torch.stack(torch.broadcast_tensors(al * bl, al * bh, ah * bl, ah * bh)) if name == "mul" else torch.stack(torch.broadcast_tensors(al / bl, al / bh, ah / bl, ah / bh))
            integer = _integer_mask(a) & _integer_mask(b) if name == "mul" else False
            result = self.fact(sample, self.call(schema, facts, floating=True), products.amin(0), products.amax(0), integer)
            variable, coefficient = (b, a) if a.fixed else (a, b)
            if name == "mul" and coefficient.fixed and variable.partition is not None:
                weight = torch.as_tensor(coefficient.sample, dtype=torch.float64)
                if bool((weight < 0).all()):
                    axis, group, _ = variable.partition
                    result.offset_partition = (axis, group, weight)
            return result
        if name in {"sum", "mean"}:
            def reduce(value):
                full = torch.broadcast_to(value, a.shape)
                if name == "sum": return _invoke(schema, [_Fact(full)] + facts[1:])
                return _invoke(schema, [_Fact(full)] + facts[1:])
            return self.fact(sample, self.call(schema, facts, floating=True), reduce(al), reduce(ah), _integer(a) and name == "sum")
        if name == "linear":
            weight, bias = facts[1].sample, facts[2].sample
            if not facts[1].fixed or not facts[2].fixed:
                raise ValueError("linear maps require fixed weights and bias")
            weight = weight.double(); positive, negative = weight.clamp_min(0), weight.clamp_max(0)
            low, high = torch.broadcast_to(al, a.shape), torch.broadcast_to(ah, a.shape)
            offset = None if bias is None else bias.double()
            low_out = F.linear(low, positive, offset) + F.linear(high, negative)
            high_out = F.linear(high, positive, offset) + F.linear(low, negative)
            if a.exclusive:
                low_out = weight.amin(-1).clamp_max(0) + (offset if offset is not None else 0)
                high_out = weight.amax(-1).clamp_min(0) + (offset if offset is not None else 0)
            result = self.fact(sample, self.call(schema, facts, floating=True), low_out, high_out,
                               _integer(a) and _integer(facts[1]) and (bias is None or _integer(facts[2])))
            if _integer(a) and float(al.min()) >= 0 and float(ah.max()) <= 1 and bias is not None:
                signed = bool(((weight == -1) | (weight == 0) | (weight == 1)).all())
                positives = (weight == 1).sum(-1)
                if signed and bool((offset == 1 - positives).all()):
                    result.relu_clauses = (weight, offset)
                if a.condition_columns is not None:
                    if signed and bool((offset == .5 - positives).all()):
                        result.threshold_columns = tuple(("and", *(a.condition_columns[i] if coefficient > 0 else ("not", a.condition_columns[i]) for i, coefficient in enumerate(row) if coefficient)) for row in weight)
                    elif bool(((weight == 0) | (weight == 1)).all()) and bool((offset == -.5).all()):
                        result.threshold_columns = tuple(("or", *(a.condition_columns[i] for i, coefficient in enumerate(row) if coefficient)) for row in weight)
            return result
        if name in {"mm", "bmm", "matmul", "addmm"}:
            raise ValueError("source matrix multiplication needs an explicit fixed-weight interval rule")
        return self.numerical(name, facts, sample)

    def compile(self):
        for ordinal, node in enumerate(self.source.nodes()):
            kind = node.kind()
            try:
                if list(node.blocks()):
                    raise ValueError("data-dependent blocks cannot compile into static neural wiring")
                if kind == "prim::Constant":
                    result = _Fact(node.output().toIValue())
                else:
                    facts = [self.environment[x] for x in node.inputs()]
                    if kind in {"prim::ListConstruct", "prim::TupleConstruct"}:
                        constructor = list if kind == "prim::ListConstruct" else tuple
                        result = _Fact(constructor(x.sample for x in facts), items=tuple(facts))
                    elif kind in {"prim::ListUnpack", "prim::TupleUnpack"}:
                        results = facts[0].items
                        assert results is not None, "unpacked source values need explicit items"
                        self.environment.update(zip(node.outputs(), results, strict=True))
                        continue
                    elif kind == "prim::NumToTensor":
                        if not facts[0].fixed: raise ValueError("shape scalar must be fixed")
                        result = _Fact(torch.tensor(facts[0].sample))
                    elif kind.startswith("aten::"):
                        result = self.operation(node, facts)
                        if kind == "aten::__iand__":
                            self.environment[list(node.inputs())[0]] = result
                    else:
                        raise ValueError(f"unsupported source node {kind}")
                outputs = list(node.outputs())
                if len(outputs) != 1:
                    raise ValueError("source node output arity is not supported")
                self.environment[outputs[0]] = result
                self.operators[kind] += 1
            except (ValueError, TypeError, RuntimeError, KeyError, IndexError, AttributeError) as error:
                raise ValueError(f"neural lowering rejected {self.method} node {ordinal} ({kind}): {error}\n{node.sourceRange()}") from error
        result_facts = [self.environment[x] for x in self.source.outputs()]
        returns = [self.argument(x, floating=True) for x in result_facts]
        self.graph.output(returns[0] if len(returns) == 1 else tuple(returns))
        module = fx.GraphModule(self.root, self.graph, f"Neural_{self.method}").eval()
        contracts = [{key: value for key, value in contract.items() if key not in {"source_file", "return_lines"}}
                     for contract in self.selectors.values() if contract["index_checks"]]
        return module, {"compiler": "JIT-to-neural-FX", "source_method": self.method,
                        "source_operators": dict(sorted(self.operators.items())),
                        "inputs": self.input_contracts, "circuits": self.circuits,
                        "outputs": self.domains(result_facts), "verified_subcircuits": contracts}

    def domains(self, facts):
        results = []
        for fact in facts:
            if fact.items is not None:
                results.extend(self.domains(fact.items))
            else:
                low, high = fact.interval()
                results.append({"shape": list(fact.shape), "minimum": float(low.min()),
                                "maximum": float(high.max()), "integer": _integer(fact)})
        return results


def lower_neural_method(source: torch.jit.ScriptModule, method: str,
                        input_bounds, *, source_modules: nn.Module | None = None):
    """Return a constructed FX module and its construction report.

    ``input_bounds`` maps source argument names to full-shaped ``(low, high)``
    tensors, or is the corresponding positional sequence. ``source_modules``
    associates a freshly frozen source's retained scopes with verified original
    selector constructors. It proves exclusion of fixed padding from selected
    real indices; loaded sources without scopes receive no such theorem.
    A caller must validate external domains and audit the exact deployed graph.
    """
    if list(source.named_parameters()) or list(source.named_buffers()):
        raise ValueError("source program must freeze construction parameters and buffers")
    return _Compiler(source, method, input_bounds, source_modules).compile()
