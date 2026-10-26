"""A restricted constructed-neural IR carried by frozen TorchScript.

TorchScript is the serializer and evaluator, not the architecture certificate.
This audit admits only fixed affine maps and wires, ReLU neurons, and strict
zero-threshold neurons. It propagates shapes and construction-constant
provenance through every node. Multiplication by an activation, data-dependent
indexing, and every unlisted operator are rejected, including built-in ATen
operators. Nonlinear source operations must be lowered into admitted circuits
before calling this audit.

Declared input shapes are part of the execution contract: the generic host
must validate them before evaluation. A terminal float-to-uint8 cast is the
only output codec; it cannot feed further model computation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import operator
from typing import Mapping, Sequence

import torch


_INPUT_FLOATS = {torch.float32, torch.float64}
_FLOATS = _INPUT_FLOATS | {torch.float16, torch.bfloat16}
_DTYPES = {0: torch.uint8, 1: torch.int8, 2: torch.int16, 3: torch.int32,
           4: torch.int64, 5: torch.float16, 6: torch.float32,
           7: torch.float64, 11: torch.bool, 15: torch.bfloat16}
_AFFINE = {
    "add": {"Tensor", "Scalar"}, "sub": {"Tensor", "Scalar"},
    "rsub": {"Scalar", "Tensor"}, "mul": {"Tensor", "Scalar"},
    "div": {"Tensor", "Scalar", "Tensor_mode", "Scalar_mode"},
    "neg": {""}, "linear": {""}, "mm": {""}, "bmm": {""},
    "matmul": {""}, "addmm": {""},
    "sum": {"", "dim_IntList"}, "mean": {"", "dim"},
}
_WIRES = {
    "slice": {"Tensor"}, "select": {"int"}, "index_select": {""},
    "gather": {""}, "index": {"Tensor"},
    "reshape": {""}, "view": {""}, "flatten": {"using_ints"},
    "unsqueeze": {""}, "squeeze": {"", "dim", "dims"},
    "transpose": {"int"}, "permute": {""}, "t": {""},
    "expand": {""}, "repeat": {""}, "cat": {""}, "stack": {""},
    "split": {"Tensor", "sizes"}, "split_with_sizes": {""},
    "chunk": {""}, "unbind": {"int"},
    "pad": {""}, "constant_pad_nd": {""},
    "clone": {""}, "contiguous": {""}, "detach": {""}, "alias": {""},
}


@dataclass(frozen=True)
class _Fact:
    value: object
    fixed: bool
    codec: bool = False
    items: tuple[_Fact, ...] | None = None


def _aggregate(values, constructor):
    return _Fact(constructor(item.value for item in values),
                 all(item.fixed for item in values), items=tuple(values))


def _meta(value):
    if isinstance(value, torch.Tensor):
        return torch.empty_strided(value.shape, value.stride(),
                                   dtype=value.dtype, device="meta")
    if isinstance(value, (list, tuple)):
        return type(value)(_meta(item) for item in value)
    return value


def _check_constant(value):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_complex():
            raise ValueError("construction tensors must be real, dense tensors")
        if value.requires_grad:
            raise ValueError("construction tensors cannot require gradients")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("construction tensors must be finite")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("construction scalars must be finite")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_constant(item)
    elif not isinstance(value, (int, float, bool, str, torch.device, type(None))):
        raise ValueError(f"unsupported construction constant {type(value).__name__}")


def _constant_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _constant_tensors(item)


def _fixed(fact, purpose):
    if not fact.fixed:
        raise ValueError(f"{purpose} must be a construction constant")
    return fact.value


def _floating(fact, purpose):
    if isinstance(fact.value, torch.Tensor) and fact.value.dtype not in _FLOATS:
        raise ValueError(f"{purpose} requires floating activations")


def _schema(node):
    schema = torch._C.parse_schema(node.schema())
    return schema, schema.name.removeprefix("aten::"), schema.overload_name


def _execute(schema, name, overload, facts, fixed=None):
    """Shape-evaluate an explicitly admitted operator, never an unknown node."""
    fixed = all(fact.fixed for fact in facts) if fixed is None else fixed
    values = [fact.value if fixed else _meta(fact.value) for fact in facts]
    positional, keywords = _arguments(schema, values)
    operation = getattr(getattr(torch.ops.aten, name), overload or "default")
    result = operation(*positional, **keywords)
    if fixed:
        _check_constant(result)
    if isinstance(result, (tuple, list)):
        return _aggregate([_Fact(item, fixed) for item in result], type(result))
    return _Fact(result, fixed)


def _arguments(schema, values):
    positional, keywords = [], {}
    for argument, value in zip(schema.arguments, values, strict=True):
        if argument.kwarg_only:
            keywords[argument.name] = value
        else:
            positional.append(value)
    return tuple(positional), keywords


def _indices(index, size, purpose, *, negative=False):
    value = _fixed(index, purpose)
    if not isinstance(value, torch.Tensor) or value.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"{purpose} requires fixed integer tensor wires")
    lower = -size if negative else 0
    if value.numel() and not bool(((value >= lower) & (value < size)).all()):
        raise ValueError(f"{purpose} contains an out-of-range wire")


def _admit(node, facts):
    schema, name, overload = _schema(node)
    if name in _AFFINE and overload in _AFFINE[name]:
        for fact in facts:
            if not fact.fixed:
                _floating(fact, "affine maps")
        if name in {"add", "sub", "rsub"}:
            _fixed(facts[2], "affine scale")
        elif name == "mul" and not (facts[0].fixed or facts[1].fixed):
            raise ValueError("dynamic multiplication must lower into a neural circuit")
        elif name == "div":
            denominator = _fixed(facts[1], "division denominator")
            if not bool(torch.as_tensor(denominator).ne(0).all()):
                raise ValueError("fixed division denominator must be nonzero")
            coefficient = torch.as_tensor(denominator, dtype=facts[0].value.dtype).reciprocal()
            if not bool(torch.isfinite(coefficient).all()):
                raise ValueError("fixed division must have finite affine weights")
            if len(facts) > 2 and _fixed(facts[2], "division rounding mode") is not None:
                raise ValueError("rounded division must lower into a neural circuit")
        elif name == "linear":
            _fixed(facts[1], "linear weights")
            _fixed(facts[2], "linear bias")
            if any(fact.value is not None and fact.value.dtype != facts[0].value.dtype for fact in facts[1:]):
                raise ValueError("linear weights and bias must match the activation dtype")
        elif name in {"mm", "bmm", "matmul", "addmm"}:
            offset = int(name == "addmm")
            if not (facts[offset].fixed or facts[offset + 1].fixed):
                raise ValueError("matrix maps require fixed weights")
            for fact in facts[offset + 2:]:
                _fixed(fact, "matrix scale")
            matrix_facts = facts[:3] if name == "addmm" else facts[:2]
            if len({fact.value.dtype for fact in matrix_facts}) != 1:
                raise ValueError("matrix operands must have the same dtype")
        elif name in {"sum", "mean"}:
            for fact in facts[1:]:
                _fixed(fact, "reduction axes and dtype")
            if name == "mean":
                shape = facts[0].value.shape
                axes = facts[1].value if overload == "dim" else None
                axes = axes or range(len(shape))
                if math.prod(shape[axis] for axis in axes) == 0:
                    raise ValueError("mean requires a nonempty fixed affine reduction")
        result = _execute(schema, name, overload, facts)
        if not result.fixed:
            _floating(result, "affine maps")
        return result, "affine"

    if name in _WIRES and overload in _WIRES[name]:
        if name in {"cat", "stack"}:
            _fixed(facts[1], "wire axis")
        else:
            for fact in facts[1:]:
                _fixed(fact, "wire shape, axes, indices, and constants")
        if name in {"index_select", "gather"}:
            dimension = facts[1].value
            _indices(facts[2], facts[0].value.shape[dimension], "static index")
        elif name == "index":
            for dimension, index in enumerate(facts[1].value):
                if index is not None:
                    _indices(_Fact(index, True), facts[0].value.shape[dimension],
                             "static index", negative=True)
        elif name == "pad" and facts[2].value != "constant":
            raise ValueError("padding must be fixed constant affine wiring")
        return _execute(schema, name, overload, facts), "wire"

    if name == "relu" and overload == "":
        _floating(facts[0], "ReLU")
        return _execute(schema, name, overload, facts), "relu"
    if name == "gt" and overload in {"Tensor", "Scalar"}:
        _floating(facts[0], "hard threshold")
        zero = _fixed(facts[1], "hard-threshold reference")
        if not bool(torch.as_tensor(zero).eq(0).all()):
            raise ValueError("hard thresholds must compare an affine activation with zero")
        return _execute(schema, name, overload, facts), "hard_threshold"

    if name == "size" and overload in {"", "int"}:
        dimensions = list(facts[0].value.shape)
        return _Fact(dimensions if len(facts) == 1 else dimensions[_fixed(facts[1], "shape axis")], True), "metadata"
    if name == "Int" and overload in {"Tensor", "Scalar", "int", "float", "bool"}:
        return _Fact(int(_fixed(facts[0], "shape integer")), True), "metadata"
    if name == "expand_as" and overload == "":
        source, target = facts[0], facts[1].value
        return _Fact(source.value.expand(target.shape), source.fixed), "wire"
    if name == "to" and overload in {"dtype", "device", "other", "dtype_layout"}:
        arguments = {argument.name: fact for argument, fact in zip(schema.arguments, facts, strict=True)}
        for key, fact in arguments.items():
            if key not in {"self", "other"}:
                _fixed(fact, "dtype/device metadata")
        dtype = (arguments["other"].value.dtype if "other" in arguments
                 else _DTYPES.get(arguments["dtype"].value) if "dtype" in arguments and arguments["dtype"].value is not None
                 else facts[0].value.dtype)
        if "layout" in arguments and arguments["layout"].value not in {None, torch.strided}:
            raise ValueError("dtype wires require dense tensors")
        if dtype is None:
            raise ValueError("unsupported dtype conversion")
        source = facts[0]
        if not source.fixed and dtype not in _FLOATS and dtype != source.value.dtype:
            if dtype != torch.uint8 or source.value.dtype not in _FLOATS:
                raise ValueError("integer/Boolean casts must lower into neural circuits")
            return _Fact(source.value.to(dtype=dtype), False, codec=True), "output_codec"
        return _Fact(source.value.to(dtype=dtype), source.fixed), "wire"
    if name in {"zeros_like", "ones_like", "full_like"} and overload == "":
        for fact in facts[1:]:
            _fixed(fact, "constant-neuron metadata")
        arguments = {argument.name: fact.value for argument, fact in zip(schema.arguments, facts, strict=True)}
        if arguments.get("layout") not in {None, torch.strided}:
            raise ValueError("constant neurons require dense tensors")
        dtype = (_DTYPES[arguments["dtype"]] if arguments.get("dtype") is not None
                 else facts[0].value.dtype)
        fill = arguments.get("fill_value", int(name == "ones_like"))
        # The represented constant has one stored scalar regardless of its
        # declared activation shape. Auditing must not allocate giant arrays
        # merely to prove that a constant neuron ignores its template's data.
        value = torch.tensor(fill, dtype=dtype).expand(facts[0].value.shape)
        _check_constant(value.reshape(-1)[:1])
        return _Fact(value, True), "constant"
    raise ValueError(f"operator {node.kind()}.{overload or 'default'} is outside the neural IR")


def _leaves(fact):
    if fact.items is not None:
        for item in fact.items:
            yield from _leaves(item)
    else:
        yield fact


def _analyze(model, method, input_shapes, input_dtypes):
    if list(model.named_parameters()) or list(model.named_buffers()):
        raise ValueError("neural IR must freeze every parameter and fixed buffer")
    graph = getattr(model, method).inlined_graph.copy()
    torch._C._jit_pass_inline(graph)
    environment, descriptors = {}, []
    tensor_inputs = [value for value in graph.inputs() if isinstance(value.type(), torch._C.TensorType)]
    for supplied, label in ((input_shapes, "input shapes"), (input_dtypes, "input dtypes")):
        if supplied is not None and len(supplied) != len(tensor_inputs):
            raise ValueError(f"{label} must describe every tensor input exactly once")
    for index, value in enumerate(tensor_inputs):
        name = value.debugName().split(".")[0]
        if isinstance(input_shapes, Mapping) and name not in input_shapes:
            raise ValueError(f"input shapes are missing tensor input {name}")
        if isinstance(input_dtypes, Mapping) and name not in input_dtypes:
            raise ValueError(f"input dtypes are missing tensor input {name}")
        shape = (input_shapes[name] if isinstance(input_shapes, Mapping) else input_shapes[index]
                 if input_shapes is not None else value.type().sizes())
        if shape is None or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
            raise ValueError(f"input {name} requires a declared static nonnegative shape")
        dtype = (input_dtypes[name] if isinstance(input_dtypes, Mapping) else input_dtypes[index]
                 if input_dtypes is not None else value.type().dtype() or torch.float32)
        if dtype not in _INPUT_FLOATS:
            raise ValueError(f"input {name} must contain float32 or float64 activations")
        environment[value] = _Fact(torch.empty(tuple(shape), dtype=dtype, device="meta"), False)
        descriptors.append({"name": name, "shape": list(shape), "dtype": str(dtype)})
    operators, families, constants = Counter(), Counter(), []
    for ordinal, node in enumerate(graph.nodes()):
        kind = node.kind()
        try:
            if list(node.blocks()):
                raise ValueError("data-dependent blocks are outside the neural IR")
            if kind == "prim::Constant":
                value = node.output().toIValue()
                _check_constant(value)
                constants.extend(tensor.numel() * tensor.element_size()
                                 for tensor in _constant_tensors(value))
                results, family = [_Fact(value, True)], "constant"
            else:
                facts = [environment[value] for value in node.inputs()]
                if any(fact.codec or any(leaf.codec for leaf in _leaves(fact)) for fact in facts) and kind != "prim::TupleConstruct":
                    raise ValueError("uint8 output codecs must be terminal return leaves")
                if kind in {"prim::ListConstruct", "prim::TupleConstruct"}:
                    results, family = [_aggregate(facts, list if kind == "prim::ListConstruct" else tuple)], "wire"
                elif kind in {"prim::ListUnpack", "prim::TupleUnpack"}:
                    results = list(facts[0].items or (_Fact(value, facts[0].fixed) for value in facts[0].value))
                    family = "wire"
                elif kind == "prim::NumToTensor":
                    results, family = [_Fact(torch.tensor(_fixed(facts[0], "shape scalar")), True)], "metadata"
                elif kind.startswith("aten::"):
                    result, family = _admit(node, facts)
                    results = [result]
                else:
                    raise ValueError(f"operator {kind} is outside the neural IR")
            outputs = list(node.outputs())
            if len(outputs) != len(results):
                raise ValueError("operator output arity does not match the neural IR")
            environment.update(zip(outputs, results, strict=True))
            operators[kind] += 1
            families[family] += 1
        except (ValueError, TypeError, RuntimeError, IndexError, KeyError) as error:
            raise ValueError(f"neural IR rejected {method} node {ordinal} ({kind}): {error}") from error
    outputs, returned_codecs = [], set()
    for value in graph.outputs():
        for leaf in _leaves(environment[value]):
            if not isinstance(leaf.value, torch.Tensor):
                raise ValueError("neural IR outputs must be tensor activations")
            if leaf.value.dtype not in _FLOATS | {torch.bool} and not leaf.codec:
                raise ValueError("integer outputs require the terminal uint8 codec")
            if leaf.codec:
                returned_codecs.add(id(leaf))
            outputs.append({"shape": list(leaf.value.shape), "dtype": str(leaf.value.dtype),
                            "codec": "uint8" if leaf.codec else None})
    if any(fact.codec and id(fact) not in returned_codecs for fact in environment.values()):
        raise ValueError("uint8 output codecs must be terminal return leaves")
    audit = {"neural_ir_version": 1, "node_count": sum(operators.values()),
             "operators": dict(sorted(operators.items())),
             "primitive_families": dict(sorted(families.items())),
             "inputs": descriptors, "outputs": outputs,
             "constant_tensor_count": len(constants), "constant_tensor_bytes": sum(constants),
             "mutable_parameters": 0, "opaque_operators": 0,
             "dynamic_weights": 0, "dynamic_indices": 0}
    return audit, graph, environment


def audit_neural_graph(model: torch.jit.ScriptModule, method: str = "forward",
                       input_shapes: Mapping[str, Sequence[int]] | Sequence[Sequence[int]] | None = None,
                       input_dtypes: Mapping[str, torch.dtype] | Sequence[torch.dtype] | None = None) -> tuple[dict, str]:
    """Validate every node as a constructed neural primitive or static wire.

    Shapes/dtypes follow positional tensor inputs or their names (without the
    serialization suffix, e.g. ``state`` for ``state.1``). Floating inputs
    default to float32 when a saved graph has erased its dtype. No operator is
    admitted merely because it is built into PyTorch or has constant-looking
    sample values.
    """
    audit, graph, _ = _analyze(model, method, input_shapes, input_dtypes)
    return audit, str(graph)


def neural_method_fx(model: torch.jit.ScriptModule, method: str = "forward",
                     input_shapes: Mapping[str, Sequence[int]] | Sequence[Sequence[int]] | None = None,
                     input_dtypes: Mapping[str, torch.dtype] | Sequence[torch.dtype] | None = None) -> torch.fx.GraphModule:
    """Convert one already-admitted neural method to a portable FX evaluator.

    The same strict analysis runs before conversion. Construction tensors keep
    their dtypes and become fixed buffers, which relocate with ``module.to``.
    Metadata and constant-only subgraphs are folded; tensor-to-Python shape
    conversion never executes at runtime. Dtype casts retain their precision
    while device placement follows the inputs and relocated buffers.

    Callers must enforce the declared shape/dtype contract and use
    ``torch.compile(module, fullgraph=True)`` when compilation is required.
    Compilation errors propagate; this function supplies no alternate path.
    """
    audit, source, facts = _analyze(model, method, input_shapes, input_dtypes)
    graph, root, values, buffers = torch.fx.Graph(), torch.nn.Module(), {}, {}

    def materialize(value):
        if isinstance(value, torch.Tensor):
            key = id(value)
            if key not in buffers:
                name = f"constant_{len(buffers)}"
                root.register_buffer(name, value.detach())
                buffers[key] = graph.get_attr(name)
            return buffers[key]
        if isinstance(value, (tuple, list)):
            return type(value)(materialize(item) for item in value)
        return value

    for value in source.inputs():
        if value in facts:
            values[value] = graph.placeholder(value.debugName().split(".")[0])
    for node in source.nodes():
        outputs = list(node.outputs())
        if all(facts[value].fixed for value in outputs):
            values.update((value, facts[value].value) for value in outputs)
            continue
        kind = node.kind()
        inputs = list(node.inputs())
        arguments = [values[value] for value in inputs]
        if kind in {"prim::ListConstruct", "prim::TupleConstruct"}:
            results = [list(arguments) if kind == "prim::ListConstruct" else tuple(arguments)]
        elif kind in {"prim::ListUnpack", "prim::TupleUnpack"}:
            sequence = arguments[0]
            results = [sequence[index] if isinstance(sequence, (list, tuple))
                       else graph.call_function(operator.getitem, (sequence, index))
                       for index in range(len(outputs))]
        else:
            schema, name, overload = _schema(node)
            if name == "to":
                result = graph.call_method("to", (materialize(arguments[0]),),
                                           {"dtype": facts[outputs[0]].value.dtype})
            elif name == "expand_as":
                result = graph.call_function(torch.ops.aten.expand.default,
                                             (materialize(arguments[0]), list(facts[inputs[1]].value.shape)))
            else:
                positional, keywords = _arguments(schema, [materialize(value) for value in arguments])
                target = getattr(getattr(torch.ops.aten, name), overload or "default")
                result = graph.call_function(target, positional, keywords)
            results = [result]
        values.update(zip(outputs, results, strict=True))
    returned = [materialize(values[value]) for value in source.outputs()]
    graph.output(returned[0] if len(returned) == 1 else tuple(returned))
    graph.lint()
    converted = torch.fx.GraphModule(root, graph, class_name="ConstructedNeuralMethod")
    converted.meta["neural_ir_audit"] = audit
    return converted.eval()
