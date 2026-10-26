"""Constructed affine/activation circuits for rules and stable visibility.

These circuits encode Boolean clauses and comparison-network connectivity in
fixed buffers. Selection uses hard threshold neurons to choose a comparator's
winner and bounded ReLU neurons to carry its score and original index. Static
``index_select`` wires only connect those neurons; no score-dependent indexing,
sorting, reduction, or engine operation decides the result.

Selection inputs are float32, finite, and in ``[0, bound]``. Call
``validate_input`` at a host boundary for external inputs; the live game supplies
internal activations under this invariant. The selected scores retain their
original float32 values, including small scores beside the far sentinel.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class BooleanCircuit(nn.Module):
    """Compile DNF rules into two fixed affine/hard-threshold layers.

    ``rules[output][clause][literal]`` uses signed one-based input indices:
    ``1`` means input zero is true, and ``-1`` means it is false. An empty
    clause is true; an output with no clauses is false. Inputs and outputs are
    floating zero/one activations with an arbitrary leading batch shape.
    """

    def __init__(self, input_count, rules):
        super().__init__()
        self.input_count = _positive_integer(input_count, "input_count")
        if not isinstance(rules, (tuple, list)) or not rules:
            raise ValueError("rules must contain at least one output")
        self.output_count = len(rules)
        clauses, clause_index, outputs = [], {}, []
        for output in rules:
            if not isinstance(output, (tuple, list)):
                raise ValueError("each rule output must be a sequence of clauses")
            connections = []
            for clause in output:
                if not isinstance(clause, (tuple, list)):
                    raise ValueError("each clause must be a sequence of literals")
                variables = set()
                for literal in clause:
                    if (isinstance(literal, bool) or not isinstance(literal, int)
                            or not 1 <= abs(literal) <= self.input_count):
                        raise ValueError("clause literals must be signed one-based input indices")
                    if abs(literal) in variables:
                        raise ValueError("a clause cannot repeat or contradict an input")
                    variables.add(abs(literal))
                key = tuple(sorted(clause))
                if key not in clause_index:
                    clause_index[key] = len(clauses)
                    clauses.append(key)
                connections.append(clause_index[key])
            outputs.append(connections)
        weight = torch.zeros(len(clauses), self.input_count)
        bias = torch.full((len(clauses),), 0.5)
        for row, clause in enumerate(clauses):
            for literal in clause:
                weight[row, abs(literal) - 1] = 1 if literal > 0 else -1
                bias[row] -= int(literal > 0)
        output_weight = torch.zeros(self.output_count, len(clauses))
        for row, connections in enumerate(outputs):
            output_weight[row, connections] = 1
        self.register_buffer("clause_weight", weight)
        self.register_buffer("clause_bias", bias)
        self.register_buffer("output_weight", output_weight)
        self.register_buffer("output_bias", torch.full((self.output_count,), -0.5))

    def forward(self, inputs):
        clauses = (F.linear(inputs, self.clause_weight, self.clause_bias) > 0).float()
        return (F.linear(clauses, self.output_weight, self.output_bias) > 0).float()

    def validate_input(self, inputs):
        """Validate external Boolean activations outside network execution."""
        if inputs.dtype != torch.float32 or inputs.ndim == 0 or inputs.shape[-1] != self.input_count:
            raise ValueError("Boolean inputs require float32 and the constructed input count")
        if not bool(((inputs == 0) | (inputs == 1)).all()):
            raise ValueError("Boolean inputs must be zero or one")

    def construction_spec(self):
        return {"type": "BooleanCircuit", "input_count": self.input_count,
                "output_count": self.output_count, "clauses": len(self.clause_bias),
                "affine_layers": 2, "activation": "hard_threshold",
                "purpose": "DNF Boolean rule decisions", "literal_indices": "signed_one_based",
                "weights": {"clause_weight": list(self.clause_weight.shape),
                            "output_weight": list(self.output_weight.shape)}}


class _ComparatorStage(nn.Module):
    """One parallel layer, with constructor-computed sparse affine wiring."""

    def __init__(self, inputs, pairs, outputs, stable_compare):
        super().__init__()
        self.input_count, self.output_count = len(inputs), len(outputs)
        self.comparator_count = len(pairs)
        self.stable_compare = stable_compare
        position = {wire: index for index, wire in enumerate(inputs)}
        self.register_buffer("pair_wires", torch.tensor([position[wire] for pair in pairs for wire in pair]))
        if stable_compare:
            routing = {wire: (pair, high) for pair, (a, b) in enumerate(pairs)
                       for wire, high in ((a, 0), (b, 1))}
            self.register_buffer("route", torch.tensor([routing[wire][0] for wire in outputs]))
            self.register_buffer("candidate_wires", torch.tensor(
                [position[wire] for output in outputs for wire in pairs[routing[output][0]]]))
            high = torch.tensor([routing[wire][1] for wire in outputs], dtype=torch.float32)
            self.register_buffer("gate_weight", 1 - 2 * high)
            self.register_buffer("gate_bias", high)
        else:
            assert outputs == [a for a, _ in pairs], "tournament outputs must be the lower pair wires"

    def forward(self, values, compare_weight, winner_weight, winner_bias, carry_weight):
        pairs = values.index_select(-2, self.pair_wires).reshape(values.shape[:-2] + (self.comparator_count, 4))
        if self.stable_compare:
            comparisons = (F.linear(pairs, compare_weight) > 0).float()
            gate = (F.linear(comparisons, winner_weight, winner_bias) > 0).float().squeeze(-1)
            off_left = gate.index_select(-1, self.route) * self.gate_weight + self.gate_bias
            candidates = values.index_select(-2, self.candidate_wires).reshape(values.shape[:-2] + (self.output_count, 2, 2))
        else:
            off_left = (pairs[..., 0] + pairs[..., 2] * compare_weight[0] > 0).float()
            candidates = pairs.reshape(values.shape[:-2] + (self.output_count, 2, 2))
        off = torch.stack((off_left, 1 - off_left), dim=-1)
        # This sparse affine map is identity(value) + weight * off. Only the
        # inactive branch subtracts its bound. The ReLU carries are summed by
        # fixed unit connections, without a dynamic winner gather.
        carried = F.relu(candidates + off[..., None] * carry_weight)
        return carried[..., 0, :] + carried[..., 1, :]

    def construction_spec(self):
        return {"type": "ComparatorStage", "input_wires": self.input_count,
                "output_wires": self.output_count, "comparators": self.comparator_count,
                "stable_index_comparison": self.stable_compare,
                "connectivity": "static pairs and output routes",
                "carry_affine": "identity(value) + bounded_gate_weight * off",
                "activations": ["hard_threshold", "relu"]}


class _SelectionCircuit(nn.Module):
    def __init__(self, candidate_count, output_count, bound, stable_compare):
        super().__init__()
        self.input_count = _positive_integer(candidate_count, "candidate_count")
        self.output_count = _positive_integer(output_count, "output_count")
        if self.output_count > self.input_count:
            raise ValueError("output_count cannot exceed candidate_count")
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound <= 0:
            raise ValueError("bound must be a finite positive number")
        self.bound = float(bound)
        self.padded_count = 1 << (self.input_count - 1).bit_length()
        if self.padded_count > 2 ** 24 or self.bound > torch.finfo(torch.float32).max:
            raise ValueError("score and index bounds must be representable in float32")
        # Inputs to every comparator are [left_score, left_index, right_score,
        # right_index]. A tie uses the original index, not the current wire.
        comparison = ([[1., 0., -1., 0.], [-1., 0., 1., 0.], [0., 1., 0., -1.]]
                      if stable_compare else [-1.])
        self.register_buffer("compare_weight", torch.tensor(comparison))
        self.register_buffer("winner_weight", torch.tensor([[2., -2., 1.]]) if stable_compare else None)
        self.register_buffer("winner_bias", torch.tensor([-0.5]) if stable_compare else None)
        # Off gates are constructed before this affine transform. Adding a
        # large positive bias and cancelling it later would quantize small
        # selected scores near the sentinel; the active branch subtracts zero.
        self.register_buffer("carry_weight", torch.tensor([-self.bound, -float(self.padded_count)]))
        self.register_buffer("initial_index", torch.arange(self.padded_count).float())
        self.stages = nn.ModuleList()

    def forward(self, scores):
        scores = F.pad(scores, (0, self.padded_count - self.input_count), value=self.bound)
        values = torch.stack((scores, self.initial_index.expand_as(scores)), dim=-1)
        for stage in self.stages:
            values = stage(values, self.compare_weight, self.winner_weight,
                           self.winner_bias, self.carry_weight)
        return values[..., 0], values[..., 1].long()

    def validate_input(self, scores):
        """Validate external scores on the host, outside network execution."""
        if scores.dtype != torch.float32 or scores.ndim == 0 or scores.shape[-1] != self.input_count:
            raise ValueError("selection scores require float32 and the constructed candidate count")
        if not bool(torch.isfinite(scores).all()) or not bool(((scores >= 0) & (scores <= self.bound)).all()):
            raise ValueError("selection scores must be finite and within [0, bound]")

    def construction_spec(self):
        return {"type": type(self).__name__, "input_count": self.input_count,
                "output_count": self.output_count, "stages": len(self.stages),
                "comparators": sum(stage.comparator_count for stage in self.stages),
                "score_bound": self.bound, "index_bound": self.padded_count,
                "stable_ties": "original_index", "activations": ["hard_threshold", "relu"],
                "purpose": "bounded score selection", "connectivity": "static_index_select",
                "weights": {name: list(buffer.shape) for name, buffer in self.named_buffers(recurse=False)
                            if name.endswith("weight")}}


class NearestCircuit(_SelectionCircuit):
    """Stable nearest ``k`` candidates through a pruned bitonic network.

    Outputs are increasing scores and their original indices, with shape
    ``scores.shape[:-1] + (k,)``. Connectivity is constructed once; forward
    evaluation consists only of fixed wiring, affine transforms, and neurons.
    """

    def __init__(self, candidate_count, k, bound=1e6):
        super().__init__(candidate_count, k, bound, stable_compare=True)
        network = []
        size = 2
        while size <= self.padded_count:
            stride = size // 2
            while stride:
                pairs = []
                for a in range(self.padded_count):
                    b = a ^ stride
                    if b > a:
                        pairs.append((a, b) if (a & size) == 0 else (b, a))
                network.append(pairs)
                stride //= 2
            size *= 2
        outputs = list(range(self.output_count))
        stages = []
        for pairs in reversed(network):
            required = set(outputs)
            pairs = [(a, b) for a, b in pairs if a in required or b in required]
            inputs = sorted({wire for pair in pairs for wire in pair})
            stages.append(_ComparatorStage(inputs, pairs, outputs, stable_compare=True))
            outputs = inputs
        assert outputs == list(range(self.padded_count)), "bitonic inputs must account for every candidate"
        self.stages.extend(reversed(stages))


class MinimumCircuit(_SelectionCircuit):
    """Stable scalar minimum through a logarithmic-depth tournament."""

    def __init__(self, candidate_count, bound=1e6):
        super().__init__(candidate_count, 1, bound, stable_compare=False)
        count = self.padded_count
        while count > 1:
            pairs = [(a, a + 1) for a in range(0, count, 2)]
            self.stages.append(_ComparatorStage(list(range(count)), pairs,
                                                 list(range(0, count, 2)), stable_compare=False))
            count //= 2

    def forward(self, scores):
        depth, index = super().forward(scores)
        return depth.squeeze(-1), index.squeeze(-1)
