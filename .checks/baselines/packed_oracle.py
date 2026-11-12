"""Unpruned triangular-packing ablation for structural sparsity claims."""

import numpy as np
from .oracles import SparseInteractionOracle, validate
from .lazy_source import source_messages
from baselines.revision_circuit import forward_index, backward_query_index


class PackedEndpointOracle(SparseInteractionOracle):
    def __init__(
        self, src, dst, stamp, base, depth, query_time, rate, candidates, nodes
    ):
        src, dst, stamp, base, candidates, nodes = validate(
            src, dst, stamp, base, depth, query_time, rate, candidates, nodes
        )
        self.candidates, self.nodes, self.depth = candidates, nodes, depth
        h, self.prefix = forward_index(
            src, dst, stamp, base, depth, query_time, rate, candidates
        )
        self.original = h[:, nodes].copy()
        self.suffix = backward_query_index(
            src, dst, stamp, len(base), depth, query_time, rate, candidates, nodes
        )
        self.weights = np.exp(-rate * (query_time - stamp[candidates]))
        self.suffix_hops = np.empty((0, 2), np.int64)
        incoming = [[] for _ in candidates]
        if depth > 1:
            for j in range(len(candidates)):
                blocks = source_messages(
                    src, dst, stamp, len(base), depth, query_time, rate, candidates, j
                )
                for i in range(j + 1, len(candidates)):
                    if stamp[candidates[j]] < stamp[candidates[i]]:
                        incoming[i].append((j, blocks[i].copy()))
        self.indptr = np.zeros(len(candidates) + 1, np.int64)
        indices = []
        values = []
        for i, row in enumerate(incoming):
            for j, block in row:
                indices.append(j)
                values.append(block)
            self.indptr[i + 1] = len(indices)
        self.indices = np.asarray(indices, np.int64)
        self.values = (
            np.asarray(values, np.float64).reshape(-1, depth - 1, 2, 2)
            if depth > 1
            else np.zeros((0, 0, 2, 2))
        )
