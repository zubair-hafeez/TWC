"""Independent generic chain evaluator using cached sparse polynomial intervals.

This baseline stores full-state forward messages and full-state backward maps
at candidate timestamps. Sparse CSR operators represent intervening history.
Unlike endpoint compilation it does not solve an edit-interaction system.
Its broad state storage restricts the measured suite to bounded graphs.
"""

import numpy as np
from scipy import sparse
from .oracles import validate, selected_ids


def group_matrix(src, dst, n, weight, deleted=()):
    keep = np.ones(len(src), bool)
    keep[np.asarray(deleted, np.int64)] = False
    u, v = src[keep], dst[keep]
    return sparse.csr_matrix(
        (np.full(2 * len(u), weight), (np.r_[u, v], np.r_[v, u])), shape=(n, n)
    )


def operator_interval(src, dst, stamp, n, depth, query_time, rate, begin, end):
    coef = [sparse.eye(n, format="csr")] + [
        sparse.csr_matrix((n, n)) for _ in range(depth)
    ]
    start = begin
    while start < end:
        stop = start + 1
        while stop < end and stamp[stop] == stamp[start]:
            stop += 1
        a = group_matrix(
            src[start:stop],
            dst[start:stop],
            n,
            np.exp(-rate * (query_time - stamp[start])),
        )
        for k in range(depth, 0, -1):
            coef[k] = coef[k] + a @ coef[k - 1]
        start = stop
    return coef


def apply_operator(coef, h):
    out = np.zeros_like(h)
    for k in range(len(h)):
        for degree in range(k + 1):
            out[k] += coef[degree] @ h[k - degree]
    return out


class SparseIntervalChain:
    def __init__(
        self, src, dst, stamp, base, depth, query_time, rate, candidates, nodes
    ):
        src, dst, stamp, base, candidates, nodes = validate(
            src, dst, stamp, base, depth, query_time, rate, candidates, nodes
        )
        self.candidates, self.nodes, self.depth = candidates, nodes, depth
        self.src, self.dst, self.stamp, self.n = src, dst, stamp, len(base)
        self.query_time, self.rate = query_time, rate
        self.times = np.unique(stamp[candidates])
        self.starts = np.searchsorted(stamp, self.times, "left")
        self.ends = np.searchsorted(stamp, self.times, "right")
        self.groups = [
            group_matrix(
                src[a:b], dst[a:b], self.n, np.exp(-rate * (query_time - stamp[a]))
            )
            for a, b in zip(self.starts, self.ends)
        ]
        self.intervals = []
        previous = 0
        for start, end in zip(self.starts, self.ends):
            self.intervals.append(
                operator_interval(
                    src, dst, stamp, self.n, depth, query_time, rate, previous, start
                )
            )
            previous = end
        self.intervals.append(
            operator_interval(
                src, dst, stamp, self.n, depth, query_time, rate, previous, len(src)
            )
        )
        h = np.zeros((depth + 1, self.n, base.shape[1]))
        h[0] = base
        self.prefix = []
        for i, group in enumerate(self.groups):
            h = apply_operator(self.intervals[i], h)
            self.prefix.append(h.copy())
            for k in range(depth, 0, -1):
                h[k] += group @ h[k - 1]
        h = apply_operator(self.intervals[-1], h)
        self.original = h[:, nodes].copy()
        selector = sparse.csr_matrix(
            (np.ones(len(nodes)), (np.arange(len(nodes)), nodes)),
            shape=(len(nodes), self.n),
        )
        tail = [selector @ p for p in self.intervals[-1]]
        self.suffix = [None] * len(self.groups)
        for i in range(len(self.groups) - 1, -1, -1):
            self.suffix[i] = tail
            after_group = [
                tail[k] + (tail[k - 1] @ self.groups[i] if k else 0)
                for k in range(depth + 1)
            ]
            tail = [
                sum(
                    (after_group[k - j] @ self.intervals[i][j] for j in range(k + 1)),
                    sparse.csr_matrix((len(nodes), self.n)),
                )
                for k in range(depth + 1)
            ]

    def evaluate(self, event_ids):
        selected = selected_ids(self.candidates, event_ids)
        if not len(selected):
            return self.original.copy()
        ids = self.candidates[selected]
        first, last = np.searchsorted(self.times, self.stamp[ids[[0, -1]]])
        h = self.prefix[first].copy()
        for i in range(first, last + 1):
            if i > first:
                h = apply_operator(self.intervals[i], h)
            a, b = self.starts[i], self.ends[i]
            chosen = ids[(ids >= a) & (ids < b)]
            # Rank-two removals act directly on full-state messages; no dense n^2 update.
            for k in range(self.depth, 0, -1):
                increment = self.groups[i] @ h[k - 1]
                w = np.exp(-self.rate * (self.query_time - self.times[i]))
                for event in chosen:
                    increment[self.src[event]] -= w * h[k - 1, self.dst[event]]
                    increment[self.dst[event]] -= w * h[k - 1, self.src[event]]
                h[k] += increment
        out = np.zeros_like(self.original)
        for k in range(self.depth + 1):
            for degree in range(k + 1):
                out[k] += self.suffix[last][degree] @ h[k - degree]
        return out

    def cache_bytes(self):
        def csr_bytes(a):
            return a.data.nbytes + a.indices.nbytes + a.indptr.nbytes

        matrices = self.groups + [a for interval in self.intervals for a in interval]
        matrices += [a for tail in self.suffix for a in tail]
        return (
            sum(csr_bytes(a) for a in matrices)
            + sum(a.nbytes for a in self.prefix)
            + self.original.nbytes
        )
