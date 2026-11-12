"""Exact deletion evaluators with independently organized message computation.

The lazy baseline intentionally shares the same low-rank product algebra as
the endpoint compiler. It is a strong cache policy comparison, not a claim
that an HMM implementation directly accepts temporal graph inputs.
"""

import numpy as np
from baselines.revision_circuit import (
    forward_index,
    backward_query_index,
    evaluate_circuit,
)


def validate(src, dst, stamp, base, depth, query_time, rate, candidates, nodes):
    src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    stamp, base = np.asarray(stamp, np.float64), np.asarray(base, np.float64)
    candidates = np.sort(np.asarray(candidates, np.int64))
    nodes = np.asarray(nodes, np.int64)
    if (
        depth < 1
        or rate < 0
        or len(src) != len(dst)
        or len(src) != len(stamp)
        or np.any(np.diff(stamp) < 0)
        or not np.all(stamp < query_time)
        or np.any(src == dst)
        or np.any(src < 0)
        or np.any(dst < 0)
        or np.any(src >= len(base))
        or np.any(dst >= len(base))
        or len(np.unique(candidates)) != len(candidates)
        or np.any(candidates < 0)
        or np.any(candidates >= len(src))
        or np.any(nodes < 0)
        or np.any(nodes >= len(base))
    ):
        raise ValueError("Invalid history, candidate set, or query")
    return src, dst, stamp, base, candidates, nodes


def selected_ids(candidates, event_ids):
    ids = np.sort(np.asarray(event_ids, np.int64))
    selected = np.searchsorted(candidates, ids)
    if (
        len(np.unique(ids)) != len(ids)
        or np.any(selected >= len(candidates))
        or not np.array_equal(candidates[selected], ids)
    ):
        raise ValueError("Deletion outside candidate set or duplicate deletion")
    return selected


def interval_transfer(src, dst, stamp, n, depth, query_time, rate, earlier, later):
    """Original interval product applied to two endpoint impulses, without n^2 arrays."""
    out = np.zeros((max(depth - 1, 0), 2, 2))
    if depth < 2 or stamp[earlier] >= stamp[later]:
        return out
    h = np.zeros((depth - 1, n, 2))
    h[0, src[earlier], 0] = 1.0
    h[0, dst[earlier], 1] = 1.0
    start = np.searchsorted(stamp, stamp[earlier], side="right")
    stop = np.searchsorted(stamp, stamp[later], side="left")
    while start < stop:
        end = start + 1
        while end < stop and stamp[end] == stamp[start]:
            end += 1
        w = np.exp(-rate * (query_time - stamp[start]))
        for k in range(depth - 2, 0, -1):
            for event in range(start, end):
                u, v = src[event], dst[event]
                for col in range(2):
                    h[k, u, col] += w * h[k - 1, v, col]
                    h[k, v, col] += w * h[k - 1, u, col]
        start = end
    out[:, 0] = h[:, src[later]]
    out[:, 1] = h[:, dst[later]]
    return out


class LazyForwardBackward:
    """Cache P/F once and requested inter-edit interval messages on demand.

    Each first request includes interval construction. Repeated hypotheses
    reuse cached messages. Empty/tied-time intervals preserve strict time.
    """

    def __init__(
        self, src, dst, stamp, base, depth, query_time, rate, candidates, nodes
    ):
        src, dst, stamp, base, candidates, nodes = validate(
            src, dst, stamp, base, depth, query_time, rate, candidates, nodes
        )
        self.src, self.dst, self.stamp = src, dst, stamp
        self.n, self.depth, self.query_time, self.rate = (
            len(base),
            depth,
            query_time,
            rate,
        )
        self.candidates, self.nodes = candidates, nodes
        h, self.prefix = forward_index(
            src, dst, stamp, base, depth, query_time, rate, candidates
        )
        self.original = h[:, nodes].copy()
        self.suffix = backward_query_index(
            src, dst, stamp, len(base), depth, query_time, rate, candidates, nodes
        )
        self.weights = np.exp(-rate * (query_time - stamp[candidates]))
        self.cache = {}
        self.cache_misses = 0

    def evaluate(self, event_ids):
        selected = selected_ids(self.candidates, event_ids)
        b = len(selected)
        transfer = np.zeros((b, max(self.depth - 1, 0), 2, 2 * b))
        for ii, i in enumerate(selected):
            for jj, j in enumerate(selected[:ii]):
                key = (int(i), int(j))
                if key not in self.cache:
                    self.cache[key] = interval_transfer(
                        self.src,
                        self.dst,
                        self.stamp,
                        self.n,
                        self.depth,
                        self.query_time,
                        self.rate,
                        self.candidates[j],
                        self.candidates[i],
                    )
                    self.cache_misses += 1
                transfer[ii, :, :, 2 * jj : 2 * jj + 2] = self.cache[key]
        return evaluate_circuit(
            self.prefix[selected],
            transfer,
            self.suffix[selected],
            self.weights[selected],
            np.arange(b),
            self.original,
        )

    def cache_bytes(self):
        # Numeric payload; Python dict overhead and resident event arrays are separate.
        return sum(
            x.nbytes for x in (self.prefix, self.suffix, self.weights, self.original)
        ) + sum(x.nbytes for x in self.cache.values())

    def resident_event_bytes(self):
        return self.src.nbytes + self.dst.nbytes + self.stamp.nbytes


def minimum_suffix_hops(src, dst, stamp, n, depth, candidates, nodes):
    """Integer strict-time reachability, independent of projections and weights."""
    distance = np.full(n, depth + 1, np.int64)
    distance[nodes] = 0
    out = np.full((len(candidates), 2), depth + 1, np.int64)
    pos, end = len(candidates) - 1, len(src)
    while end > 0:
        start = end - 1
        while start > 0 and stamp[start - 1] == stamp[end - 1]:
            start -= 1
        while pos >= 0 and candidates[pos] >= start:
            event = candidates[pos]
            out[pos, 0], out[pos, 1] = distance[src[event]], distance[dst[event]]
            pos -= 1
        left, right = distance[src[start:end]].copy(), distance[dst[start:end]].copy()
        for p in range(end - start):
            event = start + p
            distance[src[event]] = min(distance[src[event]], right[p] + 1)
            distance[dst[event]] = min(distance[dst[event]], left[p] + 1)
        end = start
    return out


def one_source_sparse_transfers(
    src, dst, stamp, n, depth, query_time, rate, candidates, suffix_hops, source_index
):
    """O(nK+qK) working storage for a single source; no dense q-by-q allocation."""
    q = len(candidates)
    blocks = np.zeros((q, max(depth - 1, 0), 2, 2))
    keep = np.zeros(q, np.bool_)
    if depth < 2:
        return blocks, keep
    h = np.zeros((depth - 1, n, 2))
    distance = np.full((n, 2), depth + 1, np.int64)
    event = candidates[source_index]
    h[0, src[event], 0], h[0, dst[event], 1] = 1.0, 1.0
    distance[src[event], 0], distance[dst[event], 1] = 0, 0
    start = np.searchsorted(stamp, stamp[event], side="right")
    pos = source_index + 1
    while pos < q and stamp[candidates[pos]] == stamp[event]:
        pos += 1
    stop = np.searchsorted(stamp, stamp[candidates[-1]], side="right")
    while start < stop and pos < q:
        end = start + 1
        while end < len(src) and stamp[end] == stamp[start]:
            end += 1
        while pos < q and candidates[pos] < end:
            i = candidates[pos]
            for side in range(2):
                vertex = src[i] if side == 0 else dst[i]
                max_degree = depth - 2 - suffix_hops[pos, 1 - side]
                for origin_side in range(2):
                    if distance[vertex, origin_side] <= max_degree:
                        keep[pos] = True
                        for degree in range(max_degree + 1):
                            blocks[pos, degree, side, origin_side] = h[
                                degree, vertex, origin_side
                            ]
            pos += 1
        w = np.exp(-rate * (query_time - stamp[start]))
        for k in range(depth - 2, 0, -1):
            for e in range(start, end):
                u, v = src[e], dst[e]
                for col in range(2):
                    h[k, u, col] += w * h[k - 1, v, col]
                    h[k, v, col] += w * h[k - 1, u, col]
        left, right = distance[src[start:end]].copy(), distance[dst[start:end]].copy()
        for offset in range(end - start):
            e = start + offset
            for col in range(2):
                distance[src[e], col] = min(
                    distance[src[e], col], right[offset, col] + 1
                )
                distance[dst[e], col] = min(
                    distance[dst[e], col], left[offset, col] + 1
                )
        start = end
    return blocks, keep


def evaluate_sparse(
    prefix, suffix, weights, original, indptr, indices, values, selected
):
    q, depth, _, d = prefix.shape
    positions = np.full(q, -1, np.int64)
    for p, i in enumerate(selected):
        positions[i] = p
    injection = np.zeros((len(selected), depth + 1, 2, d))
    for pos, i in enumerate(selected):
        for k in range(1, depth + 1):
            for side in range(2):
                for col in range(d):
                    injection[pos, k, side, col] = (
                        -weights[i] * prefix[i, k - 1, 1 - side, col]
                    )
        for edge in range(indptr[i], indptr[i + 1]):
            earlier = positions[indices[edge]]
            if earlier < 0:
                continue
            for k in range(2, depth + 1):
                for level in range(1, k):
                    degree = k - 1 - level
                    for side in range(2):
                        for origin in range(2):
                            coeff = -weights[i] * values[edge, degree, 1 - side, origin]
                            if coeff != 0.0:
                                for col in range(d):
                                    injection[pos, k, side, col] += (
                                        coeff * injection[earlier, level, origin, col]
                                    )
    out = original.copy()
    for pos, i in enumerate(selected):
        for k in range(1, depth + 1):
            for level in range(1, k + 1):
                for row in range(out.shape[1]):
                    for side in range(2):
                        coeff = suffix[i, k - level, row, side]
                        if coeff != 0.0:
                            for col in range(d):
                                out[k, row, col] += (
                                    coeff * injection[pos, level, side, col]
                                )
    return out


class SparseInteractionOracle:
    """Structurally pruned endpoint compiler with an exact independence certificate."""

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
        self.suffix_hops = minimum_suffix_hops(
            src, dst, stamp, len(base), depth, candidates, nodes
        )
        incoming = [[] for _ in candidates]
        for j in range(len(candidates)):
            blocks, keep = one_source_sparse_transfers(
                src,
                dst,
                stamp,
                len(base),
                depth,
                query_time,
                rate,
                candidates,
                self.suffix_hops,
                j,
            )
            for i in np.flatnonzero(keep):
                incoming[i].append((j, blocks[i].copy()))
        self.indptr = np.zeros(len(candidates) + 1, np.int64)
        indices, values = [], []
        for i, row in enumerate(incoming):
            for j, block in row:
                indices.append(j)
                values.append(block)
            self.indptr[i + 1] = len(indices)
        self.indices = np.asarray(indices, np.int64)
        self.values = (
            np.asarray(values, np.float64).reshape(-1, max(depth - 1, 0), 2, 2)
            if depth > 1
            else np.zeros((0, 0, 2, 2))
        )

    def evaluate(self, event_ids):
        return evaluate_sparse(
            self.prefix,
            self.suffix,
            self.weights,
            self.original,
            self.indptr,
            self.indices,
            self.values,
            selected_ids(self.candidates, event_ids),
        )

    def independent_exact(self, event_ids):
        selected = selected_ids(self.candidates, event_ids)
        chosen = set(selected.tolist())
        return not any(
            j in chosen
            for i in selected
            for j in self.indices[self.indptr[i] : self.indptr[i + 1]]
        )

    def interaction_edges(self):
        return len(self.indices)

    def cache_bytes(self):
        return sum(
            x.nbytes
            for x in (
                self.prefix,
                self.suffix,
                self.weights,
                self.original,
                self.indptr,
                self.indices,
                self.values,
                self.suffix_hops,
            )
        )
