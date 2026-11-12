"""Compile a fixed candidate set for many exact deletion counterfactuals.

This is a fixed-weight inference oracle, not parameter unlearning. The
candidate events, query time, node rows and original history are immutable.
Transfers between changed events include their interactions exactly.
"""

import numpy as np


def forward_index(src, dst, stamp, base, depth, query_time, rate, candidates):
    """Save only candidate endpoint prefixes, not prefixes of every event."""
    n, d = base.shape
    h = np.zeros((depth + 1, n, d))
    h[0] = base
    prefix = np.empty((len(candidates), depth, 2, d))
    pos = 0
    start = 0
    while start < len(src):
        end = start + 1
        while end < len(src) and stamp[end] == stamp[start]:
            end += 1
        while pos < len(candidates) and candidates[pos] < end:
            i = candidates[pos]
            prefix[pos, :, 0] = h[:depth, src[i]]
            prefix[pos, :, 1] = h[:depth, dst[i]]
            pos += 1
        w = np.exp(-rate * (query_time - stamp[start]))
        for k in range(depth, 0, -1):
            for i in range(start, end):
                u, v = src[i], dst[i]
                for col in range(d):
                    h[k, u, col] += w * h[k - 1, v, col]
                    h[k, v, col] += w * h[k - 1, u, col]
        start = end
    return h, prefix


def candidate_transfers(src, dst, stamp, n, depth, query_time, rate, candidates):
    """Original-history transfer from earlier candidate endpoints to later ones.

    Shape (target edit, walk degree, target endpoint, source edit/endpoint).
    Equal timestamp candidates have zero transfers, even at degree zero.
    """
    q = len(candidates)
    transfers = np.zeros((q, max(depth - 1, 0), 2, 2 * q))
    if depth < 2 or not q:
        return transfers
    h = np.zeros((depth - 1, n, 2 * q))
    pos, active = 0, 0
    start = np.searchsorted(stamp, stamp[candidates[0]], side="left")
    while start < len(src) and start <= candidates[-1]:
        end = start + 1
        while end < len(src) and stamp[end] == stamp[start]:
            end += 1
        next_pos = pos
        while next_pos < q and candidates[next_pos] < end:
            i = candidates[next_pos]
            transfers[next_pos, :, 0] = h[:, src[i]]
            transfers[next_pos, :, 1] = h[:, dst[i]]
            next_pos += 1
        w = np.exp(-rate * (query_time - stamp[start]))
        for k in range(depth - 2, 0, -1):
            for i in range(start, end):
                u, v = src[i], dst[i]
                for col in range(active):
                    h[k, u, col] += w * h[k - 1, v, col]
                    h[k, v, col] += w * h[k - 1, u, col]
        # A candidate timestamp belongs to neither its prefix nor its suffix.
        for j in range(pos, next_pos):
            i = candidates[j]
            h[0, src[i], 2 * j] = 1.0
            h[0, dst[i], 2 * j + 1] = 1.0
        pos, active, start = next_pos, 2 * next_pos, end
    return transfers


def backward_query_index(
    src, dst, stamp, n, depth, query_time, rate, candidates, nodes
):
    q, s = len(candidates), len(nodes)
    # h[k,v,row] = suffix matrix coefficient [requested node(row),v].
    h = np.zeros((depth, n, s))
    for row, node in enumerate(nodes):
        h[0, node, row] = 1.0
    suffix = np.zeros((q, depth, s, 2))
    pos, end = q - 1, len(src)
    while end > 0:
        start = end - 1
        while start > 0 and stamp[start - 1] == stamp[end - 1]:
            start -= 1
        while pos >= 0 and candidates[pos] >= start:
            i = candidates[pos]
            suffix[pos, :, :, 0] = h[:, src[i]]
            suffix[pos, :, :, 1] = h[:, dst[i]]
            pos -= 1
        w = np.exp(-rate * (query_time - stamp[start]))
        for k in range(depth - 1, 0, -1):
            for i in range(start, end):
                u, v = src[i], dst[i]
                for row in range(s):
                    h[k, u, row] += w * h[k - 1, v, row]
                    h[k, v, row] += w * h[k - 1, u, row]
        end = start
    return suffix


def evaluate_circuit(prefix, transfers, suffix, weights, selected, original):
    """Solve the finite, causal edit interaction equations."""
    depth, d = prefix.shape[1], prefix.shape[-1]
    injection = np.zeros((len(selected), depth + 1, 2, d))
    for pos, i in enumerate(selected):
        for k in range(1, depth + 1):
            for side in range(2):
                other = 1 - side
                for col in range(d):
                    value = prefix[i, k - 1, other, col]
                    # Earlier deletions change the input of this deletion.
                    for earlier in range(pos):
                        j = selected[earlier]
                        for level in range(1, k):
                            degree = k - 1 - level
                            for b in range(2):
                                value += (
                                    transfers[i, degree, other, 2 * j + b]
                                    * injection[earlier, level, b, col]
                                )
                    injection[pos, k, side, col] = -weights[i] * value
    out = original.copy()
    for pos, i in enumerate(selected):
        for k in range(1, depth + 1):
            for level in range(1, k + 1):
                degree = k - level
                for row in range(out.shape[1]):
                    for side in range(2):
                        coefficient = suffix[i, degree, row, side]
                        if coefficient != 0.0:
                            for col in range(d):
                                out[k, row, col] += (
                                    coefficient * injection[pos, level, side, col]
                                )
    return out


class DeletionCircuit:
    def __init__(
        self, src, dst, stamp, base, depth, query_time, rate, candidates, nodes
    ):
        self.src = src
        self.dst = dst
        self.stamp = stamp
        self.base = base
        self.depth = depth
        self.query_time = query_time
        self.rate = rate
        self.candidates = candidates
        self.nodes = nodes

        self.src = np.asarray(self.src, dtype=np.int64)
        self.dst = np.asarray(self.dst, dtype=np.int64)
        self.stamp = np.asarray(self.stamp, dtype=np.float64)
        self.base = np.asarray(self.base, dtype=np.float64)
        self.candidates = np.sort(np.asarray(self.candidates, dtype=np.int64))
        self.nodes = np.asarray(self.nodes, dtype=np.int64)
        if self.depth < 1 or self.rate < 0:
            raise ValueError("Invalid depth or decay rate")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Duplicate candidate event id")
        if np.any(self.candidates < 0) or np.any(self.candidates >= len(self.src)):
            raise ValueError("Invalid candidate event id")
        if np.any(self.nodes < 0) or np.any(self.nodes >= len(self.base)):
            raise ValueError("Invalid query node")
        if (
            len(self.src) != len(self.dst)
            or len(self.src) != len(self.stamp)
            or np.any(np.diff(self.stamp) < 0)
            or not np.all(self.stamp < self.query_time)
            or np.any(self.src == self.dst)
        ):
            raise ValueError(
                "History must be chronological, strictly before query, without self edges"
            )
        h, self.prefix = forward_index(
            self.src,
            self.dst,
            self.stamp,
            self.base,
            self.depth,
            self.query_time,
            self.rate,
            self.candidates,
        )
        self.original = h[:, self.nodes].copy()
        self.transfers = candidate_transfers(
            self.src,
            self.dst,
            self.stamp,
            len(self.base),
            self.depth,
            self.query_time,
            self.rate,
            self.candidates,
        )
        self.suffix = backward_query_index(
            self.src,
            self.dst,
            self.stamp,
            len(self.base),
            self.depth,
            self.query_time,
            self.rate,
            self.candidates,
            self.nodes,
        )
        self.weights = np.exp(
            -self.rate * (self.query_time - self.stamp[self.candidates])
        )

    def cache_bytes(self):
        return sum(
            a.nbytes
            for a in (
                self.original,
                self.prefix,
                self.transfers,
                self.suffix,
                self.weights,
            )
        )

    def evaluate(self, event_ids):
        ids = np.sort(np.asarray(event_ids, dtype=np.int64))
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate removal")
        selected = np.searchsorted(self.candidates, ids)
        if np.any(selected >= len(self.candidates)) or not np.array_equal(
            self.candidates[selected], ids
        ):
            raise ValueError("Removal outside compiled candidate set")
        return evaluate_circuit(
            self.prefix,
            self.transfers,
            self.suffix,
            self.weights,
            selected,
            self.original,
        )
