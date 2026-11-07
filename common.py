"""Shared array operations for the temporal walk comparison methods.

Events at the same timestamp always read the state from before that timestamp.
The methods share these operations so that their main difference is how they
prepare and store transfers between candidate deletions.
"""

import numpy as np


def validate(src, dst, stamp, base, depth, query_time, rate, candidates, nodes):
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    stamp = np.asarray(stamp, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    candidates = np.sort(np.asarray(candidates, dtype=np.int64))
    nodes = np.asarray(nodes, dtype=np.int64)
    if base.ndim != 2 or any(a.ndim != 1 for a in (src, dst, stamp, candidates, nodes)):
        raise ValueError(
            "History and index arrays must be vectors; base must be a matrix"
        )
    if not isinstance(depth, (int, np.integer)) or depth < 1:
        raise ValueError("Depth must be a positive integer")
    if not np.isfinite(query_time) or not np.isfinite(rate) or rate < 0:
        raise ValueError(
            "Query time and decay rate must be finite; rate must be nonnegative"
        )
    if len(src) != len(dst) or len(src) != len(stamp):
        raise ValueError("History arrays must have the same length")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(stamp)):
        raise ValueError("States and timestamps must be finite")
    if np.any(np.diff(stamp) < 0) or np.any(stamp >= query_time):
        raise ValueError("History must be sorted and strictly before the query")
    if (
        np.any(src == dst)
        or np.any(src < 0)
        or np.any(dst < 0)
        or np.any(src >= len(base))
        or np.any(dst >= len(base))
    ):
        raise ValueError("History contains an invalid endpoint or a self edge")
    if (
        len(np.unique(candidates)) != len(candidates)
        or np.any(candidates < 0)
        or np.any(candidates >= len(src))
        or np.any(nodes < 0)
        or np.any(nodes >= len(base))
    ):
        raise ValueError("Invalid candidate events or query nodes")
    return src, dst, stamp, base, candidates, nodes


def selected_ids(candidates, event_ids):
    """Convert event IDs to positions in the sorted candidate array."""
    ids = np.sort(np.asarray(event_ids, dtype=np.int64))
    selected = np.searchsorted(candidates, ids)
    if (
        ids.ndim != 1
        or len(np.unique(ids)) != len(ids)
        or np.any(selected >= len(candidates))
        or not np.array_equal(candidates[selected], ids)
    ):
        raise ValueError("Deletion outside candidate set or duplicate deletion")
    return selected


def encode(candidates, hypotheses):
    """Encode a list of event-ID tuples once, outside the timed evaluation."""
    candidates = np.sort(np.asarray(candidates, dtype=np.int64))
    width = max([len(ids) for ids in hypotheses] + [1])
    encoded = np.zeros((len(hypotheses), width), dtype=np.int64)
    lengths = np.zeros(len(hypotheses), dtype=np.int64)
    for row, ids in enumerate(hypotheses):
        positions = selected_ids(candidates, ids)
        encoded[row, : len(positions)] = positions
        lengths[row] = len(positions)
    return encoded, lengths


def group_boundaries(stamp, start=0, stop=None):
    """Offsets delimiting equal-timestamp groups in [start, stop)."""
    if stop is None:
        stop = len(stamp)
    if start >= stop:
        return np.array([start], dtype=np.int64)
    changes = np.flatnonzero(np.diff(stamp[start:stop])) + start + 1
    return np.r_[start, changes, stop]


def propagate(state, src, dst, start, stop, weight):
    """Add one timestamp's walks using a snapshot of all lower degrees.

    NumPy handles the feature columns. add.at preserves duplicate edges and
    repeated endpoints within a timestamp group.
    """
    if state.shape[0] <= 1 or state.shape[-1] == 0:
        return
    if stop == start + 1:
        u, v = src[start], dst[start]
        to_u = weight * state[:-1, v]
        to_v = weight * state[:-1, u]
        state[1:, u] += to_u
        state[1:, v] += to_v
    else:
        u, v = src[start:stop], dst[start:stop]
        to_u = weight * state[:-1, v]
        to_v = weight * state[:-1, u]
        np.add.at(state[1:], (slice(None), u), to_u)
        np.add.at(state[1:], (slice(None), v), to_v)


def propagate_reachability(reachable, src, dst, start, stop):
    """Boolean counterpart of propagate, independent of numerical weights."""
    if reachable.shape[0] <= 1 or reachable.shape[-1] == 0:
        return
    u, v = src[start:stop], dst[start:stop]
    to_u = reachable[:-1, v].copy()
    to_v = reachable[:-1, u].copy()
    np.logical_or.at(reachable[1:], (slice(None), u), to_u)
    np.logical_or.at(reachable[1:], (slice(None), v), to_v)


def forward_index(src, dst, stamp, base, depth, query_time, rate, candidates):
    state = np.zeros((depth + 1, len(base), base.shape[1]))
    state[0] = base
    prefix = np.empty((len(candidates), depth, 2, base.shape[1]))
    boundaries = group_boundaries(stamp)
    position = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        while position < len(candidates) and candidates[position] < stop:
            event = candidates[position]
            prefix[position, :, 0] = state[:depth, src[event]]
            prefix[position, :, 1] = state[:depth, dst[event]]
            position += 1
        weight = np.exp(-rate * (query_time - stamp[start]))
        propagate(state, src, dst, start, stop, weight)
    return state, prefix


def backward_query_index(
    src, dst, stamp, n, depth, query_time, rate, candidates, nodes
):
    state = np.zeros((depth, n, len(nodes)))
    state[0, nodes, np.arange(len(nodes))] = 1.0
    suffix = np.zeros((len(candidates), depth, len(nodes), 2))
    boundaries = group_boundaries(stamp)
    position = len(candidates) - 1
    for group in range(len(boundaries) - 2, -1, -1):
        start, stop = boundaries[group : group + 2]
        while position >= 0 and candidates[position] >= start:
            event = candidates[position]
            suffix[position, :, :, 0] = state[:, src[event]]
            suffix[position, :, :, 1] = state[:, dst[event]]
            position -= 1
        weight = np.exp(-rate * (query_time - stamp[start]))
        propagate(state, src, dst, start, stop, weight)
    return suffix


def add_suffix_effects(original, suffix, selected, injections):
    """Propagate the selected edit injections to the requested output rows."""
    output = original.copy()
    depth = len(original) - 1
    for position, candidate in enumerate(selected):
        for degree in range(1, depth + 1):
            for level in range(1, degree + 1):
                output[degree] += (
                    suffix[candidate, degree - level] @ injections[position, level]
                )
    return output


def evaluate_circuit(prefix, transfers, suffix, weights, selected, original):
    """Solve the causal edit recurrence using dense transfer blocks."""
    depth, dimension = prefix.shape[1], prefix.shape[-1]
    injections = np.zeros((len(selected), depth + 1, 2, dimension))
    for position, candidate in enumerate(selected):
        for degree in range(1, depth + 1):
            value = prefix[candidate, degree - 1].copy()
            for earlier in range(position):
                source = selected[earlier]
                for level in range(1, degree):
                    transfer = transfers[
                        candidate, degree - 1 - level, :, 2 * source : 2 * source + 2
                    ]
                    value += transfer @ injections[earlier, level]
            injections[position, degree] = -weights[candidate] * value[::-1]
    return add_suffix_effects(original, suffix, selected, injections)


class BaseEvaluator:
    """Common input preparation and the public single/batch evaluation API."""

    exact = True

    def __init__(
        self, src, dst, stamp, base, depth, query_time, rate, candidates, nodes
    ):
        src, dst, stamp, base, candidates, nodes = validate(
            src, dst, stamp, base, depth, query_time, rate, candidates, nodes
        )
        self.src, self.dst, self.stamp = src, dst, stamp
        self.n, self.d, self.depth = len(base), base.shape[1], depth
        self.query_time, self.rate = query_time, rate
        self.candidates, self.nodes = candidates, nodes
        state, self.prefix = forward_index(
            src, dst, stamp, base, depth, query_time, rate, candidates
        )
        self.original = state[:, nodes].copy()
        self.suffix = backward_query_index(
            src, dst, stamp, self.n, depth, query_time, rate, candidates, nodes
        )
        self.weights = np.exp(-rate * (query_time - stamp[candidates]))

    def out_buffer(self, count):
        return np.empty((count,) + self.original.shape)

    def evaluate(self, event_ids):
        hypotheses, lengths = encode(self.candidates, [tuple(event_ids)])
        output = self.out_buffer(1)
        self.batch(hypotheses, lengths, output)
        return output[0]

    def batch(self, hypotheses, lengths, output):
        raise NotImplementedError("Implement batch in the method class")

    def shared_bytes(self):
        return sum(
            array.nbytes
            for array in (self.prefix, self.suffix, self.weights, self.original)
        )


class LazyEvaluator(BaseEvaluator):
    """Storage shared by the source-lazy and pair-lazy comparisons."""

    def __init__(self, *args):
        super().__init__(*args)
        count = len(self.candidates)
        self.event_weights = np.exp(-self.rate * (self.query_time - self.stamp))
        self.transfers = np.zeros((count, max(self.depth - 1, 0), 2, 2 * count))
        self.workspace = np.zeros((max(self.depth - 1, 1), self.n, 2))
        self.misses = 0
        self.scanned = 0
