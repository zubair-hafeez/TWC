"""Sparse-ETWC: store only transfer coefficients that can reach the query."""

import numpy as np
from common import (
    BaseEvaluator,
    add_suffix_effects,
    encode,
    group_boundaries,
    propagate,
    propagate_reachability,
)


def suffix_hops(src, dst, stamp, n, depth, candidates, nodes):
    """Minimum remaining hops from each candidate endpoint to a query row."""
    beta = np.full((len(candidates), 2), depth + 1, dtype=np.int64)
    if len(candidates) == 0:
        return beta
    reachable = np.zeros((depth, n, 1), dtype=bool)
    reachable[0, nodes, 0] = True
    boundaries = group_boundaries(stamp)
    position = len(candidates) - 1
    for group in range(len(boundaries) - 2, -1, -1):
        start, stop = boundaries[group : group + 2]
        while position >= 0 and candidates[position] >= start:
            event = candidates[position]
            for side, vertex in enumerate((src[event], dst[event])):
                levels = np.flatnonzero(reachable[:, vertex, 0])
                if len(levels):
                    beta[position, side] = levels[0]
            position -= 1
        propagate_reachability(reachable, src, dst, start, stop)
    return beta


def compact_block(
    src,
    dst,
    stamp,
    weights,
    depth,
    candidates,
    beta,
    first_source,
    last_source,
    prune,
    state,
    reachable,
    target_ids,
    source_ids,
    target_sides,
    source_sides,
    low_degrees,
    high_degrees,
    values,
):
    """Compile one block of source impulses into contiguous coefficient arrays."""
    if depth < 2 or first_source >= last_source:
        return 0, 0, 0
    entry_count = 0
    value_count = 0
    first_event = np.searchsorted(stamp, stamp[candidates[first_source]], side="left")
    last_event = np.searchsorted(stamp, stamp[candidates[-1]], side="right")
    boundaries = group_boundaries(stamp, first_event, last_event)
    injected = first_source
    position = first_source
    scanned = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group_position = position
        while position < len(candidates) and candidates[position] < stop:
            event = candidates[position]
            for source in range(first_source, injected):
                column = 2 * (source - first_source)
                for side, vertex in enumerate((src[event], dst[event])):
                    high = depth - 2 - beta[position, 1 - side] if prune else depth - 2
                    for origin in range(2):
                        low = 0
                        if prune:
                            levels = np.flatnonzero(
                                reachable[:, vertex, column + origin]
                            )
                            low = levels[0] if len(levels) else depth
                        if low > high:
                            continue
                        target_ids[entry_count] = position
                        source_ids[entry_count] = source
                        target_sides[entry_count] = side
                        source_sides[entry_count] = origin
                        low_degrees[entry_count] = low
                        high_degrees[entry_count] = high
                        length = high - low + 1
                        values[value_count : value_count + length] = state[
                            low : high + 1, vertex, column + origin
                        ]
                        value_count += length
                        entry_count += 1
            position += 1
        if position == len(candidates):
            break
        width = 2 * (injected - first_source)
        propagate(state[:, :, :width], src, dst, start, stop, weights[start])
        if prune:
            propagate_reachability(reachable[:, :, :width], src, dst, start, stop)
        for source in range(max(group_position, injected), min(position, last_source)):
            column = 2 * (source - first_source)
            event = candidates[source]
            state[0, src[event], column] = 1.0
            state[0, dst[event], column + 1] = 1.0
            reachable[0, src[event], column] = True
            reachable[0, dst[event], column + 1] = True
            injected = source + 1
        scanned += stop - start
    state.fill(0.0)
    reachable.fill(False)
    return entry_count, value_count, scanned


def compact_evaluate(
    prefix,
    suffix,
    weights,
    original,
    indptr,
    source_ids,
    target_sides,
    source_sides,
    low_degrees,
    high_degrees,
    offsets,
    values,
    selected,
    positions,
):
    """Solve the edit recurrence using only the retained sparse coefficients."""
    depth, dimension = prefix.shape[1], prefix.shape[-1]
    positions[selected] = np.arange(len(selected))
    injections = np.zeros((len(selected), depth + 1, 2, dimension))
    for position, candidate in enumerate(selected):
        weight = weights[candidate]
        injections[position, 1:] = -weight * prefix[candidate, :, ::-1]
        for entry in range(indptr[candidate], indptr[candidate + 1]):
            earlier = positions[source_ids[entry]]
            if earlier < 0 or earlier >= position:
                continue
            side = 1 - target_sides[entry]
            origin = source_sides[entry]
            low = low_degrees[entry]
            for degree in range(low, high_degrees[entry] + 1):
                coefficient = -weight * values[offsets[entry] + degree - low]
                injections[position, degree + 2 :, side] += (
                    coefficient * injections[earlier, 1 : depth - degree, origin]
                )
    positions[selected] = -1
    return add_suffix_effects(original, suffix, selected, injections)


def compact_batch(
    prefix,
    suffix,
    weights,
    original,
    indptr,
    source_ids,
    target_sides,
    source_sides,
    low_degrees,
    high_degrees,
    offsets,
    values,
    positions,
    hypotheses,
    lengths,
    output,
):
    for row, length in enumerate(lengths):
        output[row] = compact_evaluate(
            prefix,
            suffix,
            weights,
            original,
            indptr,
            source_ids,
            target_sides,
            source_sides,
            low_degrees,
            high_degrees,
            offsets,
            values,
            hypotheses[row, :length],
            positions,
        )


def certified(indptr, source_ids, selected, marked):
    """True when no retained interaction connects two selected candidates."""
    marked[selected] = True
    independent = True
    for candidate in selected:
        if np.any(marked[source_ids[indptr[candidate] : indptr[candidate + 1]]]):
            independent = False
            break
    marked[selected] = False
    return independent


class Compact(BaseEvaluator):
    name = "sparse_compact"

    def __init__(self, *args, prune=True, block=None):
        super().__init__(*args)
        self.prune = prune
        self.name = "sparse_compact" if prune else "packed_compact"
        count = len(self.candidates)
        self.block = max(1, count if block is None else min(block, count))
        self.beta = suffix_hops(
            self.src,
            self.dst,
            self.stamp,
            self.n,
            self.depth,
            self.candidates,
            self.nodes,
        )
        weights = np.exp(-self.rate * (self.query_time - self.stamp))
        state = np.zeros((max(self.depth - 1, 1), self.n, 2 * self.block))
        reachable = np.zeros_like(state, dtype=bool)
        capacity = 4 * count * self.block
        buffers = [np.empty(capacity, dtype=np.int64) for _ in range(6)]
        values = np.empty(capacity * max(self.depth - 1, 1))
        self.workspace_bytes = state.nbytes + reachable.nbytes + values.nbytes
        self.workspace_bytes += sum(buffer.nbytes for buffer in buffers)
        blocks = []
        self.scanned = 0
        for first in range(0, count, self.block):
            entries, coefficients, scanned = compact_block(
                self.src,
                self.dst,
                self.stamp,
                weights,
                self.depth,
                self.candidates,
                self.beta,
                first,
                min(first + self.block, count),
                prune,
                state,
                reachable,
                *buffers,
                values,
            )
            self.scanned += scanned
            if entries:
                blocks.append(
                    [buffer[:entries].copy() for buffer in buffers]
                    + [values[:coefficients].copy()]
                )
        self.pack_entries(blocks, count)
        self.positions = np.full(count, -1, dtype=np.int64)
        self.mark = np.zeros(count, dtype=bool)

    def pack_entries(self, blocks, count):
        """Sort entries by target/source and place their coefficient slices together."""
        if blocks:
            target, source, side, origin, low, high = [
                np.concatenate([block[column] for block in blocks])
                for column in range(6)
            ]
            flat_values = np.concatenate([block[6] for block in blocks])
            lengths = high - low + 1
            original_offsets = np.r_[0, np.cumsum(lengths)[:-1]]
            order = np.lexsort((origin, side, source, target))
            target, source, side, origin = (
                target[order],
                source[order],
                side[order],
                origin[order],
            )
            low, high, lengths = low[order], high[order], lengths[order]
            self.ent_off = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
            gather = np.repeat(original_offsets[order] - self.ent_off, lengths)
            gather += np.arange(int(lengths.sum()))
            self.values = flat_values[gather]
        else:
            target = source = side = origin = low = high = np.empty(0, dtype=np.int64)
            self.ent_off = np.empty(0, dtype=np.int64)
            self.values = np.empty(0)
        self.indptr = np.searchsorted(target, np.arange(count + 1)).astype(np.int64)
        self.ent_j = source.astype(np.int32)
        self.ent_e = side.astype(np.int8)
        self.ent_a = origin.astype(np.int8)
        self.ent_lo = low.astype(np.int64)
        self.ent_hi = high.astype(np.int64)
        self.pairs = len(set(zip(target.tolist(), source.tolist())))

    def batch(self, hypotheses, lengths, output):
        compact_batch(
            self.prefix,
            self.suffix,
            self.weights,
            self.original,
            self.indptr,
            self.ent_j,
            self.ent_e,
            self.ent_a,
            self.ent_lo,
            self.ent_hi,
            self.ent_off,
            self.values,
            self.positions,
            hypotheses,
            lengths,
            output,
        )

    def independent_exact(self, event_ids):
        if not self.prune:
            raise NotImplementedError(
                "The unpruned packed method has no independence certificate"
            )
        hypotheses, lengths = encode(self.candidates, [tuple(event_ids)])
        return certified(
            self.indptr, self.ent_j, hypotheses[0, : lengths[0]], self.mark
        )

    def certificate_batch(self, hypotheses, lengths):
        return np.array(
            [
                certified(self.indptr, self.ent_j, row[:length], self.mark)
                for row, length in zip(hypotheses, lengths)
            ],
            dtype=bool,
        )

    def storage(self):
        indices = (
            self.indptr,
            self.ent_j,
            self.ent_e,
            self.ent_a,
            self.ent_lo,
            self.ent_hi,
            self.ent_off,
        )
        return {
            "shared_bytes": self.shared_bytes(),
            "coefficient_bytes": self.values.nbytes,
            "coefficients": len(self.values),
            "entries": len(self.ent_j),
            "index_bytes": sum(array.nbytes for array in indices),
            "beta_bytes": self.beta.nbytes,
            "padding_bytes": 0,
            "retained_pairs": self.pairs,
            "construction_workspace_bytes": self.workspace_bytes,
        }


# Edit these lists, then run this file. Multiple values run every combination.
DATASETS = ["mooc", "wikipedia", "lastfm"]
POLICIES = ["source", "dispersed", "shared_walk"]
PARAMETERS = {
    "queries": [4],  # Use [None] for every eligible development query.
    "candidates": [16],
    "max_deletions": [3],
    "depth": [3],
    "dimension": [64],
    "repeats": [3],
    "seed": [2026],
    "history_limit": [0],  # 0 uses the full history before each query.
}


def main():
    from experiment import run_method

    run_method(Compact, DATASETS, POLICIES, PARAMETERS)


if __name__ == "__main__":
    main()
