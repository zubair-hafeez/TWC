"""Sparse-ETWC: store only transfer coefficients that can reach the query."""
# This method saves only transfer terms that could affect the requested nodes.
# A transfer tells how a change at an earlier candidate reaches a later one.
# The method follows possible paths as well as numeric values, so a very small
# number is not mistaken for an impossible path. It packs the retained values
# into flat arrays, then reuses them for many exact removal calculations.

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
    # For each candidate endpoint, find the fewest later steps needed to reach
    # any requested node. beta has one row per candidate and two endpoint columns.
    # The value depth + 1 means no route was found within the tracked walk lengths.
    # These minimum lengths help decide which middle-walk terms can still fit
    # inside the total walk-depth limit.
    beta = np.full((len(candidates), 2), depth + 1, dtype=np.int64)
    # With no removable events, return the empty two-column table immediately.
    # There is no candidate suffix distance to discover.
    if len(candidates) == 0:
        return beta
    reachable = np.zeros((depth, n, 1), dtype=bool)
    # A requested node reaches itself in zero steps. Scanning time backwards
    # extends these routes to earlier nodes while keeping each walk length separate.
    reachable[0, nodes, 0] = True
    boundaries = group_boundaries(stamp)
    position = len(candidates) - 1
    for group in range(len(boundaries) - 2, -1, -1):
        start, stop = boundaries[group : group + 2]
        # Read candidate endpoints before extending through their timestamp.
        # Only strictly later events may belong to a candidate's suffix.
        while position >= 0 and candidates[position] >= start:
            event = candidates[position]
            for side, vertex in enumerate((src[event], dst[event])):
                # The first True degree is the shortest known later route from this node.
                # Use possible paths, not numeric sums that might be tiny or cancel to zero.
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
    # Build transfers from a group of source candidates in one history scan.
    # Each source has two separate columns, one for each endpoint's unit signal.
    # state stores numeric walk sums; reachable records whether a path exists.
    # Write retained entries and their coefficient slices into caller-owned buffers,
    # and return the numbers of entries, values, and scanned events actually used.
    # A pair of removed events needs at least two walk steps.
    # An empty source block also has no interaction terms to build.
    if depth < 2 or first_source >= last_source:
        return 0, 0, 0
    entry_count = 0
    value_count = 0
    # Start at the first source's whole timestamp and stop after the last
    # candidate's timestamp. Including the first group lets us inject its signals
    # at exactly the point after all events in that group have been handled.
    first_event = np.searchsorted(stamp, stamp[candidates[first_source]], side="left")
    last_event = np.searchsorted(stamp, stamp[candidates[-1]], side="right")
    boundaries = group_boundaries(stamp, first_event, last_event)
    injected = first_source
    position = first_source
    scanned = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        # Remember where this timestamp's candidate range starts. position advances
        # while targets are read; injected tracks sources whose signals already exist.
        group_position = position
        while position < len(candidates) and candidates[position] < stop:
            event = candidates[position]
            # Only previously injected sources can affect this target. New sources
            # at the current timestamp are introduced later, preserving strict time.
            for source in range(first_source, injected):
                column = 2 * (source - first_source)
                for side, vertex in enumerate((src[event], dst[event])):
                    # Two removed events already take two steps. Reserve enough additional
                    # steps to reach the query after the target event, leaving high steps for
                    # the middle path. The target sends its input to the opposite endpoint,
                    # so the suffix distance uses 1 - side.
                    high = depth - 2 - beta[position, 1 - side] if prune else depth - 2
                    for origin in range(2):
                        low = 0
                        if prune:
                            # Find the shortest possible middle path from this source endpoint to
                            # the target input endpoint. Degrees below it cannot contribute.
                            levels = np.flatnonzero(
                                reachable[:, vertex, column + origin]
                            )
                            low = levels[0] if len(levels) else depth
                        # Skip this endpoint pair if the shortest middle path cannot fit within
                        # the allowed total length. This decision is based on paths and time,
                        # so a zero or underflowed numeric value does not change the decision.
                        if low > high:
                            continue
                        # Store the entry's target, source, endpoint sides, and degree bounds.
                        # Its values occupy one consecutive slice, from low through high inclusive.
                        # Some degrees inside that slice can still have zero values.
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
        # Every target has now been read. No later transfer is needed, so there
        # is no reason to propagate the last timestamp or inject its sources.
        if position == len(candidates):
            break
        # Propagate only source columns that have been introduced already.
        # When pruning is enabled, update Boolean reachability alongside the numbers
        # using the same timestamp groups.
        width = 2 * (injected - first_source)
        propagate(state[:, :, :width], src, dst, start, stop, weights[start])
        if prune:
            propagate_reachability(reachable[:, :, :width], src, dst, start, stop)
        # Introduce this group's sources after propagation. Signals start at degree
        # zero with value one, in separate columns for the two source endpoints.
        # Only sources belonging to this block are introduced in this pass.
        for source in range(max(group_position, injected), min(position, last_source)):
            column = 2 * (source - first_source)
            event = candidates[source]
            state[0, src[event], column] = 1.0
            state[0, dst[event], column + 1] = 1.0
            reachable[0, src[event], column] = True
            reachable[0, dst[event], column + 1] = True
            injected = source + 1
        scanned += stop - start
    # Erase temporary numeric and reachability states before the next block.
    # The values needed for evaluation have already been copied into the buffers.
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
    # Calculate one removal set from the packed transfer entries.
    # An injection is a signed change at a removed event's two endpoints.
    # First undo that event using its saved original prefix, then adjust this
    # change using earlier selected removals. Finally carry the changes through
    # the saved suffixes to the requested output nodes.
    depth, dimension = prefix.shape[1], prefix.shape[-1]
    # Map each selected candidate to its row in this hypothesis's injection array.
    # Unselected candidates keep -1, so saved links from them can be skipped.
    positions[selected] = np.arange(len(selected))
    injections = np.zeros((len(selected), depth + 1, 2, dimension))
    for position, candidate in enumerate(selected):
        weight = weights[candidate]
        # Begin with the single-deletion change for every positive degree.
        # Reverse endpoint order because an edge sends each endpoint's old value
        # to the other endpoint; the minus sign removes that update.
        injections[position, 1:] = -weight * prefix[candidate, :, ::-1]
        # indptr marks the start and end of this target's saved entry range.
        # Visit just those entries instead of scanning a full candidate-pair table.
        for entry in range(indptr[candidate], indptr[candidate + 1]):
            earlier = positions[source_ids[entry]]
            # Only earlier removals in this hypothesis contribute here.
            # Ignore unselected sources and any source that has not yet been processed.
            if earlier < 0 or earlier >= position:
                continue
            side = 1 - target_sides[entry]
            origin = source_sides[entry]
            low = low_degrees[entry]
            for degree in range(low, high_degrees[entry] + 1):
                # Locate this degree's packed transfer value and multiply by the negative
                # target weight. A source injection of length L gains degree middle steps
                # plus one target step, so it contributes at length L + degree + 1.
                coefficient = -weight * values[offsets[entry] + degree - low]
                # Update all source-injection lengths that still fit within depth at once.
                # The first destination length is degree + 2 because the shortest source
                # injection already contains one removed event.
                injections[position, degree + 2 :, side] += (
                    coefficient * injections[earlier, 1 : depth - degree, origin]
                )
    # Remove this hypothesis's temporary position map before reusing it.
    # Then apply the saved suffixes to turn endpoint changes into output changes.
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
    # Evaluate one encoded hypothesis per output row using the same packed data.
    # Only the first length entries of a hypothesis are selected candidates;
    # the remaining slots are padding. Reuse the positions workspace because
    # each call clears the positions it used before returning.
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
    # Check whether the selected removals have any retained interaction link.
    # Mark selected candidates, then inspect each selected target's incoming
    # source IDs. If none of those sources is selected, adding separate deletion
    # effects is guaranteed to be exact for this query and depth.
    marked[selected] = True
    independent = True
    for candidate in selected:
        if np.any(marked[source_ids[indptr[candidate] : indptr[candidate + 1]]]):
            independent = False
            break
    # Always clear the marks, including when a link was found and the loop
    # ended early. Otherwise one hypothesis could affect the next one's check.
    marked[selected] = False
    return independent


class Compact(BaseEvaluator):
    name = "sparse_compact"

    def __init__(self, *args, prune=True, block=None):
        # Prepare common prefixes, suffixes, and the original answer first.
        # Then build source transfers in blocks and pack the retained terms together.
        # prune controls whether unreachable or too-long terms are dropped.
        # block controls how many source candidates share one construction pass.
        super().__init__(*args)
        self.prune = prune
        self.name = "sparse_compact" if prune else "packed_compact"
        count = len(self.candidates)
        # By default, track all sources in one pass. A smaller block reduces the
        # number of signal columns in the temporary state but requires more passes.
        # Keep a positive block size even when there are no candidates.
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
        # Each source-target pair has four endpoint combinations: two source
        # endpoints times two target endpoints. Reserve enough metadata slots for
        # all pairs in one block, and enough values for every possible degree.
        capacity = 4 * count * self.block
        # The six metadata buffers hold target ID, source ID, target side,
        # source side, lowest degree, and highest degree, in that order.
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
            # Keep only the filled part of each temporary buffer. Copies are necessary
            # because the same buffers will be overwritten while building the next block.
            if entries:
                blocks.append(
                    [buffer[:entries].copy() for buffer in buffers]
                    + [values[:coefficients].copy()]
                )
        self.pack_entries(blocks, count)
        # Keep reusable lookup arrays for evaluation and independence checks.
        # A position of -1 means absent from the current removal set; mark starts False.
        self.positions = np.full(count, -1, dtype=np.int64)
        self.mark = np.zeros(count, dtype=bool)

    def pack_entries(self, blocks, count):
        """Sort entries by target/source and place their coefficient slices together."""
        # Join the blocks and arrange entries by target, source, and endpoint pair.
        # Each entry owns a consecutive range of walk degrees and a matching slice
        # of numeric values. Moving entries requires moving those slices too.
        # The resulting row pointers let evaluation jump straight to one target.
        if blocks:
            target, source, side, origin, low, high = [
                np.concatenate([block[column] for block in blocks])
                for column in range(6)
            ]
            flat_values = np.concatenate([block[6] for block in blocks])
            lengths = high - low + 1
            # Before sorting, remember where each entry's numeric slice starts.
            # A running sum of slice lengths gives the start of every following entry.
            original_offsets = np.r_[0, np.cumsum(lengths)[:-1]]
            # lexsort uses the last key first: group entries by target, then source,
            # then target side, and finally source side. This puts each target's entries
            # together so a pair of row pointers can find them.
            order = np.lexsort((origin, side, source, target))
            target, source, side, origin = (
                target[order],
                source[order],
                side[order],
                origin[order],
            )
            low, high, lengths = low[order], high[order], lengths[order]
            self.ent_off = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
            # Build an index for copying all coefficient slices into their new order.
            # For each slice, repeat the shift between its old and new start. Adding
            # the new flat positions then gives the old positions to read.
            gather = np.repeat(original_offsets[order] - self.ent_off, lengths)
            gather += np.arange(int(lengths.sum()))
            self.values = flat_values[gather]
        else:
            # Handle the valid case where no interaction entries survive.
            # Empty arrays and zero row ranges let the normal evaluator still work.
            target = source = side = origin = low = high = np.empty(0, dtype=np.int64)
            self.ent_off = np.empty(0, dtype=np.int64)
            self.values = np.empty(0)
        # For candidate i, entries run from indptr[i] up to indptr[i + 1].
        # Store source IDs and endpoint sides in smaller integer types to reduce
        # index storage; degree bounds and value offsets remain wider integers.
        self.indptr = np.searchsorted(target, np.arange(count + 1)).astype(np.int64)
        self.ent_j = source.astype(np.int32)
        self.ent_e = side.astype(np.int8)
        self.ent_a = origin.astype(np.int8)
        self.ent_lo = low.astype(np.int64)
        self.ent_hi = high.astype(np.int64)
        # Several endpoint entries can belong to one candidate pair.
        # Count each target-source pair once for the storage report.
        self.pairs = len(set(zip(target.tolist(), source.tolist())))

    def batch(self, hypotheses, lengths, output):
        # Supply the saved compact arrays to the batch helper.
        # The same packed history summary serves every hypothesis; only the
        # selected removal positions and temporary injections vary between rows.
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
        # Check whether one set of event IDs has a structural independence proof.
        # This proof is offered only for the pruned method. Convert event IDs to
        # candidate positions, then check whether any retained link joins the set.
        if not self.prune:
            raise NotImplementedError(
                "The unpruned packed method has no independence certificate"
            )
        hypotheses, lengths = encode(self.candidates, [tuple(event_ids)])
        return certified(
            self.indptr, self.ent_j, hypotheses[0, : lengths[0]], self.mark
        )

    def certificate_batch(self, hypotheses, lengths):
        # Return one True or False independence result per encoded hypothesis.
        # The temporary mark array is reused safely because certified clears each
        # set's marks before returning. This method checks structure without computing
        # any output feature values.
        return np.array(
            [
                certified(self.indptr, self.ent_j, row[:length], self.mark)
                for row, length in zip(hypotheses, lengths)
            ],
            dtype=bool,
        )

    def storage(self):
        # Report numeric coefficients, index arrays, shared data, and suffix lengths
        # separately. Entries count endpoint-pair degree ranges; retained_pairs counts
        # distinct target-source pairs, which can each own several entries.
        # The workspace count describes allocated construction buffers, while the
        # packed arrays describe data kept for later evaluation.
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
            # Coefficient slices are placed directly beside each other, without extra
            # unused array slots between slices. A retained slice may still contain zeros.
            "padding_bytes": 0,
            "retained_pairs": self.pairs,
            "construction_workspace_bytes": self.workspace_bytes,
        }


# Edit these lists, then run this file. Multiple values run every combination.
# Choose datasets and ways to select removable events. source uses events
# touching the query source, dispersed samples across the backward cone,
# and shared_walk follows overlapping paths ending at requested nodes.
DATASETS = ["mooc", "wikipedia", "lastfm"]
POLICIES = ["source", "dispersed", "shared_walk"]
# Each setting is a list of choices. The experiment runner tries every
# combination, so adding choices expands the set of experiment runs.
PARAMETERS = {
    # Number of eligible queries per dataset; None selects all of them.
    "queries": [4],  # Use [None] for every eligible development query.
    # Number of event instances available for removal at each query.
    "candidates": [16],
    # Try every removal set from size one up to this size.
    # The number of combinations can grow quickly as this value increases.
    "max_deletions": [3],
    # Maximum number of events in a strictly time-ordered walk.
    "depth": [3],
    # Number of starting feature values for each node.
    "dimension": [64],
    # Number of fresh builds and batch evaluations used for timing.
    "repeats": [3],
    # Seed that makes random features and workload choices repeatable.
    "seed": [2026],
    # A positive limit requests a recent event window. The start may move
    # backward so an equal-time group is kept whole.
    "history_limit": [0],  # 0 uses the full history before each query.
}


def main():
    # Import the common runner when this file is launched as a program.
    # Run the Compact method with the selected datasets, policies, and settings.
    # The runner prepares matching workloads and prints timing and error results.
    from experiment import run_method

    run_method(Compact, DATASETS, POLICIES, PARAMETERS)


# Start the experiment only when running this file directly.
# Importing Compact from another module does not run an experiment.
if __name__ == "__main__":
    main()
