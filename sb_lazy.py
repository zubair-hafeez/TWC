"""SB-Lazy: prepare a source's transfers only when a hypothesis needs them."""
# This method prepares transfers for one source candidate only when needed.
# The first request for that source scans its later history and saves transfers
# to all later candidates. Other hypotheses reuse that source's saved results.
# It uses the same exact removal calculation as the eager method.

import numpy as np
from common import LazyEvaluator, evaluate_circuit, group_boundaries, propagate


def scan_source(src, dst, stamp, weights, depth, candidates, source, transfers, state):
    # Follow a unit signal from each endpoint of one earlier candidate.
    # Save both signals at every strictly later candidate's endpoints.
    # The caller supplies a reusable workspace and the full transfer table.
    # Return how many history events were actually propagated during this scan.
    # Two removed events cannot occur in a walk with fewer than two steps,
    # so no source-to-target interaction is needed at this depth.
    if depth < 2:
        return 0
    event = candidates[source]
    # Clear the shared workspace before starting this source, then place one
    # unit signal at each endpoint in its own column.
    state.fill(0.0)
    state[0, src[event], 0] = 1.0
    state[0, dst[event], 1] = 1.0
    # Start after all events tied with this source. Target candidates at the
    # same time are skipped for the same strict-time reason.
    first = np.searchsorted(stamp, stamp[event], side="right")
    last = np.searchsorted(stamp, stamp[candidates[-1]], side="right")
    position = np.searchsorted(stamp[candidates], stamp[event], side="right")
    boundaries = group_boundaries(stamp, first, last)
    scanned = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        # Save the current signals before applying the target's timestamp.
        # A transfer therefore includes only events strictly between source and target.
        while position < len(candidates) and candidates[position] < stop:
            target = candidates[position]
            transfers[position, :, 0, 2 * source : 2 * source + 2] = state[
                :, src[target]
            ]
            transfers[position, :, 1, 2 * source : 2 * source + 2] = state[
                :, dst[target]
            ]
            position += 1
        # All requested target rows have now been saved. Processing the final
        # timestamp would not change any transfer that this source needs.
        if position == len(candidates):
            break
        propagate(state, src, dst, start, stop, weights[start])
        scanned += stop - start
    # Leave the reusable workspace empty for its next use. Saved transfers
    # remain in the separate transfer array.
    state.fill(0.0)
    return scanned


class SourceLazy(LazyEvaluator):
    name = "source_lazy"

    def __init__(self, *args):
        # Set up the shared lazy storage, then record which sources have been built.
        # filled[source] becomes True after its scan, including when all saved
        # transfer values are zero. A zero result is still a completed calculation.
        super().__init__(*args)
        self.filled = np.zeros(len(self.candidates), dtype=bool)

    def batch(self, hypotheses, lengths, output):
        # For each removal set, ensure all possible earlier sources are cached.
        # The final selected candidate needs no outgoing transfer for this set,
        # because there is no later selected candidate for it to affect.
        # Once the needed source columns are ready, use the shared exact solver.
        for row, length in enumerate(lengths):
            selected = hypotheses[row, :length]
            for source in selected[:-1]:
                # A first request adds to both the scanned-event count and cache-miss count.
                # Later requests for this source skip the scan and reuse its stored columns.
                if not self.filled[source]:
                    self.scanned += scan_source(
                        self.src,
                        self.dst,
                        self.stamp,
                        self.event_weights,
                        self.depth,
                        self.candidates,
                        source,
                        self.transfers,
                        self.workspace,
                    )
                    self.filled[source] = True
                    self.misses += 1
            output[row] = evaluate_circuit(
                self.prefix,
                self.transfers,
                self.suffix,
                self.weights,
                selected,
                self.original,
            )

    def storage(self):
        # Report shared arrays, the allocated transfer table, and how many sources
        # have been computed. Lazy construction saves work when few sources are
        # requested, but this version still reserves the full dense transfer table.
        return {
            "shared_bytes": self.shared_bytes(),
            "transfer_bytes": self.transfers.nbytes,
            "sources_computed": int(self.filled.sum()),
        }


# Edit these lists, then run this file. Multiple values run every combination.
# Choose which stored datasets and candidate-selection policies to compare.
# source uses events touching the query source; dispersed spreads choices
# across its relevant history; shared_walk selects overlapping paths.
DATASETS = ["mooc", "wikipedia", "lastfm"]
POLICIES = ["source", "dispersed", "shared_walk"]
# Each value is a list of choices. The runner tries every combination
# across these lists, giving all methods the same workload settings.
PARAMETERS = {
    # Number of eligible query records to use from each dataset.
    "queries": [4],  # Use [None] for every eligible development query.
    # Number of earlier event instances that may be selected for removal.
    "candidates": [16],
    # Try all removal combinations from size one through this size.
    # More candidates or larger sets can greatly increase the number of cases.
    "max_deletions": [3],
    # Maximum number of time-ordered events allowed in a walk.
    "depth": [3],
    # Number of starting feature values stored for each node.
    "dimension": [64],
    # Number of fresh construction and batch runs used for timing.
    "repeats": [3],
    # Seed for repeatable feature values and workload choices.
    "seed": [2026],
    # A positive value requests a recent event window. Its start may move
    # earlier to avoid splitting events that share a timestamp.
    "history_limit": [0],  # 0 uses the full history before each query.
}


def main():
    # Import the shared runner only when this file is run as a program.
    # Pass this method class and the editable settings to it; the runner builds
    # the cases, measures the method, and prints the resulting rows.
    from experiment import run_method

    run_method(SourceLazy, DATASETS, POLICIES, PARAMETERS)


# Run the experiment only when this file is launched directly.
# Importing this module makes the method available without starting a run.
if __name__ == "__main__":
    main()
