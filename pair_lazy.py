"""Pair-Lazy: prepare each requested source-to-target transfer on demand."""
# This method calculates a transfer only for a requested source-target pair.
# A source is an earlier removed event; a target is a later removed event.
# Only events strictly between the pair can carry the source's change to the
# target. Completed pairs are cached for use by later removal hypotheses.

import numpy as np
from common import LazyEvaluator, evaluate_circuit, group_boundaries, propagate


def scan_pair(
    src, dst, stamp, weights, depth, candidates, source, target, transfers, state
):
    # Build the transfer for one pair using two separate unit signals.
    # Start the signals at the earlier event's endpoints and carry them through
    # the intervening history. Read the resulting values at the later event's
    # endpoints and save them in this pair's part of the transfer table.
    earlier, later = candidates[source], candidates[target]
    # An interaction needs at least two events and strictly increasing time.
    # Equal-time candidates cannot form a two-event walk, even if they share a node.
    if depth < 2 or stamp[earlier] >= stamp[later]:
        return 0
    # Reuse the same small workspace for each missing pair. Its two columns
    # identify which endpoint of the source supplied a signal.
    state.fill(0.0)
    state[0, src[earlier], 0] = 1.0
    state[0, dst[earlier], 1] = 1.0
    # The right search skips the entire source timestamp; the left search
    # stops before the entire target timestamp. Neither boundary event belongs
    # to the transfer's middle section.
    first = np.searchsorted(stamp, stamp[earlier], side="right")
    last = np.searchsorted(stamp, stamp[later], side="left")
    boundaries = group_boundaries(stamp, first, last)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        propagate(state, src, dst, start, stop, weights[start])
    # Save values at both target endpoints for each source endpoint.
    # If there are no intervening events, degree zero still connects any
    # endpoint that the source and target share.
    transfers[target, :, 0, 2 * source : 2 * source + 2] = state[:, src[later]]
    transfers[target, :, 1, 2 * source : 2 * source + 2] = state[:, dst[later]]
    # Clear temporary signals after copying their values into the saved table.
    # Return the number of events in the scanned interval for work accounting.
    state.fill(0.0)
    return last - first


class PairLazy(LazyEvaluator):
    name = "pair_lazy"

    def __init__(self, *args):
        # Prepare common lazy arrays and one completion flag per target-source pair.
        # The flags distinguish a pair that has not been calculated from a pair
        # whose calculated transfer happens to be zero.
        super().__init__(*args)
        self.filled = np.zeros((len(self.candidates), len(self.candidates)), dtype=bool)

    def batch(self, hypotheses, lengths, output):
        # Process each removal set in candidate order. For every target, look at
        # all selected sources before it and build any missing pair transfer.
        # After these cached values are ready, combine the removals with the exact
        # shared calculation, which accounts for their effects on each other.
        for row, length in enumerate(lengths):
            selected = hypotheses[row, :length]
            for position, target in enumerate(selected):
                for source in selected[:position]:
                    # Calculate this pair once. Mark it filled even if the scan visits no
                    # events, so an empty or equal-time interval is not repeatedly calculated.
                    if not self.filled[target, source]:
                        self.scanned += scan_pair(
                            self.src,
                            self.dst,
                            self.stamp,
                            self.event_weights,
                            self.depth,
                            self.candidates,
                            source,
                            target,
                            self.transfers,
                            self.workspace,
                        )
                        self.filled[target, source] = True
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
        # Report shared bytes, the allocated dense transfer table, and the number
        # of completed pairs. This method delays calculations but still reserves
        # space for every pair in the table.
        return {
            "shared_bytes": self.shared_bytes(),
            "transfer_bytes": self.transfers.nbytes,
            "pairs_computed": int(self.filled.sum()),
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



from experiment import run_method

run_method(PairLazy, DATASETS, POLICIES, PARAMETERS)


