# We prepare calculation in advance and tests many removed events and it's effect on speed and memory. 

import numpy as np # for array operations

#
from common import BaseEvaluator, evaluate_circuit, group_boundaries, propagate
# BaseEvaluator is shared class for all evaluation methods. Check and organize the input data, calculate 
# original result before removing events, and preparre information for all evaluation methods. It has
# function to test the effect of one removal or many removals. 

# Evaluate Circuit; It first finds the correction needed to cancel each removed event. 
# If several events are removed, it considers how earlier removals change later ones. 
# Finally, it follows those changes through later events to the requested users or items 
# and returns the corrected result.
# 
#  # Group Boundaries; Find groups of events that have the same time.


def build_transfers(src, dst, stamp, weights, n, depth, candidates):
    # Track two starting signals for every candidate, one at each endpoint.
    # The state columns keep these signals separate while they travel through
    # later events. At each target candidate, save how much of every earlier
    # signal has reached its two endpoints at each allowed walk length.
    count = len(candidates)
    # The axes are target candidate, middle-walk degree, target endpoint,
    # and source endpoint. Two selected events already take two walk steps,
    # so the middle section needs degrees zero through depth - 2.
    transfers = np.zeros((count, max(depth - 1, 0), 2, 2 * count))
    # A walk shorter than two events cannot contain two removals.
    # With no candidates there are no transfers to prepare either.
    if depth < 2 or count == 0:
        return transfers
    state = np.zeros((depth - 1, n, 2 * count))
    # Only scan from the first candidate's timestamp through the last one's.
    # Events before this range are covered by prefixes; later ones by suffixes.
    start = np.searchsorted(stamp, stamp[candidates[0]], side="left")
    stop = np.searchsorted(stamp, stamp[candidates[-1]], side="right")
    boundaries = group_boundaries(stamp, start, stop)
    position = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        next_position = position
        # Read every target in the timestamp from the state before that timestamp.
        # The target event itself must not be part of the path leading to its input.
        while next_position < count and candidates[next_position] < stop:
            event = candidates[next_position]
            transfers[next_position, :, 0] = state[:, src[event]]
            transfers[next_position, :, 1] = state[:, dst[event]]
            next_position += 1
        # Move only signals from previously injected candidates through this group.
        # Signals for this timestamp are added afterwards, so equal-time candidates
        # cannot affect each other.
        propagate(state[:, :, : 2 * position], src, dst, start, stop, weights[start])
        # Inject after the group: equal-time candidates cannot affect each other.
        for source in range(position, next_position):
            event = candidates[source]
            # A unit value is a starting signal before taking any middle steps.
            # It lets the same transfer values later carry feature changes of any size.
            state[0, src[event], 2 * source] = 1.0
            state[0, dst[event], 2 * source + 1] = 1.0
        position = next_position
    return transfers


def eager_batch(
    prefix, transfers, suffix, weights, original, hypotheses, lengths, output
):
    # Read one encoded removal set per row and call the shared exact solver.
    # length tells where real candidate positions end and padding begins.
    # Write directly into the caller's output buffer, keeping row order.
    for row, length in enumerate(lengths):
        output[row] = evaluate_circuit(
            prefix, transfers, suffix, weights, hypotheses[row, :length], original
        )


class Eager(BaseEvaluator):
    name = "eager_etwc"

    def __init__(self, *args):
        # Build the common original answer, prefixes, and suffixes first.
        # Then calculate a decay weight for every history event and prepare the
        # whole transfer table. This pays the transfer cost during construction.
        super().__init__(*args)
        weights = np.exp(-self.rate * (self.query_time - self.stamp))
        self.transfers = build_transfers(
            self.src, self.dst, self.stamp, weights, self.n, self.depth, self.candidates
        )

    def batch(self, hypotheses, lengths, output):
        # Pass this method's saved arrays to the common batch helper.
        # The transfer table is already complete, so this stage only combines
        # saved values for each requested set of removals.
        eager_batch(
            self.prefix,
            self.transfers,
            self.suffix,
            self.weights,
            self.original,
            hypotheses,
            lengths,
            output,
        )

    def storage(self):
        # Report shared array bytes separately from the dense transfer table.
        # The transfer count grows with both source and target candidate counts,
        # including table locations whose values stay zero.
        return {
            "shared_bytes": self.shared_bytes(),
            "transfer_bytes": self.transfers.nbytes,
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

    run_method(Eager, DATASETS, POLICIES, PARAMETERS)


# Run the experiment only when this file is launched directly.
# Importing this module makes the method available without starting a run.
if __name__ == "__main__":
    main()
