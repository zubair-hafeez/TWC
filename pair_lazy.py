"""Pair-Lazy: prepare each requested source-to-target transfer on demand."""

import numpy as np
from common import LazyEvaluator, evaluate_circuit, group_boundaries, propagate


def scan_pair(
    src, dst, stamp, weights, depth, candidates, source, target, transfers, state
):
    earlier, later = candidates[source], candidates[target]
    if depth < 2 or stamp[earlier] >= stamp[later]:
        return 0
    state.fill(0.0)
    state[0, src[earlier], 0] = 1.0
    state[0, dst[earlier], 1] = 1.0
    first = np.searchsorted(stamp, stamp[earlier], side="right")
    last = np.searchsorted(stamp, stamp[later], side="left")
    boundaries = group_boundaries(stamp, first, last)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        propagate(state, src, dst, start, stop, weights[start])
    transfers[target, :, 0, 2 * source : 2 * source + 2] = state[:, src[later]]
    transfers[target, :, 1, 2 * source : 2 * source + 2] = state[:, dst[later]]
    state.fill(0.0)
    return last - first


class PairLazy(LazyEvaluator):
    name = "pair_lazy"

    def __init__(self, *args):
        super().__init__(*args)
        self.filled = np.zeros((len(self.candidates), len(self.candidates)), dtype=bool)

    def batch(self, hypotheses, lengths, output):
        for row, length in enumerate(lengths):
            selected = hypotheses[row, :length]
            for position, target in enumerate(selected):
                for source in selected[:position]:
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
        return {
            "shared_bytes": self.shared_bytes(),
            "transfer_bytes": self.transfers.nbytes,
            "pairs_computed": int(self.filled.sum()),
        }



# Run the experiment separately on each dataset below.
DATASETS = ["mooc", "wikipedia", "lastfm"]

# Each policy chooses a pool of historical events eligible for deletion.
# Source: Select the most recent events from intraction history of the source node of the query. Destination node is ignored.
# Dispersed: Select the random events from intraction history of all nodes of the query.
# Shared_walk: we pick two latest interation for the source node first, and then from each of that interation we pick 2 latest interations of destination node.
POLICIES = ["source", "dispersed", "shared_walk"]


PARAMETERS = {
    # Number of queries to evaluate per dataset/policy. we set to 4.
    "queries": [4],

    # Number of historical events selected as deletion candidates per query/policy.
    "candidates": [16],

    # Maximum number of deletions to consider per query/policy. we set to 3.
    # sets 1 : 16C1, sets 2 : 16C2, sets 3 : 16C3, total 16C1 + 16C2 + 16C3 = 696.
    "max_deletions": [3],

    # When going backwards, we have only have 3 hops from source to destination.
    "depth": [3],

    # Dimensions of the node base feature vector.
    "dimension": [64],

    # evaluate each case 3 times; report median timings.
    "repeats": [3],

    # Random seed for query selection, dispersed sampling, and base features.
    "seed": [2026],

    # 0 uses all events strictly before the query.
    "history_limit": [0],
}


from experiment import run_method


run_method(PairLazy, DATASETS, POLICIES, PARAMETERS)
