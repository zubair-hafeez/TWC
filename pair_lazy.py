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

    run_method(PairLazy, DATASETS, POLICIES, PARAMETERS)


if __name__ == "__main__":
    main()
