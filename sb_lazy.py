"""SB-Lazy: prepare a source's transfers only when a hypothesis needs them."""

import numpy as np
from common import LazyEvaluator, evaluate_circuit, group_boundaries, propagate


def scan_source(src, dst, stamp, weights, depth, candidates, source, transfers, state):
    if depth < 2:
        return 0
    event = candidates[source]
    state.fill(0.0)
    state[0, src[event], 0] = 1.0
    state[0, dst[event], 1] = 1.0
    first = np.searchsorted(stamp, stamp[event], side="right")
    last = np.searchsorted(stamp, stamp[candidates[-1]], side="right")
    position = np.searchsorted(stamp[candidates], stamp[event], side="right")
    boundaries = group_boundaries(stamp, first, last)
    scanned = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        while position < len(candidates) and candidates[position] < stop:
            target = candidates[position]
            transfers[position, :, 0, 2 * source : 2 * source + 2] = state[
                :, src[target]
            ]
            transfers[position, :, 1, 2 * source : 2 * source + 2] = state[
                :, dst[target]
            ]
            position += 1
        if position == len(candidates):
            break
        propagate(state, src, dst, start, stop, weights[start])
        scanned += stop - start
    state.fill(0.0)
    return scanned


class SourceLazy(LazyEvaluator):
    name = "source_lazy"

    def __init__(self, *args):
        super().__init__(*args)
        self.filled = np.zeros(len(self.candidates), dtype=bool)

    def batch(self, hypotheses, lengths, output):
        for row, length in enumerate(lengths):
            selected = hypotheses[row, :length]
            for source in selected[:-1]:
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
        return {
            "shared_bytes": self.shared_bytes(),
            "transfer_bytes": self.transfers.nbytes,
            "sources_computed": int(self.filled.sum()),
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

    run_method(SourceLazy, DATASETS, POLICIES, PARAMETERS)


if __name__ == "__main__":
    main()
