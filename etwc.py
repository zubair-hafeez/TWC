"""ETWC: compile all candidate transfers in one pass, then score deletions."""

import numpy as np
from common import BaseEvaluator, evaluate_circuit, group_boundaries, propagate


def build_transfers(src, dst, stamp, weights, n, depth, candidates):
    count = len(candidates)
    transfers = np.zeros((count, max(depth - 1, 0), 2, 2 * count))
    if depth < 2 or count == 0:
        return transfers
    state = np.zeros((depth - 1, n, 2 * count))
    start = np.searchsorted(stamp, stamp[candidates[0]], side="left")
    stop = np.searchsorted(stamp, stamp[candidates[-1]], side="right")
    boundaries = group_boundaries(stamp, start, stop)
    position = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        next_position = position
        while next_position < count and candidates[next_position] < stop:
            event = candidates[next_position]
            transfers[next_position, :, 0] = state[:, src[event]]
            transfers[next_position, :, 1] = state[:, dst[event]]
            next_position += 1
        propagate(state[:, :, : 2 * position], src, dst, start, stop, weights[start])
        # Inject after the group: equal-time candidates cannot affect each other.
        for source in range(position, next_position):
            event = candidates[source]
            state[0, src[event], 2 * source] = 1.0
            state[0, dst[event], 2 * source + 1] = 1.0
        position = next_position
    return transfers


def eager_batch(
    prefix, transfers, suffix, weights, original, hypotheses, lengths, output
):
    for row, length in enumerate(lengths):
        output[row] = evaluate_circuit(
            prefix, transfers, suffix, weights, hypotheses[row, :length], original
        )


class Eager(BaseEvaluator):
    name = "eager_etwc"

    def __init__(self, *args):
        super().__init__(*args)
        weights = np.exp(-self.rate * (self.query_time - self.stamp))
        self.transfers = build_transfers(
            self.src, self.dst, self.stamp, weights, self.n, self.depth, self.candidates
        )

    def batch(self, hypotheses, lengths, output):
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
        return {
            "shared_bytes": self.shared_bytes(),
            "transfer_bytes": self.transfers.nbytes,
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

    run_method(Eager, DATASETS, POLICIES, PARAMETERS)


if __name__ == "__main__":
    main()
