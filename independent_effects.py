"""Independent Effects: sum singleton changes, ignoring deletion interactions."""

import numpy as np
from common import BaseEvaluator


def single_effects(prefix, suffix, weights):
    count, depth, _, dimension = prefix.shape
    effects = np.zeros((count, depth + 1, suffix.shape[2], dimension))
    for candidate in range(count):
        for degree in range(1, depth + 1):
            for level in range(1, degree + 1):
                effects[candidate, degree] += (
                    -weights[candidate] * suffix[candidate, degree - level]
                ) @ prefix[candidate, level - 1, ::-1]
    return effects


class Independent(BaseEvaluator):
    name = "independent_effects"
    exact = False

    def __init__(self, *args):
        super().__init__(*args)
        self.effects = single_effects(self.prefix, self.suffix, self.weights)

    def batch(self, hypotheses, lengths, output):
        for row, length in enumerate(lengths):
            output[row] = self.original
            for candidate in hypotheses[row, :length]:
                output[row] += self.effects[candidate]

    def storage(self):
        return {
            "shared_bytes": self.shared_bytes(),
            "effect_bytes": self.effects.nbytes,
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

    run_method(Independent, DATASETS, POLICIES, PARAMETERS)


if __name__ == "__main__":
    main()
