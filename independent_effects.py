"""Independent Effects: sum singleton changes, ignoring deletion interactions."""
# This method saves the effect of deleting each candidate on its own.
# For a larger removal set it adds these separate effects to the original
# answer. This is quick, but can be approximate when two removals occur on
# the same relevant walk and therefore change one another's effects.

import numpy as np
from common import BaseEvaluator


def single_effects(prefix, suffix, weights):
    # Calculate the full output change from each single-event deletion.
    # prefix supplies the state before that event, and suffix carries its
    # negative update along later walks to the requested nodes.
    # Keep a separate [degree, query node, feature] result for every candidate.
    count, depth, _, dimension = prefix.shape
    effects = np.zeros((count, depth + 1, suffix.shape[2], dimension))
    for candidate in range(count):
        for degree in range(1, depth + 1):
            # Split the requested walk length around the removed event.
            # The earlier portion has level - 1 steps, the event itself adds one,
            # and the later portion has degree - level steps.
            for level in range(1, degree + 1):
                # Negate the event weight to undo its contribution. Reversing the two
                # prefix endpoints matches the edge's update: each endpoint receives the
                # other endpoint's value. Matrix multiplication then sends both changes
                # to the requested nodes using the saved suffix weights.
                effects[candidate, degree] += (
                    -weights[candidate] * suffix[candidate, degree - level]
                ) @ prefix[candidate, level - 1, ::-1]
    return effects


class Independent(BaseEvaluator):
    name = "independent_effects"
    # Tell the experiment runner that several summed singleton effects may
    # not equal a full replay. The runner still measures and reports the error.
    exact = False

    def __init__(self, *args):
        # Prepare shared history summaries, then compute each single deletion once.
        # The saved effects are reused for every later removal set.
        super().__init__(*args)
        self.effects = single_effects(self.prefix, self.suffix, self.weights)

    def batch(self, hypotheses, lengths, output):
        # Copy the unchanged answer into each output slot, then add the saved
        # single-deletion effects named by that row. This does not correct an effect
        # for other deletions in the same set, which is why the method is approximate.
        for row, length in enumerate(lengths):
            output[row] = self.original
            for candidate in hypotheses[row, :length]:
                output[row] += self.effects[candidate]

    def storage(self):
        # Count shared arrays and the table of single-deletion effects separately.
        # This table stores output changes directly, rather than candidate-to-candidate
        # transfers, so its size also depends on the number of requested nodes.
        return {
            "shared_bytes": self.shared_bytes(),
            "effect_bytes": self.effects.nbytes,
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

    run_method(Independent, DATASETS, POLICIES, PARAMETERS)


# Run the experiment only when this file is launched directly.
# Importing this module makes the method available without starting a run.
if __name__ == "__main__":
    main()
