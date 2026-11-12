"""Adaptive evaluation: use singleton effects only when the subset is certified."""
# This method chooses between two calculations for each removal set.
# If the saved graph structure proves the selected removals cannot interact
# on a relevant walk, it adds their single-deletion effects. Otherwise it
# uses the full sparse interaction calculation. Both chosen paths are exact.

from independent_effects import single_effects
from sparse_etwc import Compact, certified, compact_evaluate


class Adaptive(Compact):
    name = "adaptive"

    def __init__(self, *args):
        # Build the pruned transfer structure and save all single-deletion effects.
        # The structure is used to check whether a removal set can use the cheaper
        # sum. used_ie counts the sets that have taken that path.
        super().__init__(*args, prune=True)
        self.name = "adaptive"
        self.effects = single_effects(self.prefix, self.suffix, self.weights)
        self.used_ie = 0

    def batch(self, hypotheses, lengths, output):
        # Check each encoded removal set against the saved interaction links.
        # A successful certificate means there is no retained link between two
        # selected candidates, so their separate effects can be added exactly.
        # When the check fails, use the sparse calculation to include interactions.
        for row, length in enumerate(lengths):
            selected = hypotheses[row, :length]
            # This decision uses possible paths in the graph, not a floating-point
            # error threshold. Tiny weights cannot make a possible interaction disappear.
            if certified(self.indptr, self.ent_j, selected, self.mark):
                output[row] = self.original
                for candidate in selected:
                    output[row] += self.effects[candidate]
                self.used_ie += 1
            else:
                # A failed certificate does not by itself prove the separate sum is wrong.
                # It means independence is not guaranteed, so compute the interactions
                # instead of relying on the shortcut.
                output[row] = compact_evaluate(
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
                    selected,
                    self.positions,
                )

    def storage(self):
        # Extend Compact's storage report with the extra single-effect table.
        # The adaptive method keeps both representations so it can choose a path
        # for each hypothesis without rebuilding either one.
        result = super().storage()
        result["effect_bytes"] = self.effects.nbytes
        return result


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

    run_method(Adaptive, DATASETS, POLICIES, PARAMETERS)


# Run the experiment only when this file is launched directly.
# Importing this module makes the method available without starting a run.
if __name__ == "__main__":
    main()
