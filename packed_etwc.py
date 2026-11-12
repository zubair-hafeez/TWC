"""Packed ETWC ablation: use compact storage without coefficient pruning."""
# This small wrapper selects the compact storage method without pruning.
# Pruning means dropping transfer terms that cannot contribute to the query.
# Keeping those terms lets experiments compare the effect of storage layout
# separately from the effect of removing unnecessary terms.

from sparse_etwc import Compact


# Inherit the batch calculation and storage reporting from Compact.
# Only the construction option and public method name differ here.
class Packed(Compact):
    name = "packed_compact"

    def __init__(self, *args):
        # Use Compact's construction and evaluation code with pruning disabled.
        # This still packs coefficient slices together, but it does not use
        # reachability to discard impossible endpoint and degree combinations.
        super().__init__(*args, prune=False)


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

    run_method(Packed, DATASETS, POLICIES, PARAMETERS)


# Run the experiment only when this file is launched directly.
# Importing this module makes the method available without starting a run.
if __name__ == "__main__":
    main()
