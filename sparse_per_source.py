"""Sparse-ETWC ablation: compile one source per pass instead of one full block."""

from sparse_etwc import Compact


class SparsePerSource(Compact):
    name = "sparse_per_source"

    def __init__(self, *args):
        super().__init__(*args, prune=True, block=1)
        self.name = "sparse_per_source"


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

    run_method(SparsePerSource, DATASETS, POLICIES, PARAMETERS)


if __name__ == "__main__":
    main()
