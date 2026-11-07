"""Packed ETWC ablation: use compact storage without coefficient pruning."""

from sparse_etwc import Compact


class Packed(Compact):
    name = "packed_compact"

    def __init__(self, *args):
        super().__init__(*args, prune=False)


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

    run_method(Packed, DATASETS, POLICIES, PARAMETERS)


if __name__ == "__main__":
    main()
