"""Adaptive evaluation: use singleton effects only when the subset is certified."""

from independent_effects import single_effects
from sparse_etwc import Compact, certified, compact_evaluate


class Adaptive(Compact):
    name = "adaptive"

    def __init__(self, *args):
        super().__init__(*args, prune=True)
        self.name = "adaptive"
        self.effects = single_effects(self.prefix, self.suffix, self.weights)
        self.used_ie = 0

    def batch(self, hypotheses, lengths, output):
        for row, length in enumerate(lengths):
            selected = hypotheses[row, :length]
            if certified(self.indptr, self.ent_j, selected, self.mark):
                output[row] = self.original
                for candidate in selected:
                    output[row] += self.effects[candidate]
                self.used_ie += 1
            else:
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
        result = super().storage()
        result["effect_bytes"] = self.effects.nbytes
        return result


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

    run_method(Adaptive, DATASETS, POLICIES, PARAMETERS)


if __name__ == "__main__":
    main()
