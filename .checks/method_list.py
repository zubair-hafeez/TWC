"""Method imports used only by the hidden correctness tests."""

from adaptive import Adaptive
from etwc import Eager
from independent_effects import Independent
from packed_etwc import Packed
from pair_lazy import PairLazy
from sb_lazy import SourceLazy
from sparse_etwc import Compact
from sparse_per_source import SparsePerSource

METHODS = {
    "eager_etwc": Eager,
    "source_lazy": SourceLazy,
    "pair_lazy": PairLazy,
    "sparse_compact": Compact,
    "packed_compact": Packed,
    "sparse_per_source": SparsePerSource,
    "independent_effects": Independent,
    "adaptive": Adaptive,
}
