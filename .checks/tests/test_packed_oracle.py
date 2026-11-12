import itertools
import numpy as np
import pytest
from baselines.packed_oracle import PackedEndpointOracle
from baselines.oracles import SparseInteractionOracle
from baselines.walks import enumerated_operator


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_packing_versus_pruning_and_enumeration(depth):
    src = np.array([0, 1, 3, 0, 2, 1])
    dst = np.array([1, 2, 4, 1, 0, 2])
    stamp = np.array([0, 1, 1, 2, 3, 4], float)
    base = np.random.default_rng(26181).normal(size=(5, 3))
    args = (src, dst, stamp, base, depth, 5.0, 0.1, np.arange(6), np.array([0, 2]))
    packed = PackedEndpointOracle(*args)
    pruned = SparseInteractionOracle(*args)
    assert packed.interaction_edges() >= pruned.interaction_edges()
    for b in range(7):
        for ids in itertools.combinations(range(6), b):
            events = [
                (int(u), int(v), float(t))
                for i, (u, v, t) in enumerate(zip(src, dst, stamp))
                if i not in ids
            ]
            target = (enumerated_operator(events, 5, depth, 5.0, 0.1) @ base)[:, [0, 2]]
            np.testing.assert_allclose(
                packed.evaluate(ids), target, atol=1e-10, rtol=1e-11
            )
            np.testing.assert_allclose(
                pruned.evaluate(ids), target, atol=1e-10, rtol=1e-11
            )
