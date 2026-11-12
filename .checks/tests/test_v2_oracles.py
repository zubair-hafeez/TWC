import itertools
import numpy as np
import pytest
from baselines.oracles import SparseInteractionOracle, LazyForwardBackward
from baselines.interval_chain import SparseIntervalChain
from baselines.revision_circuit import DeletionCircuit
from baselines.independent_effects import IndependentDeletionEffects
from baselines.walks import enumerated_operator


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_all_subsets_against_walk_enumeration(seed, depth):
    rng = np.random.default_rng(12000 + seed)
    n, m = 5, 9
    src = rng.integers(n, size=m)
    dst = (src + rng.integers(1, n, size=m)) % n
    stamp = np.sort(rng.integers(0, 6, size=m)).astype(
        float
    )  # ties and repeated instances
    base = rng.normal(size=(n, 3))
    nodes = np.array([0, 2, 4])
    candidates = np.arange(0, m, 2)
    args = (src, dst, stamp, base, depth, 7.0, 0.13, candidates, nodes)
    methods = [
        SparseInteractionOracle(*args),
        LazyForwardBackward(*args),
        SparseIntervalChain(*args),
        DeletionCircuit(*args),
    ]
    independent = IndependentDeletionEffects(*args)
    for b in range(len(candidates) + 1):
        for ids in itertools.combinations(candidates, b):
            events = [
                (int(u), int(v), float(t))
                for i, (u, v, t) in enumerate(zip(src, dst, stamp))
                if i not in ids
            ]
            expected = (enumerated_operator(events, n, depth, 7.0, 0.13) @ base)[
                :, nodes
            ]
            for method in methods:
                np.testing.assert_allclose(
                    method.evaluate(ids), expected, atol=1e-10, rtol=1e-11
                )
            if methods[0].independent_exact(ids):
                np.testing.assert_allclose(
                    independent.evaluate(ids), expected, atol=1e-10, rtol=1e-11
                )


def test_interactions_and_orientation_sensitive_pruning():
    src = np.array([0, 1, 0, 1])
    dst = np.array([1, 2, 1, 2])
    stamp = np.arange(4, dtype=float)
    args = (src, dst, stamp, np.eye(3), 2, 5.0, 0.0, np.array([0, 3]), np.array([2]))
    sparse = SparseInteractionOracle(*args)
    independent = IndependentDeletionEffects(*args)
    assert sparse.interaction_edges() == 1
    assert not sparse.independent_exact([0, 3])
    assert np.linalg.norm(sparse.evaluate([0, 3]) - independent.evaluate([0, 3])) > 0
    assert sparse.independent_exact([0])


def test_tied_candidates_have_no_interaction():
    args = (
        np.array([0, 1]),
        np.array([1, 2]),
        np.zeros(2),
        np.eye(3),
        4,
        1.0,
        0.0,
        np.array([0, 1]),
        np.arange(3),
    )
    sparse = SparseInteractionOracle(*args)
    assert sparse.interaction_edges() == 0
    assert sparse.independent_exact([0, 1])
    np.testing.assert_allclose(
        sparse.evaluate([0, 1]), np.r_[np.eye(3)[None], np.zeros((4, 3, 3))]
    )


@pytest.mark.parametrize(
    "cls", [SparseInteractionOracle, LazyForwardBackward, SparseIntervalChain]
)
def test_invalid_deletions_and_empty_pool(cls):
    args = (
        np.array([0, 1]),
        np.array([1, 2]),
        np.arange(2, dtype=float),
        np.eye(3),
        2,
        3.0,
        0.0,
        np.array([0]),
        np.arange(3),
    )
    obj = cls(*args)
    with pytest.raises(ValueError):
        obj.evaluate([1])
    with pytest.raises(ValueError):
        obj.evaluate([0, 0])
    empty = cls(*args[:7], np.empty(0, np.int64), args[-1])
    np.testing.assert_allclose(empty.evaluate([]), obj.evaluate([]))
