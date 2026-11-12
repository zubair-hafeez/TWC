import itertools
import numpy as np
import pytest
from baselines.revision_circuit import DeletionCircuit
from baselines.walks import enumerated_operator


@pytest.mark.parametrize("rate", [0.0, 0.2])
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_every_deletion_subset_matches_independent_walk_enumeration(rate, depth):
    events = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 1, 2.0),
        (0, 1, 2.0),
        (1, 2, 3.0),
        (2, 3, 4.0),
        (0, 2, 5.0),
    ]
    a = np.array(events)
    base = np.random.default_rng(406300).normal(size=(4, 7))
    nodes = np.array([0, 2, 3])
    circuit = DeletionCircuit(
        a[:, 0], a[:, 1], a[:, 2], base, depth, 6.0, rate, np.arange(len(events)), nodes
    )
    for size in range(len(events) + 1):
        for removed in itertools.combinations(range(len(events)), size):
            updated = [e for i, e in enumerate(events) if i not in removed]
            oracle = (enumerated_operator(updated, 4, depth, 6.0, rate) @ base)[
                :, nodes
            ]
            np.testing.assert_allclose(circuit.evaluate(removed), oracle, atol=1e-10)


def test_candidate_contract_and_nonconsecutive_candidate_subset():
    a = np.array([(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0), (0, 1, 4.0)])
    circuit = DeletionCircuit(
        a[:, 0],
        a[:, 1],
        a[:, 2],
        np.eye(3),
        3,
        5.0,
        0.1,
        np.array([0, 2]),
        np.arange(3),
    )
    expected = enumerated_operator([(1, 2, 2.0), (0, 1, 4.0)], 3, 3, 5.0, 0.1)
    np.testing.assert_allclose(circuit.evaluate([2, 0]), expected, atol=1e-10)
    with pytest.raises(ValueError):
        circuit.evaluate([1])
    with pytest.raises(ValueError):
        circuit.evaluate([3])
    with pytest.raises(ValueError):
        circuit.evaluate([0, 0])


def test_compact_masked_controls_for_all_interacting_subsets():
    from baselines.compact_controls import CompactRevisionControls

    events = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 1, 2.0),
        (0, 1, 2.0),
        (1, 2, 3.0),
        (2, 3, 4.0),
        (0, 2, 5.0),
    ]
    a = np.array(events)
    src, dst, t = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2]
    base = np.random.default_rng(406700).normal(size=(8, 9))
    nodes, candidates = np.array([0, 2, 7]), np.array([2, 3, 4, 5, 6])
    control = CompactRevisionControls(src, dst, t, base, 4, 6.0, 0.2, candidates, nodes)
    for size in range(len(candidates) + 1):
        for removed in itertools.combinations(candidates, size):
            changed = [e for i, e in enumerate(events) if i not in removed]
            oracle = (enumerated_operator(changed, 8, 4, 6.0, 0.2) @ base)[:, nodes]
            for method in ("differential", "factored"):
                np.testing.assert_allclose(
                    control.evaluate(removed, method), oracle, atol=1e-10
                )
