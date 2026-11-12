import itertools
import numpy as np
from baselines.independent_effects import IndependentDeletionEffects
from baselines.revision_circuit import DeletionCircuit
from baselines.walks import enumerated_operator


def test_independent_approximation_can_miss_the_only_two_edit_counterfactual():
    # Three strict-time a-b-c paths: (1,2), (1,4), (3,4).
    # The candidates are a-b at1 and b-c at4. Each single deletion leaves
    # one path; both leave zero. Independent effects wrongly predict -1.
    events = [(0, 1, 1.0), (1, 2, 2.0), (0, 1, 3.0), (1, 2, 4.0)]
    a = np.array(events)
    src, dst, t = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2]
    args = (src, dst, t, np.eye(3), 2, 5.0, 0.0, np.array([0, 3]), np.array([2]))
    exact = DeletionCircuit(*args)
    approx = IndependentDeletionEffects(*args)
    exact_values, approximate_values = [], []
    for size in range(3):
        for removed in itertools.combinations([0, 3], size):
            changed = [e for i, e in enumerate(events) if i not in removed]
            truth = enumerated_operator(changed, 3, 2, 5.0, 0.0)[2, 2, 0]
            value = exact.evaluate(removed)[2, 0, 0]
            assert value == truth
            exact_values.append(value)
            approximate_values.append(approx.evaluate(removed)[2, 0, 0])
    assert exact_values == [3.0, 1.0, 1.0, 0.0]
    assert approximate_values == [3.0, 1.0, 1.0, -1.0]
    # A simple nonlinear readout: A iff |walk feature| >0.5. It can be
    # implemented with two ReLUs. First-order evaluation would incorrectly
    # report no counterfactual within the entire two-edit candidate pool.
    assert [abs(x) > 0.5 for x in exact_values] == [True, True, True, False]
    assert all(abs(x) > 0.5 for x in approximate_values)


def test_same_interaction_failure_is_visible_in_the_used_gram_feature_family():
    events = np.array([(0, 1, 1.0), (1, 2, 2.0), (0, 1, 3.0), (1, 2, 4.0)])
    args = (
        events[:, 0].astype(np.int64),
        events[:, 1].astype(np.int64),
        events[:, 2],
        np.eye(3),
        2,
        5.0,
        0.0,
        np.array([0, 3]),
        np.array([2]),
    )
    exact, approximate = DeletionCircuit(*args), IndependentDeletionEffects(*args)
    subsets = [(), (0,), (3,), (0, 3)]
    exact_norms = [float(np.sum(exact.evaluate(ids)[2, 0] ** 2)) for ids in subsets]
    approximate_norms = [
        float(np.sum(approximate.evaluate(ids)[2, 0] ** 2)) for ids in subsets
    ]
    assert exact_norms == [10.0, 2.0, 1.0, 0.0]
    assert approximate_norms == [10.0, 2.0, 1.0, 1.0]
    # log1p(squared norm) is one of the actual Gram features. A single tree
    # split at log(1.5) realizes this decision; no absolute-value decoder needed.
    assert [np.log1p(x) > np.log(1.5) for x in exact_norms] == [True, True, True, False]
    assert all(np.log1p(x) > np.log(1.5) for x in approximate_norms)
