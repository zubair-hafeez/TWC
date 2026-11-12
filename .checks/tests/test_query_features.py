import itertools
import numpy as np
from baselines.query_features import RecencyIndex, backward_cone
from baselines.walks import enumerated_operator


def test_all_recency_features_match_fresh_independent_history_counts():
    events = [
        (0, 2, 0.0),
        (0, 2, 1.0),
        (1, 2, 2.0),
        (0, 3, 3.0),
        (0, 2, 3.0),
        (0, 3, 4.0),
        (1, 3, 5.0),
    ]
    a = np.array(events)
    src, dst, t = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2]
    nodes, candidates = np.array([0, 2, 3]), np.array([0, 1, 3, 4, 5])
    index = RecencyIndex(src, dst, t, nodes, candidates, 4)
    for size in range(len(candidates) + 1):
        for selected in itertools.combinations(range(len(candidates)), size):
            removed = set(candidates[list(selected)])
            changed = [e for i, e in enumerate(events) if i not in removed]
            h = enumerated_operator(changed, 4, 3, 6.0, 0.2)[:, nodes]
            features = index.features(h, selected, 6.0)
            for row, v in enumerate(nodes[1:]):
                groups = [
                    [e for e in changed if 0 in e[:2]],
                    [e for e in changed if v in e[:2]],
                    [e for e in changed if e[0] == 0 and e[1] == v],
                ]
                c = [len(g) for g in groups]
                gaps = [
                    np.log1p((6.0 - max(e[2] for e in g)) * 1000) if g else 20.0
                    for g in groups
                ]
                expected = [
                    np.log1p(c[0]),
                    np.log1p(c[1]),
                    np.log1p(c[2]),
                    float(c[2] > 0),
                    *gaps,
                    np.log1p(c[0] * c[1]),
                ]
                np.testing.assert_allclose(features[row, :8], expected, rtol=1e-6)


def test_backward_cone_never_prunes_a_nonzero_event_effect():
    rng = np.random.default_rng(406500)
    for _ in range(20):
        events = [
            (int(u), int(v), float(t))
            for t in range(1, 7)
            for u, v in [rng.choice(5, 2, replace=False)]
        ]
        a = np.array(events)
        nodes = np.array([0, 3])
        eligible = backward_cone(
            a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2], 5, 3, nodes
        )
        original = enumerated_operator(events, 5, 3, 7.0, 0.2)[:, nodes]
        for i in np.flatnonzero(~eligible):
            changed = [e for j, e in enumerate(events) if i != j]
            np.testing.assert_allclose(
                enumerated_operator(changed, 5, 3, 7.0, 0.2)[:, nodes],
                original,
                atol=1e-12,
            )


def test_query_cone_pruned_controls_match_all_selected_subsets():
    from baselines.compact_controls import CompactRevisionControls

    rng = np.random.default_rng(406900)
    for _ in range(12):
        events = [
            (int(u), int(v), float(t // 2))
            for t in range(14)
            for u, v in [rng.choice(8, 2, replace=False)]
        ]
        a = np.array(events)
        src, dst, stamp = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2]
        nodes, depth = np.array([0]), 3
        mask = backward_cone(src, dst, stamp, 8, depth, nodes)
        candidates = np.flatnonzero(mask)[-5:]
        if not len(candidates):
            continue
        base = rng.normal(size=(8, 5))
        control = CompactRevisionControls(
            src, dst, stamp, base, depth, 8.0, 0.2, candidates, nodes, mask
        )
        for size in range(len(candidates) + 1):
            for removed in itertools.combinations(candidates, size):
                changed = [e for i, e in enumerate(events) if i not in removed]
                oracle = (enumerated_operator(changed, 8, depth, 8.0, 0.2) @ base)[
                    :, nodes
                ]
                for method in ("differential", "factored"):
                    np.testing.assert_allclose(
                        control.evaluate(removed, method), oracle, atol=1e-10
                    )
