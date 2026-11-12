import numpy as np
import pytest
from baselines.walks import replay, enumerated_operator


@pytest.mark.parametrize("cavity", [False, True])
@pytest.mark.parametrize("rate", [0.0, 0.13])
def test_projection_equals_independent_event_enumeration(cavity, rate):
    rng = np.random.default_rng(404000)
    for _ in range(25):
        events = []
        for t in range(1, 7):
            for _ in range(int(rng.integers(1, 4))):
                u, v = rng.choice(5, size=2, replace=False)
                events.append((int(u), int(v), float(t)))
        base = rng.normal(size=(5, 7))
        state = replay(events, base, depth=4, rate=rate, cavity=cavity)
        got = state.query(range(5), 7).transpose(1, 0, 2)
        exact = enumerated_operator(events, 5, 4, 7, rate, cavity)
        np.testing.assert_allclose(got, exact @ base, atol=1e-10, rtol=1e-10)


def test_tied_events_are_not_sequential_paths():
    events = [(0, 1, 1.0), (1, 2, 1.0)]
    for cavity in (False, True):
        state = replay(events, np.eye(3), cavity=cavity)
        assert state.query([0], 2)[0, 2, 2] == 0
        reverse = replay(events[::-1], np.eye(3), cavity=cavity)
        np.testing.assert_array_equal(state.h, reverse.h)
        with pytest.raises(ValueError):
            state.update([(1, 2)], 1.0)


def test_return_component_has_identifiable_counterexample():
    # One repeated pair generates higher-hop walks despite no new neighbors.
    events = [(0, 1, float(t)) for t in range(1, 5)]
    allwalk = replay(events, np.eye(3), cavity=False)
    cavity = replay(events, np.eye(3), cavity=True)
    assert allwalk.query([0], 5)[0, 3, 1] == 4
    assert cavity.query([0], 5)[0, 3, 1] == 0
    # A time-respecting path through four different nodes survives.
    path = replay([(2, 3, 1), (1, 2, 2), (0, 1, 3)], np.eye(4), cavity=True)
    assert path.query([0], 4)[0, 3, 3] == 1


def test_future_events_and_node_names_do_not_change_current_operator():
    events = [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
    original = enumerated_operator(events, 4, 3, 2.5, cavity=True)
    without_future = enumerated_operator(events[:2], 4, 3, 2.5, cavity=True)
    np.testing.assert_array_equal(original, without_future)
    p = np.array([2, 0, 3, 1])
    renamed = [(p[u], p[v], t) for u, v, t in events]
    renamed_op = enumerated_operator(renamed, 4, 3, 2.5, cavity=True)
    np.testing.assert_array_equal(original, renamed_op[:, p][:, :, p])


def test_compiled_bipartite_state_matches_reference():
    from baselines.pilot_features import stream_features

    events = [
        (0, 3, 1.0),
        (1, 3, 1.0),
        (0, 4, 2.0),
        (0, 3, 3.0),
        (2, 4, 3.0),
        (0, 3, 3.0),
        (1, 4, 5.0),
    ]
    src, dst, stamp = np.array(events).T
    src, dst = src.astype(np.int64), dst.astype(np.int64)
    pairs = {}
    pids = []
    for u, v in zip(src, dst):
        if (u, v) not in pairs:
            pairs[u, v] = len(pairs)
        pids.append(pairs[u, v])
    base = np.random.default_rng(123).normal(size=(5, 9))
    selected = np.arange(len(events), dtype=np.int64)
    candidates = dst[:, None]
    pids = np.array(pids, np.int64)
    features, _, state, _ = stream_features(
        src,
        dst,
        stamp,
        pids,
        len(pairs),
        base,
        selected,
        candidates,
        pids[:, None],
        3,
        0.2,
    )
    for qi, (u, v, t) in enumerate(events):
        for mode in (0, 1):
            mat = enumerated_operator(events, 5, 3, t, 0.2, bool(mode)) @ base
            vectors = np.concatenate([mat[:, u], mat[:, v]])
            gram = vectors @ vectors.T
            expected = np.log1p(np.maximum(gram[np.triu_indices(8)], 0.0))
            np.testing.assert_allclose(
                features[qi, 0, 8 + 36 * mode : 44 + 36 * mode], expected, atol=1e-6
            )
