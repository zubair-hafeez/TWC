import numpy as np
import pytest
from baselines.retroactive import (
    projected_replay,
    event_correction_factors,
    materialize_delta,
    query_delta,
    batch_correction,
    apply_edits,
)
from baselines.walks import enumerated_operator


@pytest.mark.parametrize("rate", [0.0, 0.2])
def test_full_replay_matches_independent_path_enumeration(rate):
    events = [(0, 1, 1), (1, 2, 1), (0, 2, 2), (1, 2, 2), (2, 3, 3)]
    base = np.random.default_rng(45).normal(size=(4, 7))
    expected = enumerated_operator(events, 4, 4, 4, rate) @ base
    got = projected_replay(events, base, 4, 4, rate)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("operation", ["insert", "remove", "retime"])
@pytest.mark.parametrize("rate", [0.0, 0.2])
def test_exact_correction_equals_replay_and_independent_enumeration(operation, rate):
    rng = np.random.default_rng(405000)
    for _ in range(20):
        events = []
        for s in range(1, 6):
            for _ in range(2):
                u, v = rng.choice(5, 2, replace=False)
                events.append((int(u), int(v), float(s)))
        base = rng.normal(size=(5, 8))
        depth, query = 4, 7.0
        old = projected_replay(events, base, depth, query, rate)
        corrected = old.copy()
        current = events.copy()
        if operation in ("remove", "retime"):
            event = current[3]
            l, r = event_correction_factors(
                current, base, depth, query, rate, event, -1.0
            )
            corrected += materialize_delta(l, r, depth)
            np.testing.assert_allclose(
                query_delta(l, r, depth, [2, 4]),
                materialize_delta(l, r, depth)[:, [2, 4]],
                atol=1e-12,
            )
            current.remove(event)
        if operation in ("insert", "retime"):
            event = (0, 4, 2.5) if operation == "insert" else (event[0], event[1], 4.0)
            l, r = event_correction_factors(
                current, base, depth, query, rate, event, +1.0
            )
            corrected += materialize_delta(l, r, depth)
            current.append(event)
        replay = projected_replay(current, base, depth, query, rate)
        independent = enumerated_operator(current, 5, depth, query, rate) @ base
        np.testing.assert_allclose(corrected, replay, atol=1e-10)
        np.testing.assert_allclose(corrected, independent, atol=1e-10)


def test_rank_bound_on_exact_operator_perturbation():
    events = [
        (0, 1, 1),
        (1, 2, 2),
        (2, 3, 3),
        (0, 2, 4),
        (3, 4, 5),
        (1, 5, 6),
        (4, 6, 7),
        (5, 7, 8),
    ]
    base = np.eye(8)
    depth = 4
    l, r = event_correction_factors(events, base, depth, 9, 0.1, events[2], -1)
    delta = materialize_delta(l, r, depth)
    for k in range(1, depth + 1):
        assert np.linalg.matrix_rank(delta[k], tol=1e-10) <= 2 * k


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_rank_bound_is_attained_by_disjoint_past_and_future_chains(depth):
    # Each endpoint has a past arm (time decreases away from the endpoint)
    # and a future arm (time increases away from it). All arm vertices are
    # distinct. For degree k, the 2k left and right vectors are independent.
    events = [(0, 1, 0.0)]
    cursor = 2
    for endpoint in (0, 1):
        for sign in (-1, 1):
            previous = endpoint
            for step in range(1, depth):
                events.append((previous, cursor, float(sign * step)))
                previous = cursor
                cursor += 1
    l, r = event_correction_factors(
        events, np.eye(cursor), depth, depth + 1.0, 0.0, (0, 1, 0.0), -1.0
    )
    delta = materialize_delta(l, r, depth)
    assert np.linalg.matrix_rank(delta[depth], tol=1e-10) == 2 * depth


def test_indexed_removal_matches_independent_oracle_for_tied_multigraph():
    from baselines.indexed_retroactive import (
        replay_array,
        indexed_removal,
        make_checkpoints,
        checkpoint_replay,
        differential_removal,
    )

    events = [
        (0, 3, 1.0),
        (1, 3, 1.0),
        (0, 4, 2.0),
        (0, 3, 3.0),
        (2, 4, 3.0),
        (0, 3, 3.0),
        (1, 4, 5.0),
    ]
    a = np.array(events)
    src, dst, t = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2]
    base = np.random.default_rng(1234).normal(size=(5, 11))
    h, prefix = replay_array(src, dst, t, base, 4, 6.0, 0.2, make_prefix=True)
    offsets, checkpoints = make_checkpoints(src, dst, t, base, 4, 6.0, 0.2, 2)
    for index in range(len(events)):
        rows = np.array([0, 2, 4], dtype=np.int64)
        updated = indexed_removal(src, dst, t, h, prefix, index, 6.0, 0.2, rows)
        changed = [e for j, e in enumerate(events) if j != index]
        expected = enumerated_operator(changed, 5, 4, 6.0, 0.2) @ base
        np.testing.assert_allclose(updated, expected[:, rows], atol=1e-10)
        local = differential_removal(src, dst, t, h, prefix, index, 6.0, 0.2, rows)
        np.testing.assert_allclose(local, expected[:, rows], atol=1e-10)
        cp = np.searchsorted(np.array(offsets), index, side="right") - 1
        checkpoint = checkpoint_replay(
            src, dst, t, checkpoints[cp], offsets[cp], 6.0, 0.2, index
        )
        np.testing.assert_allclose(checkpoint, expected, atol=1e-10)


def test_multiple_interacting_edits_require_cross_terms():
    events = [(0, 1, 1.0), (1, 2, 2.0)]
    edits = [(0, 1, 1.0, -1), (1, 2, 2.0, -1)]
    base = np.eye(3)
    original = projected_replay(events, base, 2, 3.0, 0.0)
    naive = original.copy()
    for u, v, t, change in edits:
        l, r = event_correction_factors(events, base, 2, 3.0, 0.0, (u, v, t), change)
        naive += materialize_delta(l, r, 2)
    assert naive[2, 2, 0] == -1.0  # impossible negative walk count
    exact = original + batch_correction(events, base, 2, 3.0, 0.0, edits)
    np.testing.assert_allclose(exact, projected_replay([], base, 2, 3.0, 0.0))


@pytest.mark.parametrize("rate", [0.0, 0.2])
def test_batch_telescope_matches_enumeration_and_is_edit_order_independent(rate):
    rng = np.random.default_rng(405500)
    for _ in range(20):
        events = []
        for t in range(1, 6):
            for _ in range(2):
                u, v = rng.choice(5, 2, replace=False)
                events.append((int(u), int(v), float(t)))
        removals = rng.choice(len(events), 3, replace=False)
        edits = [(*events[i], -1) for i in removals]
        edits.extend([(0, 4, 2.0, 1), (2, 3, 2.5, 1), (0, 4, 4.0, 1)])
        base = rng.normal(size=(5, 7))
        original = projected_replay(events, base, 4, 7.0, rate)
        delta = batch_correction(events, base, 4, 7.0, rate, edits)
        updated = apply_edits(events, edits)
        expected = enumerated_operator(updated, 5, 4, 7.0, rate) @ base
        np.testing.assert_allclose(original + delta, expected, atol=1e-10)
        # Reorder only the independent edit application list. Invalid
        # intermediate negative multiplicities are correctly disallowed.
        delta2 = batch_correction(events, base, 4, 7.0, rate, edits[::-1])
        np.testing.assert_allclose(delta2, delta, atol=1e-10)


def test_indexed_revision_batch_with_insertions_deletions_and_retiming():
    from baselines.batch_revision import RevisionIndex

    rng = np.random.default_rng(405900)
    for _ in range(10):
        events = []
        for t in range(1, 6):
            for _ in range(2):
                u, v = rng.choice(5, 2, replace=False)
                events.append((int(u), int(v), float(t)))
        a = np.array(events)
        base = rng.normal(size=(5, 9))
        index = RevisionIndex(a[:, 0], a[:, 1], a[:, 2], base, 4, 7.0, 0.2)
        removes = [1, 4, 7]
        additions = [(0, 4, 0.5), (2, 3, 2.0), (events[4][0], events[4][1], 4.5)]
        corrected = index.correction(removes, additions)
        updated = [e for i, e in enumerate(events) if i not in removes] + additions
        expected = enumerated_operator(updated, 5, 4, 7.0, 0.2) @ base
        np.testing.assert_allclose(corrected, expected, atol=1e-10)
        local = index.correction(removes, additions, method="differential")
        np.testing.assert_allclose(local, expected, atol=1e-10)
        # Arbitrary insertion-time prefix lookup, including entirely inactive
        # intervals and existing timestamp groups.
        for t in (0.5, 2.0, 4.5, 6.0):
            direct = projected_replay(
                [e for e in events if e[2] < t], base, 3, 7.0, 0.2
            )
            for node in range(5):
                np.testing.assert_allclose(
                    index.prefix_at(node, t), direct[:, node], atol=1e-10
                )
