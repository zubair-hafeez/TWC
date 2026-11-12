"""v3 correctness: independent references, pruning boundaries, lazy semantics, certificates."""

import itertools
import numpy as np
import pytest
import method_list as E
from common import encode
import sparse_etwc as S
from independent_reference import replay_rows, polynomial_product, cone_events
from baselines.walks import enumerated_operator

EXACT = (
    "eager_etwc",
    "source_lazy",
    "pair_lazy",
    "sparse_compact",
    "packed_compact",
    "sparse_per_source",
    "adaptive",
)
TOL = dict(atol=1e-11, rtol=1e-11)


def random_history(seed, n=6, m=14, q=6, ties=True, duplicates=True):
    rng = np.random.default_rng(seed)
    src = rng.integers(n, size=m)
    dst = (src + rng.integers(1, n, size=m)) % n
    if ties:
        stamp = np.sort(rng.integers(0, m // 2, size=m)).astype(float) / m
    else:
        stamp = np.arange(m, dtype=float) / m
    if duplicates:  # repeat one event instance exactly (same endpoints and time)
        src[m // 2 + 1], dst[m // 2 + 1], stamp[m // 2 + 1] = (
            src[m // 2],
            dst[m // 2],
            stamp[m // 2],
        )
    base = rng.normal(size=(n, 3))  # signed projections
    candidates = np.sort(rng.choice(m, q, replace=False))
    nodes = np.sort(rng.choice(n, 2, replace=False))
    return src, dst, stamp, base, candidates, nodes


def all_subsets(candidates, limit=None):
    out = []
    for b in range(len(candidates) + 1 if limit is None else limit + 1):
        out.extend(itertools.combinations(candidates.tolist(), b))
    return out


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_independent_references_agree(seed, depth):
    src, dst, stamp, base, candidates, nodes = random_history(300 + seed)
    rate, T = 1.7, 1.2
    full = np.arange(len(src))
    for ids in all_subsets(candidates, 3):
        a = polynomial_product(src, dst, stamp, base, depth, T, rate, nodes, ids)
        b = replay_rows(src, dst, stamp, base, depth, T, rate, nodes, ids)
        c = replay_rows(src, dst, stamp, base, depth, T, rate, nodes, ids, cone=full)
        events = [
            (int(u), int(v), float(t))
            for e, (u, v, t) in enumerate(zip(src, dst, stamp))
            if e not in ids
        ]
        d = (enumerated_operator(events, len(base), depth, T, rate) @ base)[:, nodes]
        for other in (b, c, d):
            np.testing.assert_allclose(other, a, **TOL)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_every_evaluator_matches_physical_deletion(seed, depth):
    src, dst, stamp, base, candidates, nodes = random_history(400 + seed)
    args = (src, dst, stamp, base, depth, 1.2, 0.9, candidates, nodes)
    objects = {k: E.METHODS[k](*args) for k in EXACT + ("independent_effects",)}
    hypotheses = all_subsets(candidates)
    hyps, lengths = encode(candidates, hypotheses)
    reference = np.stack(
        [
            polynomial_product(src, dst, stamp, base, depth, 1.2, 0.9, nodes, ids)
            for ids in hypotheses
        ]
    )
    for key in EXACT:
        out = objects[key].out_buffer(len(lengths))
        objects[key].batch(hyps, lengths, out)
        np.testing.assert_allclose(out, reference, **TOL, err_msg=key)
    certificate = objects["sparse_compact"].certificate_batch(hyps, lengths)
    ie = objects["independent_effects"].out_buffer(len(lengths))
    objects["independent_effects"].batch(hyps, lengths, ie)
    np.testing.assert_allclose(ie[certificate], reference[certificate], **TOL)


def chain_history(length=9, depth=5):
    """Path 0-1-...-L with increasing times; output row at the far end makes beta large."""
    src = np.arange(length)
    dst = np.arange(1, length + 1)
    stamp = np.arange(length, dtype=float) / length
    base = np.random.default_rng(7).normal(size=(length + 1, 2))
    return src, dst, stamp, base, np.arange(length), np.array([length])


@pytest.mark.parametrize("depth", [4, 5, 6])
def test_pruning_removes_coefficients_and_stays_exact(depth):
    src, dst, stamp, base, candidates, nodes = chain_history()
    args = (src, dst, stamp, base, depth, 1.0, 0.5, candidates, nodes)
    pruned, packed = E.Compact(*args, prune=True), E.Compact(*args, prune=False)
    # Non-vacuous: pruning must remove both pairs and individual degrees here.
    assert pruned.storage()["coefficients"] < packed.storage()["coefficients"]
    assert pruned.pairs < packed.pairs
    assert np.any(pruned.ent_lo > 0) and np.any(pruned.ent_hi < depth - 2)
    hypotheses = all_subsets(candidates, 4)
    hyps, lengths = encode(candidates, hypotheses)
    out = pruned.out_buffer(len(lengths))
    pruned.batch(hyps, lengths, out)
    for r, ids in enumerate(hypotheses):
        np.testing.assert_allclose(
            out[r],
            replay_rows(src, dst, stamp, base, depth, 1.0, 0.5, nodes, ids),
            **TOL,
        )


def test_degree_bound_is_attained_somewhere():
    """Dropping the top retained degree of some entry changes some subset (bound is not vacuous)."""
    changed = 0
    for seed in range(12):
        src, dst, stamp, base, candidates, nodes = random_history(
            500 + seed, n=5, m=16, q=6
        )
        args = (src, dst, stamp, base, 5, 1.2, 0.3, candidates, nodes)
        obj = E.Compact(*args, prune=True)
        hypotheses = all_subsets(candidates, 4)
        hyps, lengths = encode(candidates, hypotheses)
        exact = obj.out_buffer(len(lengths))
        obj.batch(hyps, lengths, exact)
        for x in np.flatnonzero(obj.ent_hi > obj.ent_lo):
            obj.ent_hi[x] -= 1
            mutated = obj.out_buffer(len(lengths))
            obj.batch(hyps, lengths, mutated)
            obj.ent_hi[x] += 1
            if np.max(np.abs(mutated - exact)) > 1e-9:
                changed += 1
    assert changed > 0


def test_tied_candidates_and_same_time_deletions():
    src = np.array([0, 1, 2, 0, 1, 3])
    dst = np.array([1, 2, 3, 2, 3, 0])
    stamp = np.array([0, 0, 0, 0.5, 0.5, 0.5])
    base = np.random.default_rng(3).normal(size=(4, 3))
    for depth in (2, 3, 4):
        args = (src, dst, stamp, base, depth, 1.0, 1.3, np.arange(6), np.array([0, 3]))
        objects = {k: E.METHODS[k](*args) for k in EXACT}
        for ids in all_subsets(np.arange(6)):
            ref = polynomial_product(
                src, dst, stamp, base, depth, 1.0, 1.3, np.array([0, 3]), ids
            )
            for key, obj in objects.items():
                np.testing.assert_allclose(obj.evaluate(ids), ref, **TOL, err_msg=key)


def test_higher_order_interaction_breaks_independent_effects():
    src, dst, stamp, base, candidates, nodes = chain_history(length=4)
    args = (src, dst, stamp, base, 4, 1.0, 0.0, candidates, np.array([4]))
    exact, ie = E.Eager(*args), E.Independent(*args)
    ids = (0, 1, 2)
    ref = polynomial_product(src, dst, stamp, base, 4, 1.0, 0.0, np.array([4]), ids)
    np.testing.assert_allclose(exact.evaluate(ids), ref, **TOL)
    assert np.max(np.abs(ie.evaluate(ids) - ref)) > 1e-3
    assert not E.Compact(*args).independent_exact(ids)


def test_lazy_evaluators_compute_only_requested_transfers_and_are_order_free():
    src, dst, stamp, base, candidates, nodes = random_history(
        77, n=7, m=30, q=8, ties=False
    )
    args = (src, dst, stamp, base, 4, 1.2, 0.6, candidates, nodes)
    hypotheses = [
        (int(candidates[2]), int(candidates[5])),
        (int(candidates[1]), int(candidates[2]), int(candidates[7])),
    ]
    hyps, lengths = encode(candidates, hypotheses)
    sb, pl = E.SourceLazy(*args), E.PairLazy(*args)
    for obj in (sb, pl):
        obj.batch(hyps, lengths, obj.out_buffer(2))
    assert sb.filled.nonzero()[0].tolist() == [1, 2]
    assert int(pl.filled.sum()) == 4  # (5,2), (2,1), (7,1), (7,2)
    everything = all_subsets(candidates, 3)
    order = np.random.default_rng(1).permutation(len(everything))
    a_h, a_l = encode(candidates, everything)
    forward, shuffled = E.SourceLazy(*args), E.SourceLazy(*args)
    out_a, out_b = forward.out_buffer(len(a_l)), shuffled.out_buffer(len(a_l))
    forward.batch(a_h, a_l, out_a)
    shuffled.batch(a_h[order], a_l[order], out_b)
    np.testing.assert_array_equal(out_a[order], out_b)
    assert forward.misses <= len(candidates) - 1


def test_empty_and_degenerate_inputs():
    src, dst, stamp, base, candidates, nodes = random_history(9)
    for key in EXACT + ("independent_effects",):
        empty = E.METHODS[key](
            src, dst, stamp, base, 3, 1.2, 0.5, np.empty(0, np.int64), nodes
        )
        np.testing.assert_allclose(
            empty.evaluate(()),
            replay_rows(src, dst, stamp, base, 3, 1.2, 0.5, nodes),
            **TOL,
        )
        obj = E.METHODS[key](src, dst, stamp, base, 3, 1.2, 0.5, candidates, nodes)
        with pytest.raises(ValueError):
            obj.evaluate((int(candidates[0]), int(candidates[0])))
        with pytest.raises(ValueError):
            obj.evaluate((int(np.setdiff1d(np.arange(len(src)), candidates)[0]),))
    last = E.Eager(src, dst, stamp, base, 3, 1.2, 0.5, np.array([len(src) - 1]), nodes)
    np.testing.assert_allclose(
        last.evaluate((len(src) - 1,)),
        replay_rows(src, dst, stamp, base, 3, 1.2, 0.5, nodes, (len(src) - 1,)),
        **TOL,
    )


def test_compiled_representation_evaluates_without_history(tmp_path):
    src, dst, stamp, base, candidates, nodes = random_history(31, n=7, m=24, q=7)
    args = (src, dst, stamp, base, 4, 1.2, 0.8, candidates, nodes)
    eager, sparse = E.Eager(*args), E.Compact(*args)
    np.savez(
        tmp_path / "eager.npz",
        prefix=eager.prefix,
        transfers=eager.transfers,
        suffix=eager.suffix,
        weights=eager.weights,
        original=eager.original,
    )
    keys = (
        "prefix",
        "suffix",
        "weights",
        "original",
        "indptr",
        "ent_j",
        "ent_e",
        "ent_a",
        "ent_lo",
        "ent_hi",
        "ent_off",
        "values",
    )
    np.savez(tmp_path / "sparse.npz", **{k: getattr(sparse, k) for k in keys})
    del eager, sparse  # the reloaded arrays are the only remaining state
    hypotheses = all_subsets(candidates, 3)
    hyps, lengths = encode(candidates, hypotheses)
    ref = np.stack(
        [
            replay_rows(src, dst, stamp, base, 4, 1.2, 0.8, nodes, ids)
            for ids in hypotheses
        ]
    )
    a = np.load(tmp_path / "eager.npz")
    out = np.empty_like(ref)
    from etwc import eager_batch

    eager_batch(
        a["prefix"],
        a["transfers"],
        a["suffix"],
        a["weights"],
        a["original"],
        hyps,
        lengths,
        out,
    )
    np.testing.assert_allclose(out, ref, **TOL)
    b = np.load(tmp_path / "sparse.npz")
    out2 = np.empty_like(ref)
    S.compact_batch(
        *[b[k] for k in keys],
        np.full(len(candidates), -1, np.int64),
        hyps,
        lengths,
        out2,
    )
    np.testing.assert_allclose(out2, ref, **TOL)


def test_packed_ablation_has_no_certificate_regression():
    """v2 PackedEndpointOracle inherited independent_exact() with an empty support graph."""
    from baselines.packed_oracle import PackedEndpointOracle

    src, dst, stamp, base, candidates, nodes = chain_history(length=6)
    args = (src, dst, stamp, base, 3, 1.0, 0.5, candidates, nodes)
    frozen = PackedEndpointOracle(*args)
    # Frozen v2 behavior: it answers a certificate question it cannot support.
    assert frozen.independent_exact((0, 5)) is False
    assert E.Compact(*args, prune=True).independent_exact((0, 5)) is True
    with pytest.raises(NotImplementedError):
        E.Compact(*args, prune=False).independent_exact((0, 5))


def test_cone_restriction_is_exact_on_dense_cone():
    src, dst, stamp, base, candidates, nodes = random_history(55, n=8, m=40, q=6)
    cone = cone_events(src, dst, stamp, nodes, 3)
    assert len(cone) > 0
    for ids in all_subsets(candidates, 2):
        np.testing.assert_allclose(
            replay_rows(src, dst, stamp, base, 3, 1.2, 0.5, nodes, ids, cone=cone),
            polynomial_product(src, dst, stamp, base, 3, 1.2, 0.5, nodes, ids),
            **TOL,
        )


@pytest.mark.parametrize("seed", range(4))
def test_block_size_does_not_change_compact_representation(seed):
    src, dst, stamp, base, candidates, nodes = random_history(
        600 + seed, n=7, m=30, q=9
    )
    args = (src, dst, stamp, base, 5, 1.2, 0.7, candidates, nodes)
    objs = [E.Compact(*args, block=b) for b in (1, 2, 4, 9)]
    keys = (
        "indptr",
        "ent_j",
        "ent_e",
        "ent_a",
        "ent_lo",
        "ent_hi",
        "ent_off",
        "values",
    )
    for other in objs[1:]:
        for k in keys:
            np.testing.assert_array_equal(
                getattr(other, k), getattr(objs[0], k), err_msg=k
            )
