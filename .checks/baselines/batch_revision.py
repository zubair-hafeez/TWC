"""Exact interacting revision batches with original-prefix/edited-suffix factors."""

import numpy as np
from .indexed_retroactive import replay_array


def merge_revision(src, dst, stamp, removed, add_src, add_dst, add_t):
    size = len(src) - np.sum(removed) + len(add_src)
    ns, nd = np.empty(size, np.int64), np.empty(size, np.int64)
    nt = np.empty(size, np.float64)
    old, add, pos = 0, 0, 0
    while old < len(src) or add < len(add_src):
        if old < len(src) and removed[old]:
            old += 1
            continue
        if add < len(add_src) and (old == len(src) or add_t[add] < stamp[old]):
            ns[pos], nd[pos], nt[pos] = add_src[add], add_dst[add], add_t[add]
            add += 1
        else:
            ns[pos], nd[pos], nt[pos] = src[old], dst[old], stamp[old]
            old += 1
        pos += 1
    return ns, nd, nt


def shared_suffix(src, dst, stamp, edit_u, edit_v, edit_t, n, depth, query_time, rate):
    q = len(edit_u)
    state = np.zeros((depth, n, 2 * q), np.float64)
    event, edit, active = 0, 0, 0
    if q:
        event = np.searchsorted(stamp, edit_t[0], side="left")
    while event < len(src) or edit < q:
        next_event = stamp[event] if event < len(src) else np.inf
        next_edit = edit_t[edit] if edit < q else np.inf
        now = min(next_event, next_edit)
        if now >= query_time:
            break
        end = event
        while end < len(src) and stamp[end] == now:
            end += 1
        weight = np.exp(-rate * (query_time - now))
        for k in range(depth - 1, 0, -1):
            for i in range(event, end):
                u, v = src[i], dst[i]
                for col in range(active):
                    state[k, u, col] += weight * state[k - 1, v, col]
                    state[k, v, col] += weight * state[k - 1, u, col]
        event = end
        # Inject AFTER this timestamp's transition: no within-time path.
        while edit < q and edit_t[edit] == now:
            state[0, edit_u[edit], 2 * edit] = 1.0
            state[0, edit_v[edit], 2 * edit + 1] = 1.0
            edit += 1
        active = 2 * edit
    return state


def reconstruct_batch_rows(left, right, nodes, depth):
    out = np.zeros((depth + 1, len(nodes), right.shape[-1]))
    for k in range(1, depth + 1):
        for j in range(k):
            for ni in range(len(nodes)):
                for col in range(left.shape[-1]):
                    coefficient = left[j, nodes[ni], col]
                    if coefficient != 0.0:
                        for d in range(right.shape[-1]):
                            out[k, ni, d] += coefficient * right[k - 1 - j, col, d]
    return out


def differential_batch(
    src, dst, stamp, edit_u, edit_v, edit_t, right, n, depth, query_time, rate
):
    """Exact strong control: propagate dense feature deltas on edited history.

    Uses the same original-prefix vectors as the factored method. Inactive
    degree/node states are skipped; all edit interactions remain present.
    """
    delta = np.zeros((depth + 1, n, right.shape[-1]))
    active = np.zeros((depth + 1, n), np.bool_)
    event, edit = 0, 0
    if len(edit_t):
        event = np.searchsorted(stamp, edit_t[0], side="left")
    while event < len(src) or edit < len(edit_t):
        next_event = stamp[event] if event < len(src) else np.inf
        next_edit = edit_t[edit] if edit < len(edit_t) else np.inf
        now = min(next_event, next_edit)
        if now >= query_time:
            break
        end = event
        while end < len(src) and stamp[end] == now:
            end += 1
        weight = np.exp(-rate * (query_time - now))
        for k in range(depth, 1, -1):
            for i in range(event, end):
                u, v = src[i], dst[i]
                if active[k - 1, v]:
                    active[k, u] = True
                    for d in range(right.shape[-1]):
                        delta[k, u, d] += weight * delta[k - 1, v, d]
                if active[k - 1, u]:
                    active[k, v] = True
                    for d in range(right.shape[-1]):
                        delta[k, v, d] += weight * delta[k - 1, u, d]
        event = end
        # The forcing term uses the ORIGINAL prefix, even after earlier edits.
        while edit < len(edit_t) and edit_t[edit] == now:
            u, v = edit_u[edit], edit_v[edit]
            for k in range(1, depth + 1):
                active[k, u], active[k, v] = True, True
                for d in range(right.shape[-1]):
                    delta[k, u, d] += right[k - 1, 2 * edit, d]
                    delta[k, v, d] += right[k - 1, 2 * edit + 1, d]
            edit += 1
    return delta


class RevisionIndex:
    def __init__(self, src, dst, stamp, base, depth, query_time, rate):
        self.src = src
        self.dst = dst
        self.stamp = stamp
        self.base = base
        self.depth = depth
        self.query_time = query_time
        self.rate = rate

        self.src = np.ascontiguousarray(self.src, dtype=np.int64)
        self.dst = np.ascontiguousarray(self.dst, dtype=np.int64)
        self.stamp = np.ascontiguousarray(self.stamp, dtype=np.float64)
        self.base = np.ascontiguousarray(self.base, dtype=np.float64)
        assert np.all(np.diff(self.stamp) >= 0)
        assert len(self.stamp) == 0 or self.stamp[-1] < self.query_time
        assert np.all(self.src != self.dst)
        assert np.all(self.src >= 0) and np.all(self.dst >= 0)
        assert max(self.src.max(initial=0), self.dst.max(initial=0)) < len(self.base)
        self.h, self.prefix = replay_array(
            self.src,
            self.dst,
            self.stamp,
            self.base,
            self.depth,
            self.query_time,
            self.rate,
            make_prefix=True,
        )
        # Stable node sort preserves event order of the interleaved incidences.
        endpoints = np.column_stack([self.src, self.dst]).ravel()
        order = np.argsort(endpoints, kind="stable")
        self.positions = (order // 2).astype(np.int64)
        self.sides = (order % 2).astype(np.int8)
        self.offsets = np.r_[
            0, np.cumsum(np.bincount(endpoints, minlength=len(self.base)))
        ]

    def prefix_at(self, node, stamp):
        lo, hi = self.offsets[node : node + 2]
        ids = self.positions[lo:hi]
        pos = np.searchsorted(self.stamp[ids], stamp, side="left")
        if pos < len(ids):
            return self.prefix[ids[pos], self.sides[lo + pos]]
        # No incident events between this time and the final state.
        return self.h[: self.depth, node]

    def revised_stream(self, remove_ids=(), additions=()):
        remove_ids = np.asarray(remove_ids, dtype=np.int64)
        if len(np.unique(remove_ids)) != len(remove_ids):
            raise ValueError("Duplicate event removal")
        if np.any(remove_ids < 0) or np.any(remove_ids >= len(self.src)):
            raise ValueError("Invalid event id")
        removed = np.zeros(len(self.src), dtype=np.bool_)
        removed[remove_ids] = True
        additions = sorted(additions, key=lambda e: e[2])
        for u, v, t in additions:
            if (
                u == v
                or min(u, v) < 0
                or max(u, v) >= len(self.base)
                or not t < self.query_time
            ):
                raise ValueError("Addition outside the declared node/time contract")
        add_u = np.array([e[0] for e in additions], dtype=np.int64)
        add_v = np.array([e[1] for e in additions], dtype=np.int64)
        add_t = np.array([e[2] for e in additions], dtype=np.float64)
        stream = merge_revision(
            self.src, self.dst, self.stamp, removed, add_u, add_v, add_t
        )
        edits = [(self.src[i], self.dst[i], self.stamp[i], -1.0) for i in remove_ids]
        edits += [(u, v, t, 1.0) for u, v, t in additions]
        edits.sort(key=lambda e: e[2])
        return stream, edits

    def factors(self, remove_ids=(), additions=()):
        """Prepare the common index lookup/stream merge used by both controls."""
        stream, edits = self.revised_stream(remove_ids, additions)
        eu = np.array([e[0] for e in edits], dtype=np.int64)
        ev = np.array([e[1] for e in edits], dtype=np.int64)
        et = np.array([e[2] for e in edits], dtype=np.float64)
        right = np.empty((self.depth, 2 * len(edits), self.base.shape[1]))
        for i, (u, v, t, change) in enumerate(edits):
            factor = change * np.exp(-self.rate * (self.query_time - t))
            right[:, 2 * i] = factor * self.prefix_at(v, t)
            right[:, 2 * i + 1] = factor * self.prefix_at(u, t)
        return stream, eu, ev, et, right

    def correction(self, remove_ids=(), additions=(), nodes=None, method="factored"):
        if nodes is None:
            nodes = np.arange(len(self.base))
        nodes = np.asarray(nodes, dtype=np.int64)
        if np.any(nodes < 0) or np.any(nodes >= len(self.base)):
            raise ValueError("Invalid query node")
        if method not in ("factored", "differential"):
            raise ValueError("Unknown correction method")
        stream, eu, ev, et, right = self.factors(remove_ids, additions)
        if not len(eu):
            return self.h[:, nodes].copy()
        if method == "differential":
            delta = differential_batch(
                *stream,
                eu,
                ev,
                et,
                right,
                len(self.base),
                self.depth,
                self.query_time,
                self.rate,
            )
            return self.h[:, nodes] + delta[:, nodes]
        left = shared_suffix(
            *stream, eu, ev, et, len(self.base), self.depth, self.query_time, self.rate
        )
        return self.h[:, nodes] + reconstruct_batch_rows(left, right, nodes, self.depth)
