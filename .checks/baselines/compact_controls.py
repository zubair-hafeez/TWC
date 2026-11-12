"""Strong exact controls: compact suffix nodes and mask deletions in place."""

import numpy as np
from .revision_circuit import forward_index
from .batch_revision import reconstruct_batch_rows


def masked_differential(
    src, dst, stamp, removed, eu, ev, et, right, n, depth, query, rate
):
    delta = np.zeros((depth + 1, n, right.shape[-1]))
    active = np.zeros((depth + 1, n), np.bool_)
    event, edit = np.searchsorted(stamp, et[0], side="left"), 0
    while event < len(src) or edit < len(et):
        now = min(
            stamp[event] if event < len(src) else np.inf,
            et[edit] if edit < len(et) else np.inf,
        )
        end = event
        while end < len(src) and stamp[end] == now:
            end += 1
        w = np.exp(-rate * (query - now))
        for k in range(depth, 1, -1):
            for i in range(event, end):
                if removed[i]:
                    continue
                u, v = src[i], dst[i]
                if active[k - 1, v]:
                    active[k, u] = True
                    for col in range(right.shape[-1]):
                        delta[k, u, col] += w * delta[k - 1, v, col]
                if active[k - 1, u]:
                    active[k, v] = True
                    for col in range(right.shape[-1]):
                        delta[k, v, col] += w * delta[k - 1, u, col]
        while edit < len(et) and et[edit] == now:
            for k in range(1, depth + 1):
                active[k, eu[edit]], active[k, ev[edit]] = True, True
                for col in range(right.shape[-1]):
                    delta[k, eu[edit], col] += right[k - 1, 2 * edit, col]
                    delta[k, ev[edit], col] += right[k - 1, 2 * edit + 1, col]
            edit += 1
        event = end
    return delta


def masked_suffix(src, dst, stamp, removed, eu, ev, et, n, depth, query, rate):
    left = np.zeros((depth, n, 2 * len(eu)))
    event, edit, active = np.searchsorted(stamp, et[0], side="left"), 0, 0
    while event < len(src) or edit < len(et):
        now = min(
            stamp[event] if event < len(src) else np.inf,
            et[edit] if edit < len(et) else np.inf,
        )
        end = event
        while end < len(src) and stamp[end] == now:
            end += 1
        w = np.exp(-rate * (query - now))
        for k in range(depth - 1, 0, -1):
            for i in range(event, end):
                if removed[i]:
                    continue
                u, v = src[i], dst[i]
                for col in range(active):
                    left[k, u, col] += w * left[k - 1, v, col]
                    left[k, v, col] += w * left[k - 1, u, col]
        while edit < len(et) and et[edit] == now:
            left[0, eu[edit], 2 * edit] = 1.0
            left[0, ev[edit], 2 * edit + 1] = 1.0
            edit += 1
        active, event = 2 * edit, end
    return left


class CompactRevisionControls:
    def __init__(
        self,
        src,
        dst,
        stamp,
        base,
        depth,
        query_time,
        rate,
        candidates,
        nodes,
        support_mask=None,
    ):
        self.src = src
        self.dst = dst
        self.stamp = stamp
        self.base = base
        self.depth = depth
        self.query_time = query_time
        self.rate = rate
        self.candidates = candidates
        self.nodes = nodes
        self.support_mask = support_mask

        self.candidates = np.sort(np.asarray(self.candidates, np.int64))
        self.nodes = np.asarray(self.nodes, np.int64)
        h, self.prefix = forward_index(
            self.src,
            self.dst,
            self.stamp,
            self.base,
            self.depth,
            self.query_time,
            self.rate,
            self.candidates,
        )
        self.original = h[:, self.nodes].copy()
        self.offset = int(
            np.searchsorted(self.stamp, self.stamp[self.candidates[0]], "left")
        )
        if self.support_mask is None:
            self.local_event_ids = np.arange(self.offset, len(self.src))
        else:
            mask = np.asarray(self.support_mask, np.bool_)
            if len(mask) != len(self.src) or not np.all(mask[self.candidates]):
                raise ValueError("Support must contain every candidate")
            self.local_event_ids = np.flatnonzero(mask)
            self.local_event_ids = self.local_event_ids[
                self.local_event_ids >= self.offset
            ]
        vertices = np.unique(
            np.r_[
                self.src[self.local_event_ids],
                self.dst[self.local_event_ids],
                self.nodes,
            ]
        )
        self.local_n = len(vertices)
        self.local_src = np.searchsorted(vertices, self.src[self.local_event_ids])
        self.local_dst = np.searchsorted(vertices, self.dst[self.local_event_ids])
        self.local_t = self.stamp[self.local_event_ids]
        self.local_nodes = np.searchsorted(vertices, self.nodes)
        self.weights = np.exp(
            -self.rate * (self.query_time - self.stamp[self.candidates])
        )

    def evaluate(self, ids, method="differential"):
        ids = np.sort(np.asarray(ids, np.int64))
        if not len(ids):
            return self.original.copy()
        selected = np.searchsorted(self.candidates, ids)
        if (
            np.any(selected >= len(self.candidates))
            or not np.array_equal(self.candidates[selected], ids)
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Invalid deletion")
        removed = np.zeros(len(self.local_src), np.bool_)
        local_ids = np.searchsorted(self.local_event_ids, ids)
        removed[local_ids] = True
        right = np.empty((self.depth, 2 * len(ids), self.base.shape[1]))
        for pos, i in enumerate(selected):
            right[:, 2 * pos] = -self.weights[i] * self.prefix[i, :, 1]
            right[:, 2 * pos + 1] = -self.weights[i] * self.prefix[i, :, 0]
        common = (
            self.local_src,
            self.local_dst,
            self.local_t,
            removed,
            self.local_src[local_ids],
            self.local_dst[local_ids],
            self.stamp[ids],
        )
        if method == "differential":
            delta = masked_differential(
                *common, right, self.local_n, self.depth, self.query_time, self.rate
            )
            return self.original + delta[:, self.local_nodes]
        if method == "factored":
            left = masked_suffix(
                *common, self.local_n, self.depth, self.query_time, self.rate
            )
            return self.original + reconstruct_batch_rows(
                left, right, self.local_nodes, self.depth
            )
        raise ValueError("Unknown control")
