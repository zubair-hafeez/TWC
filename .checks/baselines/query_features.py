"""Exact query-local recency updates and walk readout under candidate deletions."""

import numpy as np


def backward_cone(src, dst, stamp, n, depth, nodes):
    """Events that could affect a <=depth walk ending at a requested node."""
    active = np.zeros((depth, n), np.bool_)
    active[0, nodes] = True
    eligible = np.zeros(len(src), np.bool_)
    end = len(src)
    while end > 0:
        start = end - 1
        while start > 0 and stamp[start - 1] == stamp[end - 1]:
            start -= 1
        for i in range(start, end):
            eligible[i] = np.any(active[:, src[i]]) or np.any(active[:, dst[i]])
        for k in range(depth - 1, 0, -1):
            for i in range(start, end):
                u, v = src[i], dst[i]
                active[k, u] |= active[k - 1, v]
                active[k, v] |= active[k - 1, u]
        end = start
    return eligible


def group_histories(
    src,
    dst,
    stamp,
    node_map,
    query_source,
    target_map,
    candidate_map,
    n_nodes,
    n_targets,
    pool_size,
):
    groups = n_nodes + n_targets
    counts = np.zeros(groups, np.int64)
    # Keep q+1 latest events: at most q can be deleted.
    latest_t = np.full((groups, pool_size + 1), -1.0)
    latest_c = np.full((groups, pool_size + 1), -2, np.int64)
    impacts = np.zeros((pool_size, groups), np.int64)
    for i in range(len(src)):
        u, v = src[i], dst[i]
        group_ids = np.array(
            [
                node_map[u],
                node_map[v],
                n_nodes + target_map[v]
                if u == query_source and target_map[v] >= 0
                else -1,
            ]
        )
        for group in group_ids:
            if group >= 0:
                location = counts[group] % (pool_size + 1)
                counts[group] += 1
                latest_t[group, location] = stamp[i]
                latest_c[group, location] = candidate_map[i]
                if candidate_map[i] >= 0:
                    impacts[candidate_map[i], group] += 1
    return counts, latest_t, latest_c, impacts


def revised_groups(counts, latest_t, latest_c, impacts, selected):
    new_counts = counts.copy()
    deleted = np.zeros(len(impacts), np.bool_)
    for j in selected:
        deleted[j] = True
        new_counts -= impacts[j]
    last = np.full(len(counts), -1.0)
    for group in range(len(counts)):
        for pos in range(latest_t.shape[1]):
            candidate = latest_c[group, pos]
            if candidate == -1 or (candidate >= 0 and not deleted[candidate]):
                last[group] = max(last[group], latest_t[group, pos])
    return new_counts, last


def readout_features(h, counts, last, query_time):
    """Node rows must be [source, candidate_0, ..., candidate_C-1]."""
    k_count, n_nodes, dim = h.shape
    n_targets = n_nodes - 1
    gramdim = (2 * k_count) * (2 * k_count + 1) // 2
    out = np.zeros((n_targets, 8 + gramdim), np.float32)
    for target in range(n_targets):
        v, p = target + 1, n_nodes + target
        out[target, 0] = np.log1p(counts[0])
        out[target, 1] = np.log1p(counts[v])
        out[target, 2] = np.log1p(counts[p])
        out[target, 3] = 1.0 if counts[p] > 0 else 0.0
        for col, g in ((4, 0), (5, v), (6, p)):
            out[target, col] = (
                np.log1p((query_time - last[g]) * 1000) if last[g] >= 0 else 20.0
            )
        out[target, 7] = np.log1p(counts[0] * counts[v])
        cursor = 8
        for a in range(2 * k_count):
            na, ka = (0 if a < k_count else v), a % k_count
            for b in range(a, 2 * k_count):
                nb, kb = (0 if b < k_count else v), b % k_count
                value = 0.0
                for col in range(dim):
                    value += h[ka, na, col] * h[kb, nb, col]
                out[target, cursor] = np.log1p(max(value, 0.0))
                cursor += 1
    return out


class RecencyIndex:
    def __init__(self, src, dst, stamp, nodes, candidates, n):
        self.src = src
        self.dst = dst
        self.stamp = stamp
        self.nodes = nodes
        self.candidates = candidates
        self.n = n

        self.nodes = np.asarray(self.nodes, dtype=np.int64)
        self.candidates = np.asarray(self.candidates, dtype=np.int64)
        if len(np.unique(self.nodes)) != len(self.nodes):
            raise ValueError("Query node rows must be distinct")
        node_map = np.full(self.n, -1, np.int64)
        node_map[self.nodes] = np.arange(len(self.nodes))
        target_map = np.full(self.n, -1, np.int64)
        target_map[self.nodes[1:]] = np.arange(len(self.nodes) - 1)
        candidate_map = np.full(len(self.src), -1, np.int64)
        candidate_map[self.candidates] = np.arange(len(self.candidates))
        self.counts, self.latest_t, self.latest_c, self.impacts = group_histories(
            self.src,
            self.dst,
            self.stamp,
            node_map,
            self.nodes[0],
            target_map,
            candidate_map,
            len(self.nodes),
            len(self.nodes) - 1,
            len(self.candidates),
        )

    def features(self, h, selected, query_time):
        counts, last = revised_groups(
            self.counts,
            self.latest_t,
            self.latest_c,
            self.impacts,
            np.asarray(selected, np.int64),
        )
        return readout_features(h, counts, last, query_time)
