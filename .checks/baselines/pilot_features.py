"""Compiled feature screening. All observations at one time are revealed together."""

import numpy as np


def _decay_node(h, times, node, t, rate):
    for k in range(1, h.shape[2]):
        factor = np.exp(-rate * k * (t - times[node]))
        h[0, node, k] *= factor
        h[1, node, k] *= factor
    times[node] = t


def stream_features(
    src,
    dst,
    stamp,
    pair_id,
    n_pairs,
    base,
    selected,
    candidates,
    candidate_pair,
    depth,
    rate,
):
    """Return recency baseline + all-walk and cavity Gram features.

    base is one fixed Gaussian projection shared by both operators.
    Positive is candidate column zero; negative candidates were fixed before
    fitting. No labels are used in state updates. Costs include both operators.
    """
    n, dim = base.shape
    h = np.zeros((2, n, depth + 1, dim), np.float64)
    h[0, :, 0] = base
    h[1, :, 0] = base
    c = np.zeros((n_pairs, 2, depth + 1, dim), np.float64)
    times = np.zeros(n, np.float64)
    pair_times = np.zeros(n_pairs, np.float64)
    count = np.zeros(n, np.int64)
    last = np.full(n, -1.0, np.float64)
    edge_count = np.zeros(n_pairs, np.int64)
    edge_last = np.full(n_pairs, -1.0, np.float64)
    gramdim = (2 * (depth + 1)) * (2 * (depth + 1) + 1) // 2
    features = np.zeros(
        (len(selected), candidates.shape[1], 8 + 2 * gramdim), np.float32
    )
    observed = np.zeros(len(selected), np.bool_)
    qptr = 0
    start = 0
    while start < len(src):
        end = start + 1
        while end < len(src) and stamp[end] == stamp[start]:
            end += 1
        t = stamp[start]
        while qptr < len(selected) and selected[qptr] < end:
            index = selected[qptr]
            u = src[index]
            _decay_node(h, times, u, t, rate)
            for j in range(candidates.shape[1]):
                v, p = candidates[qptr, j], candidate_pair[qptr, j]
                _decay_node(h, times, v, t, rate)
                ec = 0 if p < 0 else edge_count[p]
                el = -1.0 if p < 0 else edge_last[p]
                features[qptr, j, 0] = np.log1p(count[u])
                features[qptr, j, 1] = np.log1p(count[v])
                features[qptr, j, 2] = np.log1p(ec)
                features[qptr, j, 3] = 1.0 if ec > 0 else 0.0
                features[qptr, j, 4] = (
                    np.log1p((t - last[u]) * 1000) if last[u] >= 0 else 20.0
                )
                features[qptr, j, 5] = (
                    np.log1p((t - last[v]) * 1000) if last[v] >= 0 else 20.0
                )
                features[qptr, j, 6] = np.log1p((t - el) * 1000) if el >= 0 else 20.0
                features[qptr, j, 7] = np.log1p(count[u] * count[v])
                for mode in range(2):
                    cursor = 8 + mode * gramdim
                    for a in range(2 * (depth + 1)):
                        na = u if a < depth + 1 else v
                        ka = a % (depth + 1)
                        for b in range(a, 2 * (depth + 1)):
                            nb = u if b < depth + 1 else v
                            kb = b % (depth + 1)
                            val = 0.0
                            for d in range(dim):
                                val += h[mode, na, ka, d] * h[mode, nb, kb, d]
                            features[qptr, j, cursor] = np.log1p(max(val, 0.0))
                            cursor += 1
                if j == 0:
                    observed[qptr] = ec > 0
            qptr += 1
        # Bring every participating node and pair to t before constructing
        # any new contribution. Times may already have been advanced by queries.
        for i in range(start, end):
            u, v, p = src[i], dst[i], pair_id[i]
            _decay_node(h, times, u, t, rate)
            _decay_node(h, times, v, t, rate)
            for k in range(1, depth + 1):
                c[p, :, k] *= np.exp(-rate * k * (t - pair_times[p]))
            pair_times[p] = t
        # A whole timestamp uses frozen lower levels; descending levels also
        # avoid traversing the same event twice.
        for k in range(depth, 0, -1):
            delta = np.empty((end - start, 2, 2, dim), np.float64)
            for i in range(start, end):
                u, v, p = src[i], dst[i], pair_id[i]
                for d in range(dim):
                    delta[i - start, 0, 0, d] = h[0, v, k - 1, d]
                    delta[i - start, 0, 1, d] = h[0, u, k - 1, d]
                    delta[i - start, 1, 0, d] = h[1, v, k - 1, d] - c[p, 1, k - 1, d]
                    delta[i - start, 1, 1, d] = h[1, u, k - 1, d] - c[p, 0, k - 1, d]
            for i in range(start, end):
                u, v, p = src[i], dst[i], pair_id[i]
                for d in range(dim):
                    for mode in range(2):
                        h[mode, u, k, d] += delta[i - start, mode, 0, d]
                        h[mode, v, k, d] += delta[i - start, mode, 1, d]
                    c[p, 0, k, d] += delta[i - start, 1, 0, d]
                    c[p, 1, k, d] += delta[i - start, 1, 1, d]
        for i in range(start, end):
            u, v, p = src[i], dst[i], pair_id[i]
            count[u] += 1
            count[v] += 1
            last[u] = t
            last[v] = t
            edge_count[p] += 1
            edge_last[p] = t
        start = end
    return features, observed, h, c
