"""Independent exact references for deletion hypotheses.

Nothing here imports the compiled oracles, the v1 forward/backward kernels,
Numba, or any correction routine. Each reference physically removes the
deleted event instances and recomputes the requested walk states.

* ``replay_rows``: snapshot-per-timestamp-group replay in pure Python/NumPy,
  restricted to the strict backward cone of the requested rows (computed
  here by its own pure-Python reachability). Dropping events outside that
  cone is exact: no walk of at most K edges ending at a requested row at the
  query time uses them, and deleting events only removes walks.
* ``polynomial_product``: the defining product Q_L(z)...Q_1(z) applied to the
  base projection, with Q_g(z) = I + z w_g A_g as dense matrices truncated
  at z^K. Tiny graphs only; it is the THEORY.txt definition written directly.
"""

from collections import defaultdict
import numpy as np


def _groups(stamp, order):
    """Yield lists of event indices sharing one timestamp, in time order."""
    current, last = [], None
    for e in order:
        t = float(stamp[e])
        if current and t != last:
            yield current
            current = []
        current.append(int(e))
        last = t
    if current:
        yield current


def cone_events(src, dst, stamp, nodes, depth):
    """Event indices on some strict temporal walk of <= depth edges ending at a requested row."""
    active = [
        set() for _ in range(depth)
    ]  # active[k]: k-edge walk from vertex to a row, later in time
    active[0].update(int(v) for v in nodes)
    keep = []
    groups = list(_groups(stamp, range(len(src))))
    for group in reversed(groups):
        for e in group:
            u, v = int(src[e]), int(dst[e])
            if any(u in a or v in a for a in active):
                keep.append(e)
        additions = [set() for _ in range(depth)]
        for e in group:
            u, v = int(src[e]), int(dst[e])
            for k in range(1, depth):
                if v in active[k - 1]:
                    additions[k].add(u)
                if u in active[k - 1]:
                    additions[k].add(v)
        for k in range(1, depth):
            active[k] |= additions[k]
    return np.array(sorted(keep), np.int64)


def replay_rows(
    src, dst, stamp, base, depth, query_time, rate, nodes, deleted=(), cone=None
):
    """States h[k, row, :] at query_time after physically removing ``deleted`` events."""
    if cone is None:
        cone = cone_events(src, dst, stamp, nodes, depth)
    removed = set(int(e) for e in deleted)
    events = [int(e) for e in cone if int(e) not in removed and stamp[e] < query_time]
    state = {}

    def row(v):
        if v not in state:
            x = np.zeros((depth + 1, base.shape[1]))
            x[0] = base[v]
            state[v] = x
        return state[v]

    for group in _groups(stamp, events):
        w = np.exp(-rate * (query_time - float(stamp[group[0]])))
        touched = {int(x) for e in group for x in (src[e], dst[e])}
        snapshot = {v: row(v).copy() for v in touched}  # pre-group values
        for e in group:
            u, v = int(src[e]), int(dst[e])
            state[u][1:] += w * snapshot[v][:-1]
            state[v][1:] += w * snapshot[u][:-1]
    out = np.zeros((depth + 1, len(nodes), base.shape[1]))
    for r, v in enumerate(nodes):
        out[:, r] = row(int(v))
    return out


def polynomial_product(
    src, dst, stamp, base, depth, query_time, rate, nodes, deleted=()
):
    n = base.shape[0]
    removed = set(int(e) for e in deleted)
    events = [e for e in range(len(src)) if e not in removed and stamp[e] < query_time]
    coef = np.zeros((depth + 1, n, n))
    coef[0] = np.eye(n)
    for group in _groups(stamp, events):
        a = np.zeros((n, n))
        for e in group:
            a[src[e], dst[e]] += 1.0
            a[dst[e], src[e]] += 1.0
        a *= np.exp(-rate * (query_time - float(stamp[group[0]])))
        new = coef.copy()
        for k in range(1, depth + 1):
            new[k] += a @ coef[k - 1]
        coef = new
    return np.einsum("kij,jd->kid", coef, base)[:, nodes]


def error_report(actual, reference):
    """Absolute and normwise-relative errors with an explicit near-zero convention.

    relative = max|a-b| / max|b| when max|b| > 0; when the reference is
    exactly zero, relative is reported as None and only the absolute error
    is meaningful.
    """
    actual, reference = np.asarray(actual), np.asarray(reference)
    absolute = float(np.max(np.abs(actual - reference))) if actual.size else 0.0
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    return {
        "max_abs_error": absolute,
        "reference_max_abs": scale,
        "normwise_relative_error": absolute / scale if scale > 0 else None,
    }
