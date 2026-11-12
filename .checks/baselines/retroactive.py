"""Exact single-event corrections to finite temporal-walk projections.

For timestamp-group adjacency A_s, define the truncated polynomial
P(z)= product_s^reverse_time (I+z*w_s*A_s). At one timestamp s,
P_new-P_old=F_s(z)*z*w_s*DeltaA_s*P_before_s(z).
The coefficient at degree k is a sum of at most k rank-r terms when
DeltaA_s has rank r. This algebra follows product factorization; its novelty
as a learning contribution is not assumed.

All exponential weights here are relative to one fixed reference/query
time. The first implementation prioritizes independently verifiable
correctness; no asymptotic speed claim is attached to Python reference code.
"""

from itertools import groupby
from collections import Counter
import numpy as np


def projected_replay(events, base, depth, query_time, rate):
    """Reference full replay with dense per-node projected coefficients."""
    base = np.asarray(base, dtype=np.float64)
    h = np.zeros((depth + 1, *base.shape), dtype=np.float64)
    h[0] = base
    for t, group in groupby(sorted(events, key=lambda e: e[2]), key=lambda e: e[2]):
        if t >= query_time:
            break
        group = list(group)
        w = np.exp(-rate * (query_time - t))
        for k in range(depth, 0, -1):
            # Lower degree is still from strictly before this timestamp.
            for u, v, _ in group:
                h[k, u] += w * h[k - 1, v]
                h[k, v] += w * h[k - 1, u]
    return h


def event_correction_factors(
    events, base, depth, query_time, rate, edit, multiplier=-1.0
):
    """Factors of a single adjacency-weight change at an existing/new time.

    edit=(u,v,s); multiplier -1 removes one occurrence, +1 inserts one.
    The caller must ensure a removal corresponds to an existing occurrence.
    Inputs are the CURRENT graph before the edit. For repeated edits, factors
    must be computed against the updated graph, not blindly summed from the
    initial graph.
    """
    u, v, s = edit
    if u == v or not s < query_time:
        raise ValueError("Requires a non-self interaction strictly before query time")
    events = list(events)
    if multiplier < 0 and (u, v, s) not in events and (v, u, s) not in events:
        raise ValueError("Cannot remove an absent event")
    before = [event for event in events if event[2] < s]
    after = [event for event in events if s < event[2] < query_time]
    prefix = projected_replay(before, base, depth - 1, query_time, rate)
    impulses = np.zeros((len(base), 2))
    impulses[u, 0] = 1.0
    impulses[v, 1] = 1.0
    suffix = projected_replay(after, impulses, depth - 1, query_time, rate)
    # DeltaA=e_u*e_v^T+e_v*e_u^T. Prefix rows swap the endpoints.
    right = prefix[:, [v, u], :] * multiplier * np.exp(-rate * (query_time - s))
    return suffix, right


def materialize_delta(left, right, depth):
    result = np.zeros((depth + 1, left.shape[1], right.shape[2]))
    for k in range(1, depth + 1):
        for j in range(k):
            result[k] += left[j] @ right[k - 1 - j]
    return result


def query_delta(left, right, depth, nodes):
    """Correct only requested nodes without materializing all-node features."""
    nodes = np.asarray(nodes, dtype=np.int64)
    result = np.zeros((depth + 1, len(nodes), right.shape[2]))
    for k in range(1, depth + 1):
        for j in range(k):
            result[k] += left[j, nodes] @ right[k - 1 - j]
    return result


def apply_edits(events, edits):
    """Apply integer multiplicity edits, preserving duplicate observations."""
    counts = Counter((min(u, v), max(u, v), float(t)) for u, v, t in events)
    for u, v, t, change in edits:
        if u == v or int(change) != change:
            raise ValueError("Non-self edges and integer multiplicity edits required")
        key = (min(u, v), max(u, v), float(t))
        counts[key] += int(change)
        if counts[key] < 0:
            raise ValueError("Cannot remove more observations than exist")
    return sorted(
        [event for event, count in counts.items() for _ in range(count)],
        key=lambda event: event[2],
    )


def batch_correction(events, base, depth, query_time, rate, edits):
    """Exact telescoping correction, including cross-edit walk interactions.

    Prefixes always come from the original graph. Suffixes always come from
    the fully edited graph. Each changed path is assigned to its earliest
    changed timestamp factor in the telescoping identity.
    """
    updated = apply_edits(events, edits)
    result = np.zeros((depth + 1, *base.shape))
    for u, v, s, change in edits:
        if s >= query_time:
            continue
        prefix = projected_replay(
            [e for e in events if e[2] < s], base, depth - 1, query_time, rate
        )
        impulses = np.zeros((len(base), 2))
        impulses[u, 0], impulses[v, 1] = 1.0, 1.0
        suffix = projected_replay(
            [e for e in updated if s < e[2] < query_time],
            impulses,
            depth - 1,
            query_time,
            rate,
        )
        right = prefix[:, [v, u]] * change * np.exp(-rate * (query_time - s))
        result += materialize_delta(suffix, right, depth)
    return result
