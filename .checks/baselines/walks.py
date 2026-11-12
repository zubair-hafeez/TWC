"""Reference streaming walk features and an independent enumeration oracle.

These are mathematical development implementations, not a new-method claim.
Standard temporal walk projections follow TPNet's operator formulation.
Non-backtracking walk operators predate this project (Arrigo et al., 2022).
"""

from collections import defaultdict
from itertools import groupby
import numpy as np


class WalkState:
    """Strict-time walk sums; optionally exclude immediate node returns.

    H[k,u] = sum of decayed projected endpoints of reverse-time k-walks
    starting at u. A walk with times s_1,...,s_k has weight
    exp(-rate * sum(t - s_i)). C[k,u,v] is the corresponding sum restricted
    to first neighbor v. To prepend (u,v,t) to a non-backtracking walk,
    subtract C[k-1,v,u] from H[k-1,v].
    """

    def __init__(self, base, depth=3, rate=0.0, cavity=False):
        self.base = np.asarray(base, dtype=np.float64)
        self.depth, self.rate, self.cavity = depth, rate, cavity
        n, d = self.base.shape
        self.h = np.zeros((depth + 1, n, d), dtype=np.float64)
        self.h[0] = self.base
        self.node_time = np.zeros(n, dtype=np.float64)
        self.c = {}
        self.edge_time = {}
        self.now = -np.inf

    def _node(self, u, t):
        factor = np.exp(
            -self.rate * (t - self.node_time[u]) * np.arange(self.depth + 1)
        )
        return self.h[:, u] * factor[:, None]

    def _edge(self, u, v, t):
        if (u, v) not in self.c:
            return np.zeros((self.depth + 1, self.base.shape[1]))
        factor = np.exp(
            -self.rate * (t - self.edge_time[u, v]) * np.arange(self.depth + 1)
        )
        return self.c[u, v] * factor[:, None]

    def query(self, nodes, t):
        if t < self.now:
            raise ValueError("A past-time query requires a separate historical state.")
        return np.stack([self._node(int(u), t) for u in nodes])

    def update(self, edges, t):
        """Observe one whole timestamp after all its predictions.

        Sequential calls at the same time are forbidden: observations from
        one timestamp must not become history for another event at that time.
        Duplicate interactions remain multiplicities; self-loops are excluded.
        """
        if t <= self.now:
            raise ValueError("Provide each distinct timestamp exactly once in order.")
        edges = [(int(u), int(v)) for u, v in edges]
        if any(u == v for u, v in edges):
            raise ValueError("This reference formulation excludes self-loops.")
        nodes = {u for edge in edges for u in edge}
        old_h = {u: self._node(u, t) for u in nodes}
        increments = defaultdict(lambda: np.zeros_like(next(iter(old_h.values()))))
        for u, v in edges:
            for a, b in ((u, v), (v, u)):
                delta = np.zeros_like(old_h[a])
                delta[1:] = old_h[b][:-1]
                if self.cavity:
                    delta[2:] -= self._edge(b, a, t)[1:-1]
                increments[a, b] += delta
        for u in nodes:
            self.h[:, u] = old_h[u]
            self.node_time[u] = t
        for (u, v), delta in increments.items():
            self.h[:, u] += delta
            if self.cavity:
                self.c[u, v] = self._edge(u, v, t) + delta
                self.edge_time[u, v] = t
        self.now = t


def enumerated_operator(events, n, depth, query_time, rate=0.0, cavity=False):
    """Independent brute-force DFS over events, for tiny graphs only."""
    out = np.zeros((depth + 1, n, n), dtype=np.float64)
    out[0] = np.eye(n)
    histories = defaultdict(list)
    for u, v, t in events:
        if t >= query_time:
            continue
        histories[u].append((v, t))
        histories[v].append((u, t))

    def visit(root, current, previous, latest, length, weight):
        if length:
            out[length, root, current] += weight
        if length == depth:
            return
        for other, event_time in histories[current]:
            if event_time >= latest or (cavity and other == previous):
                continue
            visit(
                root,
                other,
                current,
                event_time,
                length + 1,
                weight * np.exp(-rate * (query_time - event_time)),
            )

    for root in range(n):
        visit(root, root, -1, query_time, 0, 1.0)
    return out


def replay(events, base, depth=3, rate=0.0, cavity=False):
    state = WalkState(base, depth, rate, cavity)
    for t, group in groupby(sorted(events, key=lambda e: e[2]), key=lambda e: e[2]):
        state.update([(u, v) for u, v, _ in group], t)
    return state
