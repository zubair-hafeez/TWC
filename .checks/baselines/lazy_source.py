"""Stronger lazy comparator: batch all target messages for each requested source."""

import numpy as np
from .oracles import LazyForwardBackward, selected_ids
from baselines.revision_circuit import evaluate_circuit


def source_messages(src, dst, stamp, n, depth, query_time, rate, candidates, j):
    q = len(candidates)
    blocks = np.zeros((q, max(depth - 1, 0), 2, 2))
    if depth < 2:
        return blocks
    h = np.zeros((depth - 1, n, 2))
    event = candidates[j]
    h[0, src[event], 0] = 1.0
    h[0, dst[event], 1] = 1.0
    start = np.searchsorted(stamp, stamp[event], side="right")
    pos = j + 1
    while pos < q and stamp[candidates[pos]] == stamp[event]:
        pos += 1
    while start < len(src) and pos < q:
        end = start + 1
        while end < len(src) and stamp[end] == stamp[start]:
            end += 1
        while pos < q and candidates[pos] < end:
            i = candidates[pos]
            blocks[pos, :, 0] = h[:, src[i]]
            blocks[pos, :, 1] = h[:, dst[i]]
            pos += 1
        weight = np.exp(-rate * (query_time - stamp[start]))
        for k in range(depth - 2, 0, -1):
            for i in range(start, end):
                for a in range(2):
                    h[k, src[i], a] += weight * h[k - 1, dst[i], a]
                    h[k, dst[i], a] += weight * h[k - 1, src[i], a]
        start = end
    return blocks


class LazySourceForwardBackward(LazyForwardBackward):
    def evaluate(self, event_ids):
        selected = selected_ids(self.candidates, event_ids)
        b = len(selected)
        transfer = np.zeros((b, max(self.depth - 1, 0), 2, 2 * b))
        for earlier, j in enumerate(selected[:-1]):
            key = int(j)
            if key not in self.cache:
                self.cache[key] = source_messages(
                    self.src,
                    self.dst,
                    self.stamp,
                    self.n,
                    self.depth,
                    self.query_time,
                    self.rate,
                    self.candidates,
                    j,
                )
                self.cache_misses += 1
            for later, i in enumerate(selected[earlier + 1 :], start=earlier + 1):
                transfer[later, :, :, 2 * earlier : 2 * earlier + 2] = self.cache[key][
                    i
                ]
        return evaluate_circuit(
            self.prefix[selected],
            transfer,
            self.suffix[selected],
            self.weights[selected],
            np.arange(b),
            self.original,
        )
