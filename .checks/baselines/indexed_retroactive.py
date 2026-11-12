"""Compiled exact finite-walk correction prototype with a prefix index.

Index storage is O(m K d), explicitly measured. It is NOT constant-memory
streaming. Two suffix impulse columns per undirected edit replace d columns
of full replay. The current interface handles one edit against an immutable
indexed graph; no multi-edit independence assumption is made.
"""

import numpy as np


def replay_array(
    src, dst, stamp, base, depth, query_time, rate, skip_index=-1, make_prefix=False
):
    n, d = base.shape
    h = np.zeros((depth + 1, n, d), np.float64)
    h[0] = base
    # Pre-group values for each event endpoint. Store normalized at fixed T.
    prefix = np.empty((len(src) if make_prefix else 0, 2, depth, d), np.float64)
    i = 0
    while i < len(src) and stamp[i] < query_time:
        end = i + 1
        while end < len(src) and stamp[end] == stamp[i]:
            end += 1
        if make_prefix:
            for j in range(i, end):
                prefix[j, 0] = h[:depth, src[j]]
                prefix[j, 1] = h[:depth, dst[j]]
        weight = np.exp(-rate * (query_time - stamp[i]))
        for k in range(depth, 0, -1):
            for j in range(i, end):
                if j == skip_index:
                    continue
                u, v = src[j], dst[j]
                for col in range(d):
                    h[k, u, col] += weight * h[k - 1, v, col]
                    h[k, v, col] += weight * h[k - 1, u, col]
        i = end
    return h, prefix


def suffix_impulses(src, dst, stamp, n, depth, edit_index, query_time, rate):
    # Strictly later timestamp groups only; tied events cannot be traversed.
    u, v, s = src[edit_index], dst[edit_index], stamp[edit_index]
    h = np.zeros((depth, n, 2), np.float64)
    h[0, u, 0], h[0, v, 1] = 1.0, 1.0
    i = np.searchsorted(stamp, s, side="right")
    while i < len(src) and stamp[i] < query_time:
        end = i + 1
        while end < len(src) and stamp[end] == stamp[i]:
            end += 1
        w = np.exp(-rate * (query_time - stamp[i]))
        for k in range(depth - 1, 0, -1):
            for j in range(i, end):
                a, b = src[j], dst[j]
                for col in range(2):
                    h[k, a, col] += w * h[k - 1, b, col]
                    h[k, b, col] += w * h[k - 1, a, col]
        i = end
    return h


def reconstruct_rows(left, right, nodes, depth):
    out = np.zeros((depth + 1, len(nodes), right.shape[-1]), np.float64)
    for k in range(1, depth + 1):
        for j in range(k):
            for ni in range(len(nodes)):
                u = nodes[ni]
                for col in range(right.shape[-1]):
                    out[k, ni, col] += (
                        left[j, u, 0] * right[k - 1 - j, 0, col]
                        + left[j, u, 1] * right[k - 1 - j, 1, col]
                    )
    return out


def indexed_removal(src, dst, stamp, h, prefix, index, query_time, rate, nodes):
    depth = len(h) - 1
    left = suffix_impulses(src, dst, stamp, h.shape[1], depth, index, query_time, rate)
    right = -np.exp(-rate * (query_time - stamp[index])) * prefix[index, [1, 0]]
    # prefix shape (2, depth, d); convert to (depth, 2, d).
    right = right.transpose(1, 0, 2).copy()
    return h[:, nodes] + reconstruct_rows(left, right, np.asarray(nodes), depth)


def make_checkpoints(src, dst, stamp, base, depth, query_time, rate, interval):
    """Timestamp-boundary full-state checkpoints for a strong replay control."""
    n, d = base.shape
    h = np.zeros((depth + 1, n, d), np.float64)
    h[0] = base
    checkpoints = [h.copy()]
    offsets = [0]
    next_checkpoint = interval
    i = 0
    while i < len(src):
        end = i + 1
        while end < len(src) and stamp[end] == stamp[i]:
            end += 1
        w = np.exp(-rate * (query_time - stamp[i]))
        for k in range(depth, 0, -1):
            for j in range(i, end):
                u, v = src[j], dst[j]
                for col in range(d):
                    h[k, u, col] += w * h[k - 1, v, col]
                    h[k, v, col] += w * h[k - 1, u, col]
        i = end
        if i >= next_checkpoint and i < len(src):
            checkpoints.append(h.copy())
            offsets.append(i)
            next_checkpoint = i + interval
    return offsets, checkpoints


def checkpoint_replay(src, dst, stamp, initial, offset, query_time, rate, skip_index):
    h = initial.copy()
    depth = len(h) - 1
    i = offset
    while i < len(src):
        end = i + 1
        while end < len(src) and stamp[end] == stamp[i]:
            end += 1
        w = np.exp(-rate * (query_time - stamp[i]))
        for k in range(depth, 0, -1):
            for j in range(i, end):
                if j == skip_index:
                    continue
                u, v = src[j], dst[j]
                for col in range(h.shape[-1]):
                    h[k, u, col] += w * h[k - 1, v, col]
                    h[k, v, col] += w * h[k - 1, u, col]
        i = end
    return h


def differential_removal(src, dst, stamp, h, prefix, index, query_time, rate, nodes):
    """Exact local differential replay, skipping inactive degree-node states."""
    depth, n, dim = len(h) - 1, h.shape[1], h.shape[2]
    delta = np.zeros_like(h)
    active = np.zeros((depth + 1, n), np.bool_)
    u, v, s = src[index], dst[index], stamp[index]
    weight = np.exp(-rate * (query_time - s))
    for k in range(1, depth + 1):
        for col in range(dim):
            delta[k, u, col] = -weight * prefix[index, 1, k - 1, col]
            delta[k, v, col] = -weight * prefix[index, 0, k - 1, col]
        active[k, u], active[k, v] = True, True
    i = np.searchsorted(stamp, s, side="right")
    while i < len(src):
        end = i + 1
        while end < len(src) and stamp[end] == stamp[i]:
            end += 1
        w = np.exp(-rate * (query_time - stamp[i]))
        for k in range(depth, 1, -1):
            for j in range(i, end):
                a, b = src[j], dst[j]
                if active[k - 1, b]:
                    active[k, a] = True
                    for col in range(dim):
                        delta[k, a, col] += w * delta[k - 1, b, col]
                if active[k - 1, a]:
                    active[k, b] = True
                    for col in range(dim):
                        delta[k, b, col] += w * delta[k - 1, a, col]
        i = end
    out = np.empty((depth + 1, len(nodes), dim))
    for ni, node in enumerate(nodes):
        out[:, ni] = h[:, node] + delta[:, node]
    return out
