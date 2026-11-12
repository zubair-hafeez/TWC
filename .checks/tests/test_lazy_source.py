import itertools
import numpy as np
import pytest
from baselines.lazy_source import LazySourceForwardBackward
from baselines.walks import enumerated_operator


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_source_batched_messages_against_enumeration(depth):
    rng = np.random.default_rng(26099)
    src = rng.integers(5, size=9)
    dst = (src + rng.integers(1, 5, size=9)) % 5
    stamp = np.array([0, 0, 1, 2, 3, 3, 3, 4, 5], float)
    base = rng.normal(size=(5, 3))
    candidates = np.array([0, 1, 3, 5, 8])
    obj = LazySourceForwardBackward(
        src, dst, stamp, base, depth, 6.0, 0.1, candidates, np.arange(5)
    )
    for b in range(6):
        for ids in itertools.combinations(candidates, b):
            events = [
                (int(u), int(v), float(t))
                for i, (u, v, t) in enumerate(zip(src, dst, stamp))
                if i not in ids
            ]
            target = enumerated_operator(events, 5, depth, 6.0, 0.1) @ base
            np.testing.assert_allclose(
                obj.evaluate(ids), target, atol=1e-10, rtol=1e-11
            )
    assert obj.cache_misses <= len(candidates) - 1
