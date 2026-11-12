import itertools
import numpy as np
import pytest
from baselines.revision_circuit import DeletionCircuit
from baselines.compiled_oracle import CompiledDeletionOracle, from_builder, load
from baselines.walks import enumerated_operator


def test_saved_oracle_evaluates_without_the_history_or_projection(tmp_path):
    events = [(0, 1, 1.0), (1, 2, 2.0), (0, 1, 3.0), (1, 2, 4.0)]
    a, base = np.array(events), np.eye(3)
    src, dst, t = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2].copy()
    builder = DeletionCircuit(
        src, dst, t, base, 3, 5.0, 0.2, np.array([0, 3]), np.array([2])
    )
    oracle = from_builder(builder, {"purpose": "unit test"})
    path = tmp_path / "oracle.npz"
    assert len(oracle.save(path)) == 64
    with pytest.raises(FileExistsError):
        oracle.save(path)
    del oracle, builder
    # Deliberately destroy every raw input after compilation and serialization.
    src[:] = -1
    dst[:] = -1
    t[:] = np.nan
    base[:] = np.nan
    restored = load(path)
    assert not hasattr(restored, "src")
    assert not hasattr(restored, "base")
    for size in range(3):
        for removed in itertools.combinations([0, 3], size):
            changed = [e for i, e in enumerate(events) if i not in removed]
            expected = enumerated_operator(changed, 3, 3, 5.0, 0.2)[:, [2]]
            np.testing.assert_allclose(restored.evaluate(removed), expected, atol=1e-12)
    with pytest.raises(ValueError):
        restored.evaluate([1])
    with pytest.raises(ValueError):
        restored.original[0, 0, 0] = 1.0
