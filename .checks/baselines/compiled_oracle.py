"""Portable deletion oracle containing no raw event stream or base projections.

This separation establishes an evaluation dependency contract, not privacy:
derived arrays may still reveal information about their source history.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
from .revision_circuit import evaluate_circuit


class CompiledDeletionOracle:
    def __init__(
        self,
        candidates,
        nodes,
        prefix,
        transfers,
        suffix,
        weights,
        original,
        metadata_json="{}",
    ):
        self.candidates = candidates
        self.nodes = nodes
        self.prefix = prefix
        self.transfers = transfers
        self.suffix = suffix
        self.weights = weights
        self.original = original
        self.metadata_json = metadata_json

        for name in (
            "candidates",
            "nodes",
            "prefix",
            "transfers",
            "suffix",
            "weights",
            "original",
        ):
            dtype = np.int64 if name in ("candidates", "nodes") else np.float64
            value = np.array(getattr(self, name), dtype=dtype, copy=True, order="C")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        q = len(self.candidates)
        if self.prefix.ndim != 4:
            raise ValueError("Invalid prefix rank")
        _, k, _, d = self.prefix.shape
        s = len(self.nodes)
        shapes = {
            "prefix": (q, k, 2, d),
            "transfers": (q, max(k - 1, 0), 2, 2 * q),
            "suffix": (q, k, s, 2),
            "weights": (q,),
            "original": (k + 1, s, d),
        }
        if k < 1 or any(
            getattr(self, name).shape != shape for name, shape in shapes.items()
        ):
            raise ValueError("Incompatible compiled arrays")
        if np.any(self.candidates < 0) or np.any(np.diff(self.candidates) <= 0):
            raise ValueError("Candidate identifiers must be distinct and sorted")
        if any(not np.all(np.isfinite(getattr(self, name))) for name in shapes):
            raise ValueError("Non-finite compiled state")
        if not isinstance(json.loads(self.metadata_json), dict):
            raise ValueError("Metadata must be a JSON object")

    def nbytes(self):
        return sum(
            getattr(self, name).nbytes
            for name in (
                "candidates",
                "nodes",
                "prefix",
                "transfers",
                "suffix",
                "weights",
                "original",
            )
        )

    def evaluate(self, event_ids):
        ids = np.sort(np.asarray(event_ids, dtype=np.int64))
        selected = np.searchsorted(self.candidates, ids)
        if (
            np.any(selected >= len(self.candidates))
            or not np.array_equal(self.candidates[selected], ids)
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Deletion outside compiled candidate set")
        return evaluate_circuit(
            self.prefix,
            self.transfers,
            self.suffix,
            self.weights,
            selected,
            self.original,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents an existing audit artifact being replaced.
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=np.array(self.metadata_json),
                **{
                    name: getattr(self, name)
                    for name in (
                        "candidates",
                        "nodes",
                        "prefix",
                        "transfers",
                        "suffix",
                        "weights",
                        "original",
                    )
                },
            )
        return hashlib.sha256(path.read_bytes()).hexdigest()


def from_builder(builder, provenance=None):
    metadata = {
        "schema": 1,
        "query_time": builder.query_time,
        "rate": builder.rate,
        "depth": builder.depth,
        "provenance": provenance or {},
    }
    return CompiledDeletionOracle(
        builder.candidates,
        builder.nodes,
        builder.prefix,
        builder.transfers,
        builder.suffix,
        builder.weights,
        builder.original,
        json.dumps(metadata, sort_keys=True),
    )


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return CompiledDeletionOracle(
            *(
                data[name]
                for name in (
                    "candidates",
                    "nodes",
                    "prefix",
                    "transfers",
                    "suffix",
                    "weights",
                    "original",
                )
            ),
            metadata_json=str(data["metadata_json"]),
        )
