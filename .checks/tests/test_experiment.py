"""Check that editable lists run comparable cases and only print results."""

import importlib

import experiment
import numpy as np
import pytest
from datasets import build_cases, history_bounds, load_dataset
from experiment import measure_case, run_method
from method_list import METHODS
from sb_lazy import SourceLazy

MODULES = [
    "etwc",
    "sb_lazy",
    "pair_lazy",
    "sparse_etwc",
    "packed_etwc",
    "sparse_per_source",
    "independent_effects",
    "adaptive",
]


def make_dataset(directory):
    path = directory / "processed" / "mooc"
    path.mkdir(parents=True)
    split = np.ones(24, dtype=np.int8)
    split[:6] = 0
    np.savez(
        path / "development.npz",
        src=np.zeros(24, dtype=np.int64),
        dst=np.arange(24) % 3 + 1,
        t=(np.arange(24) // 2).astype(float),
        split=split,
        n_source=np.array(1),
    )
    np.savez(
        path / "neural_decoder_development.npz",
        selected=np.array([12, 18, 22]),
        candidates=np.tile(np.array([1, 2, 3]), (3, 1)),
    )


def small_parameters():
    return {
        "queries": [1],
        "candidates": [3],
        "max_deletions": [2],
        "depth": [3],
        "dimension": [4],
        "repeats": [1],
        "seed": [2026],
        "history_limit": [8],
    }


def case_settings(parameters=None):
    values = parameters or small_parameters()
    settings = {name: entries[0] for name, entries in values.items()}
    settings["policies"] = ["source", "dispersed", "shared_walk"]
    return settings


def test_defaults_match_across_standalone_methods():
    for name in MODULES:
        module = importlib.import_module(name)
        assert module.DATASETS == ["mooc", "wikipedia", "lastfm"]
        assert module.POLICIES == ["source", "dispersed", "shared_walk"]
        assert module.PARAMETERS == {
            "queries": [4],
            "candidates": [16],
            "max_deletions": [3],
            "depth": [3],
            "dimension": [64],
            "repeats": [3],
            "seed": [2026],
            "history_limit": [0],
        }


def test_workloads_are_reproducible_and_respect_timestamp_boundaries(tmp_path):
    make_dataset(tmp_path)
    settings = case_settings()
    data = load_dataset("mooc", tmp_path, settings["dimension"], settings["seed"])
    first = list(build_cases(data, settings))
    second = list(build_cases(data, settings))
    assert len(first) == 3
    for left, right in zip(first, second):
        assert left["query_event"] == right["query_event"]
        assert left["hypotheses"] == right["hypotheses"]
        for a, b in zip(left["args"], right["args"]):
            np.testing.assert_array_equal(a, b)
        src, dst, stamp, base, depth, query_time, rate, candidates, nodes = left["args"]
        assert np.all(stamp < query_time)
        start, stop = history_bounds(
            data["stamp"], left["query_event"], settings["history_limit"]
        )
        assert start % 2 == 0 and stop % 2 == 0
        assert len(candidates) == 3 and len(left["hypotheses"]) == 6
        assert max(src.max(), dst.max(), nodes.max()) < len(base)


def test_none_runs_every_eligible_development_query(tmp_path):
    make_dataset(tmp_path)
    settings = case_settings()
    settings["queries"] = None
    settings["policies"] = ["source"]
    data = load_dataset("mooc", tmp_path, settings["dimension"], settings["seed"])
    cases = list(build_cases(data, settings))
    assert [case["query_event"] for case in cases] == [12, 18, 22]


def test_methods_print_results_without_writing_files(tmp_path, monkeypatch, capsys):
    make_dataset(tmp_path)
    monkeypatch.setattr(experiment, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    for method in METHODS.values():
        run_method(
            method, ["mooc"], ["source", "dispersed", "shared_walk"], small_parameters()
        )
    output = capsys.readouterr().out
    assert len([line for line in output.splitlines() if line.startswith("mooc ")]) == 24
    assert all(name in output for name in METHODS)
    assert "independent_effects (approximate)" in output
    assert "Sample_error" in output and "State_MiB" in output
    assert set(tmp_path.rglob("*")) == before


def test_parameter_lists_run_each_combination(tmp_path, monkeypatch, capsys):
    make_dataset(tmp_path)
    monkeypatch.setattr(experiment, "DEFAULT_DATA_DIR", tmp_path)
    parameters = small_parameters()
    parameters["depth"] = [2, 3]
    parameters["candidates"] = [2, 3]
    run_method(METHODS["eager_etwc"], ["mooc"], ["source"], parameters)
    output = capsys.readouterr().out
    rows = [line.split() for line in output.splitlines() if line.startswith("mooc ")]
    assert len(rows) == 4
    assert sorted(int(row[5]) for row in rows) == [3, 3, 6, 6]
    assert parameters["depth"] == [2, 3] and parameters["candidates"] == [2, 3]


def test_repeats_rebuild_lazy_state(tmp_path):
    make_dataset(tmp_path)
    settings = case_settings()
    data = load_dataset("mooc", tmp_path, settings["dimension"], settings["seed"])
    case = next(build_cases(data, settings))
    misses = []

    class RecordingLazy(SourceLazy):
        def batch(self, hypotheses, lengths, output):
            before = self.misses
            super().batch(hypotheses, lengths, output)
            misses.append((before, self.misses))

    measure_case(RecordingLazy, case["args"], case["hypotheses"], 3)
    assert len(misses) == 3
    assert all(before == 0 and after > 0 for before, after in misses)


@pytest.mark.parametrize("name", MODULES)
def test_method_main_uses_its_editable_lists(name, tmp_path, monkeypatch, capsys):
    make_dataset(tmp_path)
    module = importlib.import_module(name)
    monkeypatch.setattr(experiment, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "DATASETS", ["mooc"])
    monkeypatch.setattr(module, "POLICIES", ["source"])
    monkeypatch.setattr(module, "PARAMETERS", small_parameters())
    monkeypatch.chdir(tmp_path)
    module.main()
    assert (
        len(
            [
                line
                for line in capsys.readouterr().out.splitlines()
                if line.startswith("mooc ")
            ]
        )
        == 1
    )


def test_bad_settings_fail_before_loading_data():
    parameters = small_parameters()
    parameters["depth"] = []
    with pytest.raises(ValueError):
        run_method(METHODS["eager_etwc"], ["mooc"], ["source"], parameters)
    parameters = small_parameters()
    parameters["max_deletions"] = [4]
    with pytest.raises(ValueError):
        run_method(METHODS["eager_etwc"], ["mooc"], ["source"], parameters)
