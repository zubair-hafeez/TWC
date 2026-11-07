"""Run one method and print its results. Nothing is written to disk."""

import sys
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from common import encode
from datasets import DATASET_NAMES, DEFAULT_DATA_DIR, build_cases, load_dataset
from threadpoolctl import threadpool_limits

# Keep the independent correctness reference out of the visible code folder.
sys.path.insert(0, str(Path(__file__).resolve().parent / ".checks"))


def measure_case(method_class, args, hypotheses, repeats):
    from independent_reference import replay_rows

    encoded, lengths = encode(args[-2], hypotheses)
    # Check one maximum-budget deletion outside the timed sections.
    expected = replay_rows(*args[:7], nodes=args[-1], deleted=hypotheses[-1])
    build_times, evaluation_times, total_times, errors = [], [], [], []
    state_bytes = 0
    for _ in range(repeats):
        start = perf_counter()
        method = method_class(*args)
        build_seconds = perf_counter() - start
        output = method.out_buffer(len(hypotheses))
        start = perf_counter()
        method.batch(encoded, lengths, output)
        evaluation_seconds = perf_counter() - start
        error = float(np.max(np.abs(output[-1] - expected), initial=0.0))
        scale = float(np.max(np.abs(expected), initial=0.0))
        if not np.isfinite(output).all():
            raise ArithmeticError(f"{method.name} produced non-finite states")
        if method.exact and error > 1e-8 + 1e-9 * scale:
            raise AssertionError(
                f"{method.name} failed the replay check: error={error}"
            )
        # Retained array storage, excluding input history and output buffers.
        state_bytes = sum(
            value.nbytes
            for name, value in vars(method).items()
            if isinstance(value, np.ndarray)
            and name not in {"src", "dst", "stamp", "candidates", "nodes"}
        )
        build_times.append(build_seconds)
        evaluation_times.append(evaluation_seconds)
        total_times.append(build_seconds + evaluation_seconds)
        errors.append(error)
        del method, output
    return (
        median(build_times),
        median(evaluation_times),
        median(total_times),
        state_bytes / 1024**2,
        max(errors),
    )


def run_method(method_class, dataset_names, policies, parameters):
    """Try every combination of the values in the editable parameter lists."""
    required = {
        "queries",
        "candidates",
        "max_deletions",
        "depth",
        "dimension",
        "repeats",
        "seed",
        "history_limit",
    }
    with threadpool_limits(limits=1):
        for values in product(*parameters.values()):
            settings = dict(zip(parameters, values))
            for name in (
                "queries",
                "candidates",
                "max_deletions",
                "depth",
                "dimension",
                "repeats",
            ):
                value = settings[name]
                if name == "queries" and value is None:
                    continue
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must contain positive integers")
            if any(
                not isinstance(settings[name], int) or settings[name] < 0
                for name in ("seed", "history_limit")
            ):
                raise ValueError(
                    "seed and history_limit must contain nonnegative integers"
                )
            if settings["max_deletions"] > settings["candidates"]:
                raise ValueError("max_deletions cannot exceed candidates")
            note = " (approximate)" if not method_class.exact else ""
            print(f"\n{method_class.name}{note}: {settings}", flush=True)
            print(
                "Dataset    Query    Policy       Events    q   Hyp   Build_s    Eval_s   Total_s  State_MiB  Sample_error",
                flush=True,
            )
            settings["policies"] = policies
            for name in dataset_names:
                data = load_dataset(
                    name, DEFAULT_DATA_DIR, settings["dimension"], settings["seed"]
                )
                for case in build_cases(data, settings):
                    build, evaluate, total, memory, error = measure_case(
                        method_class,
                        case["args"],
                        case["hypotheses"],
                        settings["repeats"],
                    )
                    print(
                        f"{name:<10} {case['query_event']:>7}  {case['policy']:<12} "
                        f"{len(case['args'][0]):>7} {len(case['args'][-2]):>4} "
                        f"{len(case['hypotheses']):>5} {build:>9.4f} {evaluate:>9.4f} "
                        f"{total:>9.4f} {memory:>10.3f} {error:>13.3g}",
                        flush=True,
                    )
