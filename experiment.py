"""Run one method and print its results. Nothing is written to disk."""
# This is the shared experiment runner used by the method files.
# It builds each method, times its batch answers, and prints one result row.
# It also compares one removal set with a separate replay calculation.
# The runner itself prints results rather than saving result files.

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
    # Measure one fixed history, query, and list of removal sets.
    # Encode the removal IDs and calculate a reference answer before timing.
    # Each repeat creates a fresh method so cached work from a previous repeat
    # cannot make construction or the first batch appear faster.
    from independent_reference import replay_rows

    # args[-2] is the candidate event list and args[-1] is the query node list.
    # The last hypothesis is a maximum-size removal set because build_cases
    # lists combinations in increasing order of removal count.
    encoded, lengths = encode(args[-2], hypotheses)
    # Check one maximum-budget deletion outside the timed sections.
    expected = replay_rows(*args[:7], nodes=args[-1], deleted=hypotheses[-1])
    build_times, evaluation_times, total_times, errors = [], [], [], []
    state_bytes = 0
    # Measure setup and evaluation separately. The output array is allocated
    # between these timers, so its allocation is excluded from both times.
    for _ in range(repeats):
        start = perf_counter()
        method = method_class(*args)
        build_seconds = perf_counter() - start
        output = method.out_buffer(len(hypotheses))
        start = perf_counter()
        method.batch(encoded, lengths, output)
        evaluation_seconds = perf_counter() - start
        # Compare the last batch answer with the replay result using the largest
        # absolute difference. This is a sample correctness check, not a replay
        # check of every hypothesis; the finite-value check covers the whole batch.
        error = float(np.max(np.abs(output[-1] - expected), initial=0.0))
        scale = float(np.max(np.abs(expected), initial=0.0))
        if not np.isfinite(output).all():
            raise ArithmeticError(f"{method.name} produced non-finite states")
        # Exact methods may differ slightly because floating-point sums round.
        # Allow a small fixed error plus an amount scaled to the reference size.
        # The independent-effects approximation reports its error without this failure.
        if method.exact and error > 1e-8 + 1e-9 * scale:
            raise AssertionError(
                f"{method.name} failed the replay check: error={error}"
            )
        # Retained array storage, excluding input history and output buffers.
        # Count retained NumPy arrays directly stored on the method object.
        # Skip the shared input history and index arrays. This is array storage,
        # not a measurement of the Python process's total memory.
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
    # Use the middle timing across repeats to reduce the effect of one unusual
    # run. Convert bytes to MiB, where one MiB is 1024 squared bytes, and
    # report the largest observed sample error.
    return (
        median(build_times),
        median(evaluation_times),
        median(total_times),
        state_bytes / 1024**2,
        max(errors),
    )


def run_method(method_class, dataset_names, policies, parameters):
    """Try every combination of the values in the editable parameter lists."""
    # Run the chosen method for every combination of the supplied setting lists.
    # For example, two depth values and two dimension values give four settings.
    # Validate each chosen setting, load each dataset, and print the measurements
    # for its queries and candidate policies.
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
    # Limit numerical library thread pools to one thread while measuring.
    # This keeps hidden parallel work from changing the comparison.
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
                # Only queries accepts None, meaning all eligible queries.
                # The other counts must be positive integers; seed and history_limit may be zero.
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
            # Each dataset is loaded once for this setting, then build_cases supplies
            # its individual workloads. The printed columns show history size, candidate
            # count q, hypothesis count Hyp, times, stored array size, and sample error.
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
