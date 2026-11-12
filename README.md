# Run a method

Each method is a normal Python file. Edit its lists near the bottom, then run it.
Results are printed to the terminal; no result files are created.

Install dependencies once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run whichever method you want:

```bash
python -B etwc.py
python -B sb_lazy.py
python -B pair_lazy.py
python -B sparse_etwc.py
python -B independent_effects.py
python -B adaptive.py
python -B packed_etwc.py
python -B sparse_per_source.py
```

`-B` prevents Python from creating bytecode cache files.

Each file has `DATASETS`, `POLICIES`, and `PARAMETERS` lists. For example:

```python
DATASETS = ["mooc", "wikipedia", "lastfm"]
POLICIES = ["source", "dispersed", "shared_walk"]
PARAMETERS = {
    "queries": [4],  # [None] runs every eligible development query.
    "candidates": [16],
    "max_deletions": [3],
    "depth": [3],  # [2, 3, 4] runs all three depths.
    "dimension": [64],
    "repeats": [3],
    "seed": [2026],
    "history_limit": [0],  # 0 uses the full history before each query.
}
```

Every combination of parameter values is run. Keep the same lists in different
method files for a fair comparison. A quick run can use one query, four
candidates, maximum deletion size two, dimension eight, one repeat, and a
history limit of 128 events.

Each printed row reports a dataset, query, and policy, followed by median build,
evaluation, and total times; retained array storage in MiB; and the error for
one maximum-budget deletion checked against independent replay. That check is
outside the timed sections. Independent Effects is approximate. Memory excludes
input history and output buffers and is not peak process memory.

The defaults use a new deterministic development cohort and plain NumPy, so
these timings do not reproduce the paper's original JIT measurements.

`common.py` contains shared math, `datasets.py` loads data from `../datasets/`,
and `experiment.py` runs one method and prints its results. Internal verification
files are hidden in `.checks/`.
