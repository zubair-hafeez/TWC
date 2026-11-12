# Internal verification

This hidden folder holds the independent replay implementation, earlier
reference implementations, and automated tests. These files are not experiment
entry points. The runner checks one maximum-budget deletion using independent replay outside measured sections.

From the code folder:

```bash
python -m pip install -r .checks/requirements.txt
python -B -m pytest -p no:cacheprovider -q .checks/tests
```

The original 177 tests check exactness, ties, duplicates, sparse pruning,
certificates, lazy caching, serialization, and the earlier reference methods.
Additional tests cover editable parameter lists, all-query selection, cold repeats, and print-only output.
