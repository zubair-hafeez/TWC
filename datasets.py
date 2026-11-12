"""Load the retained datasets and build identical workloads for every method."""
# This file turns stored graph data into matching workloads for all methods.
# It chooses a query, keeps only its earlier events, selects removable events,
# and lists the removal combinations to try. Fixed random seeds make these
# choices repeatable, so different methods receive the same inputs.

import itertools
from pathlib import Path

import numpy as np
from common import group_boundaries, propagate_reachability

# These names also give each dataset a stable number for random seeds.
# DECAY_RATES controls how quickly older events lose weight for each dataset.
# The default data folder sits beside TWC; this code only names that location.
DATASET_NAMES = ("mooc", "wikipedia", "lastfm")
DECAY_RATES = {"mooc": 10.0, "wikipedia": 1.0, "lastfm": 10.0}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def load_dataset(name, data_dir, dimension, seed):
    # Load the event history and the saved query choices for one dataset.
    # The history contains two endpoint IDs, a time, and a split label per event.
    # The second archive lists query event IDs and their destination choices.
    # allow_pickle=False reads array data without loading pickled Python objects.
    directory = Path(data_dir) / "processed" / name
    with np.load(directory / "development.npz", allow_pickle=False) as data:
        src = data["src"]
        dst = data["dst"]
        stamp = data["t"].astype(np.float64)
        split = data["split"]
    with np.load(
        directory / "neural_decoder_development.npz", allow_pickle=False
    ) as data:
        selected = data["selected"]
        destinations = data["candidates"]
    # Check that all event fields line up and that the history is sorted.
    # Saved query IDs must also point to real events and have destination rows.
    if not (len(src) == len(dst) == len(stamp) == len(split)) or np.any(
        np.diff(stamp) < 0
    ):
        raise ValueError(f"{name}: malformed or unsorted history")
    if (
        np.any(selected < 0)
        or np.any(selected >= len(src))
        or len(selected) != len(destinations)
    ):
        raise ValueError(f"{name}: query records do not match the history")
    # Use only split 0 to set the time scale. Its earliest and latest times
    # become 0 and 1; later splits may therefore have times above 1.
    # A nonzero range is needed to avoid division by zero.
    fit_times = stamp[split == 0]
    if len(fit_times) == 0 or fit_times.max() <= fit_times.min():
        raise ValueError(f"{name}: no usable fit-time range for normalization")
    stamp = (stamp - fit_times.min()) / (fit_times.max() - fit_times.min())
    node_count = int(max(src.max(), dst.max(), destinations.max())) + 1
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, DATASET_NAMES.index(name)])
    )
    # Give each node a repeatable random feature row. Dividing by the square
    # root of the feature count keeps the typical row size comparable when
    # the requested number of features changes.
    base = rng.normal(size=(node_count, dimension)) / np.sqrt(dimension)
    return {
        "name": name,
        "src": src,
        "dst": dst,
        "stamp": stamp,
        "split": split,
        "selected": selected,
        "destinations": destinations,
        "base": base,
    }


def history_bounds(stamp, event, history_limit):
    # Choose the history slice for a query event.
    # The left search stops before every event at the query's timestamp.
    # A positive history_limit keeps a recent window; zero keeps all earlier events.
    # If the window cuts a timestamp group, move its start back to keep the group.
    stop = int(np.searchsorted(stamp, stamp[event], side="left"))
    start = max(0, stop - history_limit) if history_limit else 0
    if start < stop:
        # A window starts at a complete timestamp group, never partway through one.
        start = int(np.searchsorted(stamp, stamp[start], side="left"))
    return start, stop


def query_positions(data, count, candidate_count, history_limit, seed):
    """Choose a reproducible development cohort with enough source candidates."""
    # Choose query records from the development split, marked with label 1.
    # Try them in a seeded random order and keep only queries whose earlier
    # history has enough events touching the query's source node.
    # This supplies a common eligible query pool for the candidate policies.
    available = np.flatnonzero(data["split"][data["selected"]] == 1)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, DATASET_NAMES.index(data["name"]), 1])
    )
    chosen = []
    # Check a query against its own history window, since different queries
    # can have different numbers of usable earlier events.
    for position in rng.permutation(available):
        event = data["selected"][position]
        start, stop = history_bounds(data["stamp"], event, history_limit)
        source = data["src"][event]
        incident = (data["src"][start:stop] == source) | (
            data["dst"][start:stop] == source
        )
        if np.count_nonzero(incident) >= candidate_count:
            chosen.append(int(position))
            if len(chosen) == count:
                return sorted(chosen)
    # None asks for every eligible query. For a fixed count, failing to find
    # enough is reported clearly instead of silently shrinking the workload.
    if count is None and chosen:
        return sorted(chosen)
    raise ValueError(
        f"{data['name']}: only {len(chosen)} eligible queries; reduce queries or candidates, or increase history_limit"
    )


def backward_cone(src, dst, stamp, nodes, depth, node_count):
    # Find earlier events that could matter to the requested output nodes.
    # Start at those nodes and move backwards in time, tracking up to depth - 1
    # later steps. An event is included if either endpoint can finish such a walk.
    # This uses graph connections only, so small or zero numeric weights do not
    # hide a possible route.
    reachable = np.zeros((depth, node_count, 1), dtype=bool)
    reachable[0, nodes, 0] = True
    included = np.zeros(len(src), dtype=bool)
    boundaries = group_boundaries(stamp)
    for group in range(len(boundaries) - 2, -1, -1):
        start, stop = boundaries[group : group + 2]
        # Read reachability before updating this timestamp. Otherwise one event
        # could reach the query through another event at exactly the same time.
        included[start:stop] = reachable[:, src[start:stop], 0].any(axis=0) | reachable[
            :, dst[start:stop], 0
        ].any(axis=0)
        propagate_reachability(reachable, src, dst, start, stop)
    return np.flatnonzero(included)


def shared_walk_candidates(src, dst, stamp, nodes, depth, count, cone, query_time):
    """Collect a topology-only union of strict backward paths to the query rows."""
    # Choose candidates from short backward walks ending at the query nodes.
    # The search follows up to two recent earlier events at each visited node.
    # This tends to give removals that share walks and may affect one another.
    # Only connections and times guide the choice, not calculated feature values.
    endpoints = np.column_stack((src, dst)).ravel()
    # Group endpoint entries by node while keeping each node's event order.
    # Each event supplied two endpoint entries, so integer division by two
    # turns a sorted endpoint position back into its event ID.
    order = np.argsort(endpoints, kind="stable")
    incidents = order // 2
    node_count = (
        int(max(src.max(initial=0), dst.max(initial=0), nodes.max(initial=0))) + 1
    )
    offsets = np.r_[0, np.cumsum(np.bincount(endpoints, minlength=node_count))]
    pending = [(int(node), query_time, depth) for node in reversed(nodes)]
    chosen = set()
    while pending and len(chosen) < count:
        node, before, remaining = pending.pop()
        events = incidents[offsets[node] : offsets[node + 1]]
        # Use only events strictly earlier than the current step.
        # Moving to an earlier timestamp at every step prevents equal-time walks.
        stop = np.searchsorted(stamp[events], before, side="left")
        # Two branches per node give overlapping paths without numerical selection.
        for event in events[max(0, stop - 2) : stop][::-1]:
            chosen.add(int(event))
            if remaining > 1:
                other = dst[event] if src[event] == node else src[event]
                pending.append((int(other), stamp[event], remaining - 1))
            if len(chosen) == count:
                break
    # If the branching search found too few candidates, fill the remaining
    # places from recent events in the query's backward cone. The set avoids
    # duplicates when the same event is found along more than one walk.
    for event in cone[::-1]:
        if len(chosen) == count:
            break
        chosen.add(int(event))
    return np.array(sorted(chosen), dtype=np.int64)


def build_cases(data, settings):
    # Build one workload for each chosen query and candidate policy.
    # Every yielded dictionary holds the method inputs and the removal sets.
    # Using a generator lets the experiment consume one case at a time instead
    # of keeping every query's arrays and combinations in memory.
    positions = query_positions(
        data,
        settings["queries"],
        settings["candidates"],
        settings["history_limit"],
        settings["seed"],
    )
    for position in positions:
        event = int(data["selected"][position])
        start, stop = history_bounds(data["stamp"], event, settings["history_limit"])
        src = data["src"][start:stop]
        dst = data["dst"][start:stop]
        stamp = data["stamp"][start:stop]
        nodes = np.r_[data["src"][event], data["destinations"][position]]
        # Replace the original node IDs with consecutive local IDs for this case.
        # Keep every history endpoint and requested node, and select the matching
        # base rows so the smaller arrays still describe the same graph.
        vertices = np.unique(np.r_[src, dst, nodes])
        src = np.searchsorted(vertices, src)
        dst = np.searchsorted(vertices, dst)
        nodes = np.searchsorted(vertices, nodes)
        base = data["base"][vertices]
        query_time = float(data["stamp"][event])
        cone = None
        # The source policy needs only events touching one node. Other policies
        # need the backward cone, so compute that shared list only when needed.
        if any(policy != "source" for policy in settings["policies"]):
            cone = backward_cone(src, dst, stamp, nodes, settings["depth"], len(base))
        for policy in settings["policies"]:
            # source selects recent events touching the query source.
            # dispersed draws distinct events across the backward cone.
            # shared_walk follows overlapping backward paths to the requested nodes.
            if policy == "source":
                candidates = np.flatnonzero((src == nodes[0]) | (dst == nodes[0]))[
                    -settings["candidates"] :
                ]
            elif policy == "dispersed":
                rng = np.random.default_rng(
                    np.random.SeedSequence([settings["seed"], event, 2])
                )
                candidates = np.sort(
                    rng.choice(cone, settings["candidates"], replace=False)
                )
            else:
                candidates = shared_walk_candidates(
                    src,
                    dst,
                    stamp,
                    nodes,
                    settings["depth"],
                    settings["candidates"],
                    cone,
                    query_time,
                )
            # List every candidate combination of sizes 1 through max_deletions.
            # A combination removes event instances by ID, so repeated interactions
            # between the same two nodes can still be removed separately.
            hypotheses = [
                ids
                for budget in range(1, settings["max_deletions"] + 1)
                for ids in itertools.combinations(candidates.tolist(), budget)
            ]
            # Keep this tuple in the constructor order expected by every method:
            # history, node features, walk depth, query time, decay, candidates, and
            # requested nodes. All methods can therefore consume exactly the same case.
            args = (
                src,
                dst,
                stamp,
                base,
                settings["depth"],
                query_time,
                DECAY_RATES[data["name"]],
                candidates,
                nodes,
            )
            yield {
                "dataset": data["name"],
                "query_event": event,
                "policy": policy,
                "args": args,
                "hypotheses": hypotheses,
            }
