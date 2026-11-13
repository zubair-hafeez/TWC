"""Load the retained datasets and build identical workloads for every method."""

import itertools
from pathlib import Path

import numpy as np
from common import group_boundaries, propagate_reachability

DATASET_NAMES = ("mooc", "wikipedia", "lastfm")
DECAY_RATES = {"mooc": 10.0, "wikipedia": 1.0, "lastfm": 10.0}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def load_dataset(name, data_dir, dimension, seed):
    """Load the retained dataset and generate a reproducible base feature vector."""

    ## Path to the processed dataset directory.
    ## TODO: check if needs to changes.
    directory = Path(data_dir) / "processed" / name



   ## Loads data from the .npz files in the processed dataset directory.
    with np.load(directory / "development.npz", allow_pickle=False) as data:
        # Source node ids of the historical events.
        src = data["src"]
        # Destination node ids of the historical events.
        dst = data["dst"]
        # Timestamps of the historical events, 
        stamp = data["t"].astype(np.float64)
        # Split label: 0 = fit, 1 = development.
        split = data["split"]

    ## Load query data. 
    with np.load(directory / "neural_decoder_development.npz", allow_pickle=False) as data:
        ## IDs pointinmg to main data
        ## they represent querries. 
        selected = data["selected"]

        ## Candidate destination node IDs for each query. 
        # They are considered durring eval or sampling under dispersed and shared walk policy.
        destinations = data["candidates"]

    ## Filter timestamps that belong to fit split.
    fit_times = stamp[split == 0]

    ## Normalize timestamps to [0, 1], min max scaling.
    stamp = (stamp - fit_times.min()) / (fit_times.max() - fit_times.min())


    ## Max node id in all data. 
    node_count = int(max(src.max(), dst.max(), destinations.max())) + 1

    ## Random seed generator.
    rng = np.random.default_rng( np.random.SeedSequence([seed, DATASET_NAMES.index(name)]))

    ## Random base feature vector for each node,  total nodes X 64
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
    stop = int(np.searchsorted(stamp, stamp[event], side="left"))
    start = max(0, stop - history_limit) if history_limit else 0
    if start < stop:
        # A window starts at a complete timestamp group, never partway through one.
        start = int(np.searchsorted(stamp, stamp[start], side="left"))
    return start, stop


def query_positions(data, count, candidate_count, history_limit, seed):
    """Choose a reproducible development cohort with enough source candidates."""

    # Filter queries ids that belong to development split.
    available = np.flatnonzero(data["split"][data["selected"]] == 1)

    # 
    rng = np.random.default_rng(np.random.SeedSequence([seed, DATASET_NAMES.index(data["name"]), 1]))
    chosen = []
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
    if count is None and chosen:
        return sorted(chosen)
    raise ValueError(
        f"{data['name']}: only {len(chosen)} eligible queries; reduce queries or candidates, or increase history_limit"
    )


def backward_cone(src, dst, stamp, nodes, depth, node_count):
    reachable = np.zeros((depth, node_count, 1), dtype=bool)
    reachable[0, nodes, 0] = True
    included = np.zeros(len(src), dtype=bool)
    boundaries = group_boundaries(stamp)
    for group in range(len(boundaries) - 2, -1, -1):
        start, stop = boundaries[group : group + 2]
        included[start:stop] = reachable[:, src[start:stop], 0].any(axis=0) | reachable[
            :, dst[start:stop], 0
        ].any(axis=0)
        propagate_reachability(reachable, src, dst, start, stop)
    return np.flatnonzero(included)


def shared_walk_candidates(src, dst, stamp, nodes, depth, count, cone, query_time):
    """Collect a topology-only union of strict backward paths to the query rows."""
    endpoints = np.column_stack((src, dst)).ravel()
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
        stop = np.searchsorted(stamp[events], before, side="left")
        # Two branches per node give overlapping paths without numerical selection.
        for event in events[max(0, stop - 2) : stop][::-1]:
            chosen.add(int(event))
            if remaining > 1:
                other = dst[event] if src[event] == node else src[event]
                pending.append((int(other), stamp[event], remaining - 1))
            if len(chosen) == count:
                break
    for event in cone[::-1]:
        if len(chosen) == count:
            break
        chosen.add(int(event))
    return np.array(sorted(chosen), dtype=np.int64)


def build_cases(data, settings):
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
        vertices = np.unique(np.r_[src, dst, nodes])
        src = np.searchsorted(vertices, src)
        dst = np.searchsorted(vertices, dst)
        nodes = np.searchsorted(vertices, nodes)
        base = data["base"][vertices]
        query_time = float(data["stamp"][event])
        cone = None
        if any(policy != "source" for policy in settings["policies"]):
            cone = backward_cone(src, dst, stamp, nodes, settings["depth"], len(base))
        for policy in settings["policies"]:
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
            hypotheses = [
                ids
                for budget in range(1, settings["max_deletions"] + 1)
                for ids in itertools.combinations(candidates.tolist(), budget)
            ]
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
