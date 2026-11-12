"""Shared array operations for the temporal walk comparison methods.

Events at the same timestamp always read the state from before that timestamp.
The methods share these operations so that their main difference is how they
prepare and store transfers between candidate deletions.
"""
# These helpers do the work shared by the different deletion methods.
# A node is a point in the graph. An event joins two nodes at a given time.
# A walk follows events in time order. Its degree is the number of events used.
# The arrays keep a separate sum for each degree and each feature number.
# A candidate is an event we are allowed to remove. A hypothesis is one set
# of removals to try. All events with the same time must read the old state.

import numpy as np


#src: start node, dst: ending node, stamp: time of the event, base: initial state or features of the nodes, 
# depth: Going back to k steps following time order. , query_time: time at which we want to evaluate the state of future node.,
# rate: Decides how quickly old events become less influential
#candidates:  Events that we select to remove
#  nodes: list of future nodes for which we want to evaluate the state.
def validate(src, dst, stamp, base, depth, query_time, rate, candidates, nodes):
    """Checking input arrays and convert them to the right NumPy types."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    stamp = np.asarray(stamp, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    candidates = np.sort(np.asarray(candidates, dtype=np.int64))
    nodes = np.asarray(nodes, dtype=np.int64)

    # Check array shapes before using them as node rows or event lists.
    # Also reject values that would make time decay or walk depth invalid.
    if base.ndim != 2 or any(a.ndim != 1 for a in (src, dst, stamp, candidates, nodes)):
        raise ValueError(
            "History and index arrays must be vectors; base must be a matrix"
        )
    if not isinstance(depth, (int, np.integer)) or depth < 1:
        raise ValueError("Depth must be a positive integer")
    if not np.isfinite(query_time) or not np.isfinite(rate) or rate < 0:
        raise ValueError(
            "Query time and decay rate must be finite; rate must be nonnegative"
        )
    
    if len(src) != len(dst) or len(src) != len(stamp):
        raise ValueError("History arrays must have the same length")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(stamp)):
        raise ValueError("States and timestamps must be finite")
    
    
    if np.any(np.diff(stamp) < 0) or np.any(stamp >= query_time):
        raise ValueError("History must be sorted and strictly before the query")
    if (
        # An event must join two different, existing nodes. Negative IDs could
        # otherwise select rows from the end of a NumPy array by mistake.
        np.any(src == dst)
        or np.any(src < 0)
        or np.any(dst < 0)
        or np.any(src >= len(base))
        or np.any(dst >= len(base))
    ):
        raise ValueError("History contains an invalid endpoint or a self edge")
    
    if (
        # Candidate events must be unique and within the history. Requested nodes
        # must exist too; repeated requested nodes are allowed by this check.
        len(np.unique(candidates)) != len(candidates)
        or np.any(candidates < 0)
        or np.any(candidates >= len(src))
        or np.any(nodes < 0)
        or np.any(nodes >= len(base))
    ):
        raise ValueError("Invalid candidate events or query nodes")
    return src, dst, stamp, base, candidates, nodes


def selected_ids(candidates, event_ids):
    """Convert event IDs to positions in the sorted candidate array."""
    # The saved tables use candidate positions, not original event IDs.
    # For example, if candidates is [4, 9], deleting event 9 means position 1.
    # Search the sorted list, then check that every requested ID really exists.
    # Reject repeated IDs so the same event cannot be removed twice.
    ids = np.sort(np.asarray(event_ids, dtype=np.int64))
    selected = np.searchsorted(candidates, ids)
    if (
        ids.ndim != 1
        or len(np.unique(ids)) != len(ids)
        or np.any(selected >= len(candidates))
        or not np.array_equal(candidates[selected], ids)
    ):
        raise ValueError("Deletion outside candidate set or duplicate deletion")
    return selected


def encode(candidates, hypotheses):
    """Encode a list of event-ID tuples once, outside the timed evaluation."""
    # Store many sets of removals in one rectangular integer array.
    # Each row holds positions in the candidate list. Short rows have extra zeros,
    # but lengths says how many entries are real, so padding is never a deletion.
    # Keeping this conversion separate lets timing focus on the calculation.
    candidates = np.sort(np.asarray(candidates, dtype=np.int64))
    width = max([len(ids) for ids in hypotheses] + [1])
    encoded = np.zeros((len(hypotheses), width), dtype=np.int64)
    lengths = np.zeros(len(hypotheses), dtype=np.int64)
    for row, ids in enumerate(hypotheses):
        positions = selected_ids(candidates, ids)
        encoded[row, : len(positions)] = positions
        lengths[row] = len(positions)
    return encoded, lengths

#Find groups of events that have the same time
def group_boundaries(stamp, start=0, stop=None):
   
    if stop is None:
        stop = len(stamp)
    if start >= stop:
        return np.array([start], dtype=np.int64)
    changes = np.flatnonzero(np.diff(stamp[start:stop])) + start + 1
    return np.r_[start, changes, stop]

#it propagates graph-walk information across all edges occurring at a particular time, 
# extending walks by exactly one step while preventing movement through multiple simultaneous edges.
def propagate(state, src, dst, start, stop, weight):
    """Add the effect of one timestamp's events to the next timestamp."""
    if state.shape[0] <= 1 or state.shape[-1] == 0:
        return
    if stop == start + 1:
        u, v = src[start], dst[start]
        to_u = weight * state[:-1, v]
        to_v = weight * state[:-1, u]
        state[1:, u] += to_u
        state[1:, v] += to_v
    else:
        u, v = src[start:stop], dst[start:stop]
        to_u = weight * state[:-1, v]
        to_v = weight * state[:-1, u]
        # add.at adds every contribution, including repeated node indices.
        # Ordinary indexed addition could lose contributions when endpoints repeat.
        np.add.at(state[1:], (slice(None), u), to_u)
        np.add.at(state[1:], (slice(None), v), to_v)


def propagate_reachability(reachable, src, dst, start, stop):
    """Boolean counterpart of propagate, independent of numerical weights."""
    # Do the same one-step update using True and False instead of numbers.
    # True means that a walk of this degree can reach this node and column.
    # Copies preserve the state from before this whole timestamp. Logical OR
    # keeps a reachable node True even when several events reach it.
    if reachable.shape[0] <= 1 or reachable.shape[-1] == 0:
        return
    u, v = src[start:stop], dst[start:stop]
    to_u = reachable[:-1, v].copy()
    to_v = reachable[:-1, u].copy()
    np.logical_or.at(reachable[1:], (slice(None), u), to_u)
    np.logical_or.at(reachable[1:], (slice(None), v), to_v)


def forward_index(src, dst, stamp, base, depth, query_time, rate, candidates):
    # Walk through the original history from oldest to newest.
    # state[0] contains the starting node features; higher degrees start at zero.
    # prefix saves the values at both endpoints just before each candidate event.
    # These saved values describe what that event would send before any deletion.
    state = np.zeros((depth + 1, len(base), base.shape[1]))
    state[0] = base
    prefix = np.empty((len(candidates), depth, 2, base.shape[1]))
    boundaries = group_boundaries(stamp)
    position = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        # Save every candidate in this timestamp before applying any event in it.
        # This keeps equal-time events out of one another's saved past.
        while position < len(candidates) and candidates[position] < stop:
            event = candidates[position]
            prefix[position, :, 0] = state[:depth, src[event]]
            prefix[position, :, 1] = state[:depth, dst[event]]
            position += 1
        weight = np.exp(-rate * (query_time - stamp[start]))
        propagate(state, src, dst, start, stop, weight)
    return state, prefix


def backward_query_index(
    src, dst, stamp, n, depth, query_time, rate, candidates, nodes
):
    # Work from newest to oldest to find how a change can reach the output.
    # Each requested node gets its own column, starting with a value of one
    # at that node. suffix stores the later-walk weights from each candidate's
    # endpoints to those requested nodes, for every supported remaining degree.
    state = np.zeros((depth, n, len(nodes)))
    state[0, nodes, np.arange(len(nodes))] = 1.0
    suffix = np.zeros((len(candidates), depth, len(nodes), 2))
    boundaries = group_boundaries(stamp)
    position = len(candidates) - 1
    for group in range(len(boundaries) - 2, -1, -1):
        start, stop = boundaries[group : group + 2]
        # Save the suffix before adding this timestamp to the backward state.
        # A candidate can use only strictly later events to reach the query.
        while position >= 0 and candidates[position] >= start:
            event = candidates[position]
            suffix[position, :, :, 0] = state[:, src[event]]
            suffix[position, :, :, 1] = state[:, dst[event]]
            position -= 1
        weight = np.exp(-rate * (query_time - stamp[start]))
        propagate(state, src, dst, start, stop, weight)
    return suffix

#This function starts with the original result. 
# Then, for every removed event, it calculates how that removal affects the final nodes 
# through later events. It adds those changes to the original result and returns the new corrected result.
def add_suffix_effects(original, suffix, selected, injections):
    """Add the effects of later walks to the original answer."""
    output = original.copy()
    depth = len(original) - 1
    for position, candidate in enumerate(selected):
        for degree in range(1, depth + 1):
            for level in range(1, degree + 1):
                output[degree] += (
                    suffix[candidate, degree - level] @ injections[position, level]
                )
    return output

#It's job is to calculate the new results after selected past event are removed. 

def evaluate_circuit(prefix, transfers, suffix, weights, selected, original):
    """Calculate the effect of removing one set of events."""
#prefix: information that existed before each removable event.
#transfers: shows how a change from an earlier removed event reaches a later removed event.
#suffix: shows how a change reaches the final requested nodes.
#weights: importance of each event based on its age.
#selected: positions of the events we want to remove.
#original: result when no events are removed.

#two sizes from the prefix NumPy array:
# prefix.shape[1]: maximum path length, saved as depth.
# prefix.shape[-1]: amount of information stored for each node, saved as dimension.
#-1 means the last part of the array’s shape.
    depth, dimension = prefix.shape[1], prefix.shape[-1]
    # Injection: NumPy table filled with zeros. It will store the change caused by every removed event.
    #len(selected): number of events being removed.
    injections = np.zeros((len(selected), depth + 1, 2, dimension))

    #his loop goes through every event that should be removed.
    for position, candidate in enumerate(selected):
        for degree in range(1, depth + 1):
            value = prefix[candidate, degree - 1].copy()
            # The original prefix still includes every event. Correct it using the
            # changes from removals already handled, carried through the original history.
            for earlier in range(position):
                source = selected[earlier]
                for level in range(1, degree):
                    transfer = transfers[
                        candidate, degree - 1 - level, :, 2 * source : 2 * source + 2
                    ]
                    value += transfer @ injections[earlier, level]
            # An edge sends each endpoint's value to the opposite endpoint.
            # Reverse the two endpoint rows, then negate the weighted update to undo it.
            injections[position, degree] = -weights[candidate] * value[::-1]
    return add_suffix_effects(original, suffix, selected, injections)
#It returns a NumPy array containing:
#- Results for each allowed path length.
#- Results for each requested node.
#- All information values for those nodes.



class BaseEvaluator:
    """Shared code for other evaluator classes to ueuse it. """

    exact = True #This evaluator provide exact results, not approximate results.

#src: start node, dst: ending node, stamp: time of the event, base: initial state or features of the nodes, 
# depth: Going back to k steps following time order. , query_time: time at which we want to evaluate the state of future node.,
# rate: Decides how quickly old events become less influential
#candidates:  Events that we select to remove
#  nodes: list of future nodes for which we want to evaluate the state.
    def __init__(
        self, src, dst, stamp, base, depth, query_time, rate, candidates, nodes
    ):
        # Checking that the input data is valid and converting it proper numpy formate.
        src, dst, stamp, base, candidates, nodes = validate(
            src, dst, stamp, base, depth, query_time, rate, candidates, nodes
        )
# n is number of nodes, d is dimension of features and depth is number of steps to go back in time.
        self.src, self.dst, self.stamp = src, dst, stamp
        self.n, self.d, self.depth = len(base), base.shape[1], depth

        self.query_time, self.rate = query_time, rate
        self.candidates, self.nodes = candidates, nodes
        # state is the original answer of the current nodes and prefix is the saved information upto removal event. 

        state, self.prefix = forward_index(
            src, dst, stamp, base, depth, query_time, rate, candidates
        )
        #original is the original answer of the current nodes 
        # Suffix read the history backwards to find how a deleted event can affect the quuery nodes.
        self.original = state[:, nodes].copy()
        self.suffix = backward_query_index(
            src, dst, stamp, self.n, depth, query_time, rate, candidates, nodes
        )
        # older events receive smaller weight while recent events receive larger weight.
        self.weights = np.exp(-rate * (query_time - stamp[candidates]))

# It create space and count answers.
    def out_buffer(self, count):
        return np.empty((count,) + self.original.shape)

#Test the effect of removing one group of nodes. 
    def evaluate(self, event_ids):
        """Evaluate the effect of removing one set of events."""
        hypotheses, lengths = encode(self.candidates, [tuple(event_ids)])
        output = self.out_buffer(1)
        self.batch(hypotheses, lengths, output)
        return output[0]

#The function for testing many groups of removed events.
# hypotheses: groups of events to remove.
# lengths: number of events in every group.
# output: where the answers will be saved.

    def batch(self, hypotheses, lengths, output):
        
        raise NotImplementedError("Implement batch in the method class")
#
#Calculates how much computer memory the shared NumPy arrays use.

    def shared_bytes(self):
        
        return sum(
            array.nbytes
            for array in (self.prefix, self.suffix, self.weights, self.original)
        )


class LazyEvaluator(BaseEvaluator):
    """Storage shared by the source-lazy and pair-lazy comparisons."""

    def __init__(self, *args):
        # Prepare space for methods that calculate transfers only when requested.
        # A transfer tells how a change at one candidate reaches another candidate.
        # The full table is reserved here, even though lazy methods fill only the
        # parts they need. A small two-column workspace follows one source at a time.
        super().__init__(*args)
        count = len(self.candidates)
        # Older events receive smaller weights when rate is positive.
        # At rate zero every event has weight one.
        self.event_weights = np.exp(-self.rate * (self.query_time - self.stamp))
        # The axes mean target candidate, middle-walk degree, target endpoint,
        # and source endpoint. Source j owns columns 2*j and 2*j+1.
        # Two removed events already use two steps, so middle walks need at most
        # depth - 2 steps; max keeps small-depth array sizes valid.
        self.transfers = np.zeros((count, max(self.depth - 1, 0), 2, 2 * count))
        self.workspace = np.zeros((max(self.depth - 1, 1), self.n, 2))
        # misses counts new cache calculations. scanned counts history events
        # visited while building those transfers; repeated requests can reuse them.
        self.misses = 0
        self.scanned = 0
