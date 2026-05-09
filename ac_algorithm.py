"""
Phase 2: Activation Checkpointing Selection Algorithm (μ-TWO)

Given the profiler output from Phase 1, this module decides which activations
to discard during the forward pass (to be recomputed during backward) and
which to retain in memory.

Algorithm (from μ-TWO, MLSys 2023)
------------------------------------
For each candidate activation `act`:
  memory_saved(act)  = byte size of act's output tensor
  recompute_cost(act) = sum of runtimes of all forward nodes needed to
                        produce act, starting from placeholder / retained
                        activation boundaries
  efficiency(act)    = memory_saved / recompute_cost

Sort candidates by efficiency descending.  Greedily select activations to
checkpoint until estimated peak activation memory <= memory_budget_bytes.

Dependency constraint
---------------------
Activation A can only be checkpointed if every activation node that is a
direct boundary input to A's recomputation subgraph is still retained (not
also checkpointed).  This guarantees that at recompute-time all inputs to
A's subgraph are present in GPU memory.
"""

from collections import deque
from typing import Dict, List, Set, Tuple

import torch.fx as fx

from graph_prof import GraphProfiler, NodeType


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _recompute_subgraph_nodes(
    act: fx.Node,
    profiler: GraphProfiler,
    retained: Set[fx.Node],
) -> List[fx.Node]:
    """
    Return the ordered list of forward nodes (including `act` itself) that
    must execute to recompute `act`, given that `retained` activations and
    all placeholder nodes are available as inputs.

    BFS backwards through forward_nodes; stop (treat as boundary) at:
      - any placeholder node  (params, inputs, opt-states — always present)
      - any node in `retained`  (kept in memory, available at recompute time)
    """
    needed: List[fx.Node] = []
    visited: Set[fx.Node] = set()
    queue: deque = deque([act])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        # Boundary check: stop, don't include this node in the subgraph
        if node != act:
            if node.op == "placeholder":
                continue
            if node in retained:
                continue
            if node not in profiler.forward_nodes:
                # crossed into backward / optimizer region — skip
                continue

        needed.append(node)
        for inp in node.all_input_nodes:
            if inp not in visited:
                queue.append(inp)

    # Return in topological (graph) order
    node_to_idx = {n: i for i, n in enumerate(profiler.module.graph.nodes)}
    needed.sort(key=lambda n: node_to_idx.get(n, 0))
    return needed


def _recompute_cost(
    act: fx.Node,
    profiler: GraphProfiler,
    retained: Set[fx.Node],
) -> float:
    """
    Sum of average runtimes (ms) for all nodes in act's recomputation
    subgraph, given that `retained` activations are available as inputs.
    """
    nodes = _recompute_subgraph_nodes(act, profiler, retained)
    return sum(profiler.node_avg_runtime.get(n, 0.0) for n in nodes)


def _boundary_activation_inputs(
    act: fx.Node,
    profiler: GraphProfiler,
    retained: Set[fx.Node],
) -> Set[fx.Node]:
    """
    Return the activation nodes from `retained` that sit just outside the
    border of act's recomputation subgraph.  These are the nodes that
    _extract_graph_with_inputs_outputs needs as its `inputs` argument, and
    the nodes that must remain retained for the checkpoint to be valid.
    """
    subgraph_set = set(_recompute_subgraph_nodes(act, profiler, retained))
    boundary: Set[fx.Node] = set()
    for n in subgraph_set:
        for inp in n.all_input_nodes:
            if inp not in subgraph_set and inp in profiler.activation_nodes:
                boundary.add(inp)
    return boundary


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def mu_two_selection(
    profiler: GraphProfiler,
    memory_budget_bytes: int,
) -> Tuple[Set[fx.Node], Set[fx.Node]]:
    """
    μ-TWO greedy activation checkpointing selection.

    Parameters
    ----------
    profiler : GraphProfiler
        Fully profiled graph profiler (aggregate_stats already called).
    memory_budget_bytes : int
        Target peak activation memory in bytes.  The algorithm checkpoints
        activations (highest efficiency first) until the estimated peak
        activation memory falls at or below this budget.

    Returns
    -------
    nodes_to_recompute : Set[fx.Node]
        Activations to DISCARD after forward and recompute before backward.
    nodes_to_retain : Set[fx.Node]
        Activations to KEEP in memory throughout the iteration.
    """
    # Only consider activations that have a measured size and a genuine
    # forward-to-backward lifetime window.
    candidates: List[fx.Node] = [
        act for act in profiler.activation_nodes
        if profiler._tensor_sizes.get(act, 0) > 0
        and profiler.act_last_fwd_use.get(act) is not None
        and profiler.act_first_bwd_use.get(act) is not None
    ]

    # Start with all candidates retained.
    retained: Set[fx.Node] = set(candidates)

    # Estimated peak activation memory: conservative upper bound = sum of all
    # activation sizes (they are all simultaneously alive at peak in the worst
    # case, which is true for typical sequential networks).
    current_peak = sum(profiler._tensor_sizes.get(a, 0) for a in retained)

    if current_peak <= memory_budget_bytes:
        return set(), set(retained)

    def efficiency(act: fx.Node, retained_set: Set[fx.Node]) -> float:
        mem = profiler._tensor_sizes.get(act, 0)
        # Compute cost as if `act` is NOT in the retained set
        cost = _recompute_cost(act, profiler, retained_set - {act})
        return (mem / cost) if cost > 1e-9 else float("inf")

    # Sort once by efficiency (all-retained baseline).  This order is a good
    # approximation even as the retained set shrinks during selection.
    sorted_candidates = sorted(
        candidates,
        key=lambda a: efficiency(a, retained),
        reverse=True,
    )

    nodes_to_recompute: Set[fx.Node] = set()

    for act in sorted_candidates:
        if current_peak <= memory_budget_bytes:
            break

        # Dependency constraint: every activation that is a direct boundary
        # input to act's recomputation subgraph must still be retained.
        required_inputs = _boundary_activation_inputs(act, profiler, retained)
        if not required_inputs.issubset(retained):
            # A required input has already been checkpointed — skip act.
            continue

        # Checkpoint this activation.
        nodes_to_recompute.add(act)
        retained.discard(act)
        current_peak -= profiler._tensor_sizes.get(act, 0)

    return nodes_to_recompute, retained


def get_recompute_info(
    act: fx.Node,
    profiler: GraphProfiler,
    nodes_to_retain: Set[fx.Node],
) -> Tuple[List[fx.Node], List[fx.Node]]:
    """
    For a single activation to be recomputed, return the information needed
    by the graph rewriter.

    Returns
    -------
    subgraph_nodes : List[fx.Node]
        Ordered list of forward nodes (including `act`) to re-execute.
    boundary_inputs : List[fx.Node]
        Placeholder nodes + retained activations that are the inputs to the
        subgraph (pass these as `inputs` to _extract_graph_with_inputs_outputs).
    """
    subgraph_nodes = _recompute_subgraph_nodes(act, profiler, nodes_to_retain)
    subgraph_set = set(subgraph_nodes)

    boundary_inputs: List[fx.Node] = []
    for n in subgraph_nodes:
        for inp in n.all_input_nodes:
            if inp not in subgraph_set and inp not in boundary_inputs:
                boundary_inputs.append(inp)

    return subgraph_nodes, boundary_inputs


def print_ac_plan(
    nodes_to_recompute: Set[fx.Node],
    nodes_to_retain: Set[fx.Node],
    profiler: GraphProfiler,
) -> None:
    """Pretty-print the checkpointing plan produced by mu_two_selection."""
    total = len(nodes_to_recompute) + len(nodes_to_retain)
    mem_saved = sum(profiler._tensor_sizes.get(n, 0) for n in nodes_to_recompute)
    mem_kept  = sum(profiler._tensor_sizes.get(n, 0) for n in nodes_to_retain)

    print(f"\n{'='*65}")
    print(f"  \u03bc-TWO Activation Checkpointing Plan")
    print(f"{'='*65}")
    print(f"  Total activations : {total}")
    print(f"  To recompute      : {len(nodes_to_recompute)}"
          f"  ({mem_saved / 1e6:.1f} MB freed)")
    print(f"  To retain         : {len(nodes_to_retain)}"
          f"  ({mem_kept / 1e6:.1f} MB kept in memory)")
    print(f"{'='*65}")

    if not nodes_to_recompute:
        print("  (no activations checkpointed — already within budget)")
        return

    col = "{:<35} {:>10} {:>14} {:>30}"
    print(col.format("Activation (recompute)", "Size(MB)", "RecmpCost(ms)", "first_bwd_use"))
    print("-" * 92)

    for act in sorted(
        nodes_to_recompute,
        key=lambda a: profiler._tensor_sizes.get(a, 0),
        reverse=True,
    ):
        sz   = profiler._tensor_sizes.get(act, 0)
        cost = _recompute_cost(act, profiler, nodes_to_retain)
        fbwd = profiler.act_first_bwd_use.get(act)
        eff  = (sz / cost) if cost > 1e-9 else float("inf")
        print(col.format(
            act.name[:35],
            f"{sz / 1e6:.3f}",
            f"{cost:.3f}",
            fbwd.name[:30] if fbwd else "—",
        ))

    print(f"{'='*65}\n")
