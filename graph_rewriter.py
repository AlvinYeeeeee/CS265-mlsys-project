from typing import Dict, List, Set

import torch.fx as fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs

from graph_prof import GraphProfiler
from ac_algorithm import get_recompute_info


# Helpers (mirrors activation_checkpoint.py helpers)

def _replace_subsequent_uses_of(
    graph: fx.Graph,
    old_node: fx.Node,
    new_node: fx.Node,
) -> None:
    """
    Replace uses of `old_node` with `new_node` for all nodes that appear
    AFTER `new_node` in the graph.  Nodes before `new_node` keep using the
    original (this preserves the forward-pass computation).
    """
    old_node_users = set(old_node.users.keys())
    for node in reversed(list(graph.nodes)):
        if node is new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def _build_name_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    return {n.name: n for n in gm.graph.nodes}


# Main rewrite function

def apply_activation_checkpointing(
    gm: fx.GraphModule,
    nodes_to_recompute: Set[fx.Node],
    nodes_to_retain: Set[fx.Node],
    profiler: GraphProfiler,
) -> fx.GraphModule:
    """
    Rewrite `gm` so that each activation in `nodes_to_recompute` is discarded
    after the forward pass and recomputed before its first backward use.

    Parameters
    ----------
    gm : fx.GraphModule
        The combined fwd+bwd+optimizer graph produced by graph_tracer.compile.
    nodes_to_recompute : Set[fx.Node]
        Activations selected by mu_two_selection to be checkpointed.
    nodes_to_retain : Set[fx.Node]
        Activations that stay in memory (used as subgraph boundary inputs).
    profiler : GraphProfiler
        The profiler whose static analysis produced the above sets.

    Returns
    -------
    gm : fx.GraphModule
        The modified graph module (modified in place, also returned).
    """
    if not nodes_to_recompute:
        return gm

    # Build name → node map; updated as we insert new nodes.
    name_to_node: Dict[str, fx.Node] = _build_name_map(gm)

    # Sort activations by first_bwd_use position so we insert them in graph
    # order (earliest-needed first).  This keeps the graph topologically valid
    # after each insertion.
    node_to_idx = {n: i for i, n in enumerate(gm.graph.nodes)}

    def _first_bwd_idx(act: fx.Node) -> int:
        fbwd = profiler.act_first_bwd_use.get(act)
        return node_to_idx.get(fbwd, int(1e9)) if fbwd is not None else int(1e9)

    ordered = sorted(nodes_to_recompute, key=_first_bwd_idx)

    for act in ordered:
        first_bwd_use = profiler.act_first_bwd_use.get(act)
        if first_bwd_use is None:
            # No backward user — nothing to recompute, skip.
            continue

        # Get the ordered subgraph node list and its boundary input nodes.
        _, boundary_inputs = get_recompute_info(act, profiler, nodes_to_retain)

        # Extract a fresh graph representing the recomputation.
        # inputs  = boundary nodes (placeholders + retained activations)
        # outputs = [act]  (the single activation we want to reproduce)
        recompute_graph = _extract_graph_with_inputs_outputs(
            joint_graph=gm.graph,
            inputs=boundary_inputs,
            outputs=[act],
        )
        # asd 

        # Insert the recomputation nodes just before first_bwd_use.
        with gm.graph.inserting_before(first_bwd_use):
            for n in recompute_graph.nodes:
                if n.op in ("placeholder", "output"):
                    # placeholder  → maps to an existing boundary node
                    # output       → nothing to insert
                    continue

                # Copy node into the main graph; resolve its arguments via
                # name_to_node so they point to real nodes in gm.graph.
                new_node = gm.graph.node_copy(
                    n,
                    arg_transform=lambda arg: name_to_node[arg.name],
                )

                # If this copied node is the activation we want to recompute,
                # redirect all subsequent (backward) uses to the new copy.
                if n.name == act.name:
                    _replace_subsequent_uses_of(gm.graph, act, new_node)

                # Update the name map so later nodes in this subgraph can
                # reference this newly inserted node by name.
                name_to_node[n.name] = new_node

        # Refresh the name map after insertion (names may have been de-duped
        # by fx when new_node was created).
        name_to_node = _build_name_map(gm)

    gm.graph.lint()
    gm.recompile()
    return gm
