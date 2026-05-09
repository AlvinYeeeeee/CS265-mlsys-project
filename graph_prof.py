from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.fx as fx


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType records the semantic role of each node in the combined
    forward + backward + optimizer graph.

    PARAM   : model parameter placeholder (updated by optimizer)
    ACT     : activation / intermediate tensor — created in forward,
              consumed in backward (the targets for AC)
    GRAD    : gradient node (backward region, non-placeholder)
    OPT     : optimizer-state placeholder (exp_avg, exp_avg_sq, step)
    INPUT   : input mini-batch placeholder
    OTHER   : everything else (loss, sentinel ops, getitem, copy_, output)
    """
    PARAM = 0
    ACT = 1
    GRAD = 2
    OPT = 3
    INPUT = 4
    OTHER = 5

class GraphProfiler(fx.Interpreter):
    """
    Profiles the combined fwd+bwd+optimizer fx.GraphModule produced by
    graph_tracer.compile.

    Responsibilities
    ----------------
    Static analysis (in __init__, no execution):
      • Locate the forward/backward sentinel nodes.
      • Classify every node by NodeType.
      • Identify activation nodes and record their lifetime boundaries
        (last_fwd_use, first_bwd_use).

    Runtime profiling (in run_node, repeated across iterations):
      • Time each node with torch.cuda.Event.
      • Record GPU memory delta per node.

    Usage
    -----
        profiler = GraphProfiler(gm)
        for _ in range(warmup): profiler.run(*args)
        profiler.reset_stats()
        for _ in range(iters):  profiler.run(*args)
        profiler.aggregate_stats()
        profiler.print_stats()
    """

    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # Static analysis

        nodes: List[fx.Node] = list(module.graph.nodes)
        node_to_idx: Dict[fx.Node, int] = {n: i for i, n in enumerate(nodes)}

        # 1. Find the fwd/bwd sentinel boundaries
        self.sep_node: Optional[fx.Node] = None
        self.sep_bwd_node: Optional[fx.Node] = None
        sep_idx = -1
        sep_bwd_idx = len(nodes)

        for i, node in enumerate(nodes):
            if node.target is torch.ops.separator.sep.default:
                self.sep_node = node
                sep_idx = i
            elif node.target is torch.ops.separator.sep_backward.default:
                self.sep_bwd_node = node
                sep_bwd_idx = i

        assert self.sep_node is not None, "sep sentinel node not found in graph"
        assert self.sep_bwd_node is not None, "sep_backward sentinel node not found in graph"

        # 2. Partition nodes into regions
        self.forward_nodes: Set[fx.Node] = {
            n for n in nodes if node_to_idx[n] < sep_idx
        }
        self.backward_nodes: Set[fx.Node] = {
            n for n in nodes if node_to_idx[n] > sep_bwd_idx
        }

        # 3. Identify model params and optimizer-state placeholders 
        param_nodes: Set[fx.Node] = set()
        opt_state_nodes: Set[fx.Node] = set()
        self._find_param_and_opt_nodes(nodes, param_nodes, opt_state_nodes)

        # 4. Classify every node
        self.node_type: Dict[fx.Node, NodeType] = {}
        for node in nodes:
            self.node_type[node] = self._classify_node(
                node, node_to_idx, sep_idx, sep_bwd_idx,
                param_nodes, opt_state_nodes,
            )

        # 5. Find activations (fwd-created, bwd-consumed)
        self.activation_nodes: Set[fx.Node] = {
            n for n in nodes
            if self.node_type[n] == NodeType.ACT
        }

        # 6. Record activation lifetime boundaries
        self.act_last_fwd_use:  Dict[fx.Node, Optional[fx.Node]] = {}
        self.act_first_bwd_use: Dict[fx.Node, Optional[fx.Node]] = {}

        for act in self.activation_nodes:
            fwd_users = [u for u in act.users if u in self.forward_nodes]
            bwd_users = [u for u in act.users if u in self.backward_nodes]

            self.act_last_fwd_use[act] = (
                max(fwd_users, key=lambda u: node_to_idx[u])
                if fwd_users else None
            )
            self.act_first_bwd_use[act] = (
                min(bwd_users, key=lambda u: node_to_idx[u])
                if bwd_users else None
            )

        # Runtime stat storage
        # Filled during run_node; reset_stats clears these.
        self._runtimes:  Dict[fx.Node, List[float]] = defaultdict(list)  # ms
        self._mem_delta: Dict[fx.Node, List[int]]   = defaultdict(list)  # bytes
        # Tensor sizes observed at runtime (node -> bytes); used for memory timeline.
        self._tensor_sizes: Dict[fx.Node, int] = {}

        # Aggregated after profile_iters (filled by aggregate_stats).
        self.node_avg_runtime:   Dict[fx.Node, float] = {}
        self.node_avg_mem_delta: Dict[fx.Node, float] = {}

    # Static analysis helpers                                                 #

    def _find_param_and_opt_nodes(
        self,
        nodes: List[fx.Node],
        param_nodes: Set[fx.Node],
        opt_state_nodes: Set[fx.Node],
    ) -> None:
        """
        Identify placeholder nodes that are model parameters vs optimizer
        states by inspecting the optimizer kernel node's argument list.

        Supports both:
          • fused=True  → single _fused_adam node
          • foreach=True → decomposed; final weight-write is _foreach_addcdiv
        """
        optimizer_node: Optional[fx.Node] = None

        for node in nodes:
            tgt = node.target
            if tgt is torch.ops.aten._fused_adam.default:
                # fused Adam: args[0] = params list, args[1] = grads list
                optimizer_node = node
                raw_params = node.args[0]  # list/tuple of nodes
                for n in raw_params:
                    if isinstance(n, fx.Node) and n.op == OP.PLACEHOLDER:
                        param_nodes.add(n)
                # All other placeholder args → optimizer states
                for arg in node.args[2:]:
                    if isinstance(arg, (list, tuple)):
                        for n in arg:
                            if isinstance(n, fx.Node) and n.op == OP.PLACEHOLDER:
                                opt_state_nodes.add(n)
                break

        if optimizer_node is None:
            # foreach=True path: look for _foreach_addcdiv (the param update)
            for node in reversed(nodes):
                if node.target is torch.ops.aten._foreach_addcdiv.Scalar:
                    optimizer_node = node
                    # First half of the flat arg list are the params (placeholders
                    # that appear in the output's input list)
                    flat_args = node.args[0]  # params
                    for n in flat_args:
                        if isinstance(n, fx.Node) and n.op == OP.PLACEHOLDER:
                            param_nodes.add(n)
                    # Second flat list are the bias-corrected 2nd moments (opt states)
                    for n in node.args[1]:
                        if isinstance(n, fx.Node) and n.op == OP.PLACEHOLDER:
                            opt_state_nodes.add(n)
                    # Third list: the sqrt(v_hat) — derived, not placeholders
                    break

        if optimizer_node is None:
            # Fallback: mark nothing — every placeholder stays INPUT/OTHER
            return

        # Any placeholder not already classified stays as a candidate for
        # opt_state. Walk all placeholder nodes and pick up the rest.
        output_node = next(n for n in nodes if n.op == OP.OUTPUT)
        output_inputs = set(output_node.all_input_nodes)
        for node in nodes:
            if node.op != OP.PLACEHOLDER:
                break  # placeholders are always at the top
            if node in param_nodes:
                continue
            # If the placeholder feeds into optimizer ops but not into the
            # forward region exclusively, treat it as an opt state.
            users_in_bwd = any(u in self.backward_nodes for u in node.users)
            if users_in_bwd and node not in param_nodes:
                opt_state_nodes.add(node)

    def _classify_node(
        self,
        node: fx.Node,
        node_to_idx: Dict[fx.Node, int],
        sep_idx: int,
        sep_bwd_idx: int,
        param_nodes: Set[fx.Node],
        opt_state_nodes: Set[fx.Node],
    ) -> NodeType:
        idx = node_to_idx[node]

        if node.op == OP.PLACEHOLDER:
            if node in param_nodes:
                return NodeType.PARAM
            if node in opt_state_nodes:
                return NodeType.OPT
            return NodeType.INPUT

        if node.op in (OP.OUTPUT,):
            return NodeType.OTHER

        # Sentinel nodes
        if node.target in (
            torch.ops.separator.sep.default,
            torch.ops.separator.sep_backward.default,
        ):
            return NodeType.OTHER

        if idx < sep_idx:
            # Forward region: non-placeholder nodes are activations if any
            # of their users are in the backward region.
            if any(node_to_idx.get(u, -1) > sep_bwd_idx for u in node.users):
                return NodeType.ACT
            return NodeType.OTHER

        if idx > sep_bwd_idx:
            return NodeType.GRAD

        # Between sep and sep_bwd (loss region)
        return NodeType.OTHER

    # Interpreter overrides

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    def run_node(self, n: fx.Node) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        mem_before = torch.cuda.memory_allocated()
        start.record()

        result = super().run_node(n)

        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        mem_after  = torch.cuda.memory_allocated()

        self._runtimes[n].append(elapsed_ms)
        self._mem_delta[n].append(mem_after - mem_before)

        # Record tensor size for memory timeline (first time only is fine;
        # sizes are static across iterations).
        if n not in self._tensor_sizes:
            self._tensor_sizes[n] = self._measure_output_bytes(result)

        return result

    @staticmethod
    def _measure_output_bytes(result: Any) -> int:
        """Return the number of bytes consumed by a node's output."""
        if isinstance(result, torch.Tensor):
            return result.element_size() * result.nelement()
        if isinstance(result, (list, tuple)):
            return sum(
                t.element_size() * t.nelement()
                for t in result
                if isinstance(t, torch.Tensor)
            )
        return 0

    # Stats lifecycle

    def reset_stats(self) -> None:
        """Clear accumulated runtime/memory lists (call after warm-up)."""
        self._runtimes.clear()
        self._mem_delta.clear()

    def aggregate_stats(self) -> None:
        """Average runtime and memory delta over profiling iterations."""
        for node in self.module.graph.nodes:
            rts = self._runtimes.get(node, [])
            mds = self._mem_delta.get(node, [])
            if rts:
                self.node_avg_runtime[node]   = sum(rts) / len(rts)
                self.node_avg_mem_delta[node] = sum(mds) / len(mds)

    def print_stats(self) -> None:
        """
        Print a per-node stats table and a summary of activation lifetimes.
        """
        col = "{:<35} {:<8} {:<10} {:>12} {:>14}"
        header = col.format("Node", "Op", "NodeType", "Runtime(ms)", "MemDelta(MB)")
        print("\n" + "=" * len(header))
        print("Per-node profiling stats")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        total_fwd_ms  = 0.0
        total_bwd_ms  = 0.0
        total_act_mem = 0

        for node in self.module.graph.nodes:
            rt  = self.node_avg_runtime.get(node, 0.0)
            md  = self.node_avg_mem_delta.get(node, 0.0)
            ntype = self.node_type.get(node, NodeType.OTHER)

            print(col.format(
                node.name[:35],
                node.op[:8],
                ntype.name,
                f"{rt:.3f}",
                f"{md / 1e6:.3f}",
            ))

            if node in self.forward_nodes:
                total_fwd_ms += rt
            elif node in self.backward_nodes:
                total_bwd_ms += rt
            if ntype == NodeType.ACT:
                total_act_mem += self._tensor_sizes.get(node, 0)

        print("=" * len(header))
        print(f"  Total forward  runtime : {total_fwd_ms:.3f} ms")
        print(f"  Total backward runtime : {total_bwd_ms:.3f} ms")
        print(f"  Total activation memory: {total_act_mem / 1e6:.2f} MB\n")

        # Activation lifetime summary
        if not self.activation_nodes:
            print("No activation nodes found.")
            return

        act_col = "{:<35} {:>10} {:>30} {:>30}"
        print(act_col.format("Activation", "Size(MB)", "last_fwd_use", "first_bwd_use"))
        print("-" * 110)
        for act in sorted(self.activation_nodes, key=lambda n: n.name):
            sz = self._tensor_sizes.get(act, 0)
            lf = self.act_last_fwd_use.get(act)
            fb = self.act_first_bwd_use.get(act)
            print(act_col.format(
                act.name[:35],
                f"{sz / 1e6:.3f}",
                lf.name[:30] if lf else "—",
                fb.name[:30] if fb else "—",
            ))

    def peak_memory_by_category(self) -> Dict[str, int]:
        """
        Simulate the memory timeline and return peak bytes broken down by
        category: params, activations, gradients, optimizer states.

        This is a post-processing utility for the deliverable 4(b) bar graph.
        """
        nodes = list(self.module.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # last_use[node] = index of the last node that uses `node` as input
        last_use: Dict[fx.Node, int] = {}
        for node in nodes:
            for user in node.users:
                idx = node_to_idx.get(user, -1)
                if idx > last_use.get(node, -1):
                    last_use[node] = idx

        live: Dict[fx.Node, int] = {}      # node -> bytes currently in memory
        peak_by_cat: Dict[str, int] = {
            "params": 0, "activations": 0,
            "gradients": 0, "opt_states": 0, "other": 0,
        }

        def _cat(node: fx.Node) -> str:
            t = self.node_type.get(node, NodeType.OTHER)
            return {
                NodeType.PARAM: "params",
                NodeType.ACT:   "activations",
                NodeType.GRAD:  "gradients",
                NodeType.OPT:   "opt_states",
            }.get(t, "other")

        for i, node in enumerate(nodes):
            sz = self._tensor_sizes.get(node, 0)
            if sz:
                live[node] = sz

            # Free tensors whose last user has just executed
            to_free = [n for n, lu in last_use.items() if lu == i and n in live]
            for n in to_free:
                del live[n]

            # Snapshot peak by category at this point in execution
            by_cat: Dict[str, int] = defaultdict(int)
            for n, b in live.items():
                by_cat[_cat(n)] += b

            for cat, total in by_cat.items():
                if total > peak_by_cat[cat]:
                    peak_by_cat[cat] = total

        return peak_by_cat
