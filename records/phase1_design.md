# Phase 1 Design: Graph Profiler

**CS265 MLSys Project — Spring 2025**
**Date:** 2026-03-22

---

## Overview

Phase 1 implements `GraphProfiler` in `graph_prof.py`. The profiler extends
`torch.fx.Interpreter` and operates on the combined forward + backward +
optimizer `fx.GraphModule` produced by `graph_tracer.compile`. It has two
responsibilities:

1. **Static analysis** (done once in `__init__`, no GPU execution needed):
   classify every node, identify activations, and record their lifetime
   boundaries in the graph.

2. **Runtime profiling** (done during `run_node` across multiple iterations):
   measure per-node execution time and GPU memory delta using CUDA Events.

The profiler's output feeds directly into the Phase 2 AC selection algorithm.

---

## Graph Structure

The combined graph produced by `graph_tracer.compile` has the following
section order (topologically):

```
[placeholder nodes]        <- model params, optimizer states, input batch
        |
[forward ops]              <- linear, relu, conv, etc.
        |
  sep                      <- torch.ops.separator.sep.default
        |
  loss
        |
  sep_backward             <- torch.ops.separator.sep_backward.default
        |
[backward ops]             <- gradient computations
        |
[optimizer ops]            <- _foreach_mul, _foreach_sqrt, _foreach_addcdiv, copy_...
        |
   output
```

The two sentinel nodes are the anchors for all static analysis. Everything
before `sep` is in the **forward region**. Everything after `sep_backward` is
in the **backward/optimizer region**.

---

## Node Classification

### Placeholder Node Types

All `placeholder` nodes appear at the top of the graph and are the "inputs"
to the compiled function (`flat_inps = flat_state + [args, kwargs]`).

The `flat_state` is constructed in `graph_tracer._compile` as:
```
flat_state = list(params_and_buffers.values()) + list(named_states.values())
```
followed by the call arguments (model object, optimizer object, batch tensor).

We classify placeholders as follows:

| Type | How to identify |
|------|----------------|
| **PARAM** | Appears as input to the final `_foreach_addcdiv` (or `_fused_adam`) node at arg positions 0..N-1 |
| **GRAD** | Appears as input to the final `_foreach_addcdiv` at the gradient argument positions (not needed separately — grads are intermediate nodes in the backward region) |
| **OPT_STATE** | Other placeholders that appear in optimizer ops (`_foreach_mul`, `_foreach_add`, etc.) but are not params |
| **INPUT** | Placeholders used in the forward region but not in optimizer ops (the input batch) |

In practice, the simplest rule: scan the `_foreach_addcdiv` node's args.
The first `N` tensor args (those that are `placeholder` op nodes) are params.
The remaining `placeholder` args that appear in optimizer ops are optimizer
states. The rest are inputs.

### Non-Placeholder Node Types

| Type | How to identify |
|------|----------------|
| **ACT (activation)** | Non-placeholder node created in the **forward region** (`node_idx < sep_idx`) that has at least one user in the **backward region** (`user_idx > sep_bwd_idx`) |
| **GRAD** | Non-placeholder node in the backward region |
| **OTHER** | Boundary nodes (sep, sep_backward, loss, output) and pure optimizer nodes |

---

## Static Analysis (`__init__`)

### Algorithm

```
Pass 1 — Find boundaries:
    For each node in graph.nodes:
        if node.target == torch.ops.separator.sep.default:
            sep_node = node; sep_idx = i
        if node.target == torch.ops.separator.sep_backward.default:
            sep_bwd_node = node; sep_bwd_idx = i

Pass 2 — Classify regions:
    forward_nodes  = {node | idx < sep_idx}
    backward_nodes = {node | idx > sep_bwd_idx}

Pass 3 — Identify activations:
    activation_nodes = set()
    for node in forward_nodes:
        if node.op == 'placeholder': continue
        if any(user in backward_nodes for user in node.users):
            activation_nodes.add(node)

Pass 4 — Record activation lifetimes:
    for act in activation_nodes:
        last_fwd_use  = max(user for user in act.users if user in forward_nodes)
                        [by graph order]
        first_bwd_use = min(user for user in act.users if user in backward_nodes)
                        [by graph order]
```

### Data Stored After `__init__`

```python
self.sep_node          : fx.Node
self.sep_bwd_node      : fx.Node
self.forward_nodes     : Set[fx.Node]
self.backward_nodes    : Set[fx.Node]
self.node_type         : Dict[fx.Node, NodeType]    # PARAM/ACT/GRAD/OTHER
self.activation_nodes  : Set[fx.Node]
self.act_last_fwd_use  : Dict[fx.Node, fx.Node]     # act -> last fwd user
self.act_first_bwd_use : Dict[fx.Node, fx.Node]     # act -> first bwd user

# Runtime stat accumulators (initialized empty, filled during run_node):
self.node_runtimes     : Dict[fx.Node, List[float]] # ms per run
self.node_mem_delta    : Dict[fx.Node, List[int]]   # bytes delta per run

# Aggregated (filled by aggregate_stats):
self.node_avg_runtime  : Dict[fx.Node, float]       # avg ms
self.node_avg_mem_delta: Dict[fx.Node, float]       # avg bytes delta
```

---

## Runtime Profiling (`run_node`)

### Timing

Use `torch.cuda.Event(enable_timing=True)` — **not** Python `time.time()`.
CUDA operations are asynchronous; only CUDA Events give accurate per-op timing.

```python
def run_node(self, n):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    mem_before = torch.cuda.memory_allocated()

    start.record()
    result = super().run_node(n)
    end.record()

    torch.cuda.synchronize()
    self.node_runtimes[n].append(start.elapsed_time(end))   # ms
    self.node_mem_delta[n].append(
        torch.cuda.memory_allocated() - mem_before           # bytes
    )
    return result
```

### Why synchronize after each node?

Without `synchronize()`, `elapsed_time` would block until the kernel
completes, but the memory delta read would be premature. Synchronizing ensures
both measurements are accurate. This adds overhead, but profiling runs are
separate from the warm-up and training runs so accuracy is preferred here.

---

## Stats Lifecycle

```
__init__             -> accumulators initialized (empty lists)
warm_up_iters × run  -> stats accumulate but will be discarded
reset_stats()        -> clears all lists
profile_iters × run  -> stats accumulate for real
aggregate_stats()    -> computes per-node averages
print_stats()        -> outputs table
```

### `reset_stats`
```python
def reset_stats(self):
    for node in self.module.graph.nodes:
        self.node_runtimes[node].clear()
        self.node_mem_delta[node].clear()
```

### `aggregate_stats`
```python
def aggregate_stats(self):
    for node in self.module.graph.nodes:
        rts = self.node_runtimes[node]
        mds = self.node_mem_delta[node]
        if rts:
            self.node_avg_runtime[node]   = sum(rts) / len(rts)
            self.node_avg_mem_delta[node] = sum(mds) / len(mds)
```

### `print_stats`

Print a table with columns:

```
Node Name | Op Type | Node Type | Avg Runtime (ms) | Avg Mem Delta (MB)
```

For activations, also print:
```
  [ACT] last_fwd_use: <node_name>  first_bwd_use: <node_name>
        idle window:  <N nodes sit between those two>
```

---

## Peak Memory Timeline (Deliverable 4a)

To produce the **peak memory breakdown bar graph** (deliverable 4a), we need a
memory-over-time simulation. This can be done as a post-processing step using
the static analysis data, **without additional GPU execution**:

1. Walk nodes in graph order.
2. When node `n` executes and produces a tensor result, add its size in bytes
   to `current_memory`. Record the tensor-to-node assignment.
3. When a node `n` is the **last user** of some tensor `t`, subtract `t`'s
   size from `current_memory` (because it is freed).
4. Track `peak_memory = max(current_memory)` throughout.
5. At the `sep_bwd_node`, record `fwd_peak_memory` (the peak at the fwd/bwd
   boundary — this is when all activations are alive simultaneously).

Tensor sizes are obtained at runtime: during `run_node`, if `result` is a
`torch.Tensor`, store `result.element_size() * result.nelement()` keyed by
node. For non-tensor results (scalars, tuples, None), record 0.

---

## Output for Phase 2 (AC Algorithm)

Phase 1 must produce the following per-activation data for Phase 2 to consume:

| Field | Description |
|-------|-------------|
| `node` | The `fx.Node` object of the activation |
| `size_bytes` | Memory consumed by this activation (from runtime) |
| `recompute_time_ms` | Avg runtime of the subgraph needed to recompute it |
| `last_fwd_use` | Last node in forward that uses this activation |
| `first_bwd_use` | First node in backward that uses this activation |

`recompute_time_ms` is the sum of runtimes of nodes in the recomputation
subgraph (extracted using `_extract_graph_with_inputs_outputs`).

---

## Files to Modify

| File | Changes |
|------|---------|
| `graph_prof.py` | Full implementation of `GraphProfiler` |
| `starter_code.py` | `graph_transformation` can remain as-is for now; print_stats output will be verified here |
| `benchmarks.py` | Used for running the profiler on real models (ResNet18/50, Transformer) — no changes needed until we have profiler working |

---

## Implementation Order

1. Static analysis (`__init__`) — write and verify by printing classification
   results on the DummyModel graph (no GPU needed).
2. `run_node` timing + memory — verify that runtimes are nonzero and memory
   deltas look sensible for the DummyModel.
3. `reset_stats` / `aggregate_stats` / `print_stats` — produce the stats table.
4. Memory timeline simulation — produce the peak memory breakdown.
5. Run on ResNet18/ResNet50/Transformer via `benchmarks.py`.

---

## Open Questions

- **Gradient nodes as activations?** Some tensors created in the backward pass
  are used by later backward ops (e.g., intermediate gradient accumulations).
  These are NOT activations by definition — activations are forward-created
  tensors used in backward. The filter `node in forward_nodes` handles this.

- **In-place ops (`copy_`)**: In-place operations modify a tensor in place. The
  `copy_` nodes at the end of the graph (optimizer) write the updated param
  values back in-place. These should be classified as OTHER/optimizer, not
  activations.

- **Multi-output nodes** (e.g., `_foreach_sqrt` returns a list): When a node
  returns a list/tuple, `getitem` nodes extract individual elements. The memory
  is actually allocated in the aggregate op, not in the `getitem` nodes. We
  should attribute memory to the source op and treat `getitem` nodes as
  zero-memory.
