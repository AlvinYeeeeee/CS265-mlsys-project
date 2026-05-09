# Phase 1 Presentation: Computational Graph Profiler
**CS265 MLSys Project — Spring 2025**

---

## Part 1: Background and Starter Code

### The Big Picture

The project goal is to implement **Activation Checkpointing (AC)** — a
technique that reduces peak GPU memory during neural network training by
discarding a subset of intermediate tensors after the forward pass and
recomputing them on demand during the backward pass. Phase 1 builds the
profiler that measures what needs to be measured before any checkpointing
decisions can be made.

---

### How Neural Network Training Uses Memory

A single training iteration has three phases:

```
Input
  │
  ▼
[Forward Pass]  → produces activations Z1, Z2, Z3, ... Zn
                  all kept in GPU memory
  │
  ▼
[Loss]
  │
  ▼
[Backward Pass] → consumes Z1...Zn one by one (in reverse order)
                  to compute weight gradients ∇W
  │
  ▼
[Optimizer Step] → updates W using ∇W and optimizer states (Adam moments)
```

The activations are the problem. They are created at the start of the forward
pass but not consumed until the backward pass — in reverse order. So the very
first activation (e.g. from layer 1) sits in memory doing nothing until the
very last backward op. This idle window is where memory is wasted.
Activations account for roughly **70–85% of peak GPU memory**.

---

### The Starter Code: What Was Given

Five files were provided. Here is what each one does.

#### `graph_tracer.py` — The Compiler (do not modify)

This is the most complex file and was provided complete. It does one job:
take a normal Python training function and convert it into a single static
`fx.GraphModule` that contains all three phases (forward, backward, optimizer)
as a flat sequence of nodes.

The key pieces inside it:

**`SEPFunction`** — A custom PyTorch autograd function that acts as a no-op
(it just returns its input unchanged). Its sole purpose is to inject two
special marker nodes into the traced graph:
- `separator.sep` — marks the **end of the forward pass**
- `separator.sep_backward` — marks the **start of the backward pass**

Without these markers, there is no way to tell where the forward pass ends
and the backward pass begins in the combined graph.

**`_compile(func, *args)`** — The core tracing function. It:
1. Extracts the model's parameters and optimizer states and lifts them as
   explicit function arguments (so the optimizer update appears as graph nodes)
2. Calls `make_fx` with `FakeTensorMode` — this runs the function with fake
   symbolic tensors and records every ATen-level operation into a graph
3. Cleans up `detach` and `tag_grad` helper nodes

**`compile(func, gm_transformation)`** — The user-facing wrapper. It calls
`_compile` on the first invocation, then calls your `gm_transformation`
function with the captured graph, then runs the (possibly modified) graph on
every subsequent call.

#### `graph_prof.py` — The Profiler Skeleton (our main work)

The provided skeleton had:
- `OP` enum (node operation types: call_function, placeholder, output, etc.)
- `NodeType` enum (PARAM, ACT, GRAD, OTHER — but only these 4)
- `GraphProfiler(fx.Interpreter)` class with empty method bodies and a
  debug print loop in `__init__` that printed every node's name, type,
  target, inputs, and users

The `fx.Interpreter` base class is PyTorch's built-in graph runner that
executes a `GraphModule` node by node, calling `run_node` for each one. By
subclassing it we can intercept each node execution.

#### `starter_code.py` — The Entry Point (minor modification)

Provided complete. Defines:
- `DummyModel`: 10 Linear(100→100) + ReLU layers stacked sequentially
- `train_step`: forward pass → SEPFunction → backward → optimizer step
- `graph_transformation`: creates a `GraphProfiler`, runs warm-up and
  profile iterations, calls `aggregate_stats()` and `print_stats()`
- `experiment()`: wires everything together and calls `compile`

#### `benchmarks.py` — Real Model Experiments (extended by us)

The starter version had the `Experiment` class for Transformer, ResNet18,
and ResNet50. The optimizer was set to `fused=True` for both models.

#### `utils.py` — Decomposition Table (do not modify)

Provides `SPMD_DECOMP_TABLE` which tells `make_fx` how to decompose
in-place `_foreach_*` optimizer operations into functional equivalents. For
example, `_foreach_add_` (in-place) becomes `_foreach_add` (out-of-place)
followed by `copy_` nodes. This makes the graph fully functional and
inspectable.

---

### The Graph Structure After Tracing

After `compile` runs, the `fx.GraphModule` has this structure (in node order):

```
[placeholder nodes]     ← arg0_1...arg0_N  = model parameters
                          arg2_*, arg3_*   = Adam optimizer states (exp_avg, exp_avg_sq, step)
                          arg_batch        = input mini-batch
        ↓
[forward ops]           ← addmm, relu, conv2d, etc.
        ↓
   separator.sep         ← SEPFunction forward marker
        ↓
   loss computation
        ↓
   separator.sep_backward ← SEPFunction backward marker
        ↓
[backward ops]          ← mm, threshold_backward, etc.
        ↓
[optimizer ops]         ← _foreach_mul, _foreach_sqrt, _foreach_addcdiv, copy_, ...
        ↓
   output
```

For the DummyModel (10 layers), this graph has roughly 320 nodes.
For ResNet50, it exceeds 4,000 nodes.

---

## Part 2: What We Built

### Files Modified

| File | What changed |
|------|-------------|
| `graph_prof.py` | Complete rewrite — the entire implementation |
| `benchmarks.py` | Fixed optimizer bug; added sweep infrastructure and bar chart generation |
| `graph_tracer.py` | One small fix: suppress the `detect_anomaly` warning |

### Files Added

| File | Purpose |
|------|---------|
| `phase1_run.py` | Single entry point to run all Phase 1 experiments |
| `records/phase1_design.md` | Design document written before implementation |
| `records/midway_report.tex` | LaTeX source for the midway check-in report |
| `records/stats_resnet18.txt` | Full per-node profiling output for ResNet18 |
| `records/stats_resnet50.txt` | Full per-node profiling output for ResNet50 |
| `records/stats_transformer.txt` | Full per-node profiling output for Transformer |
| `records/peak_memory_resnet18.png` | Peak memory vs batch size bar chart |
| `records/peak_memory_resnet50.png` | Peak memory vs batch size bar chart |
| `records/peak_memory_transformer.png` | Peak memory vs batch size bar chart |

---

### Change 1: Complete Rewrite of `graph_prof.py`

This is the core deliverable. The original file had empty method bodies.
We implemented four separate capabilities.

#### (a) Static Analysis in `__init__`

Static analysis runs once when the `GraphProfiler` is constructed, before
any graph execution. It answers questions about graph structure purely by
inspecting node connections.

**Step 1 — Find the forward/backward boundary**

We scan all nodes for the two sentinel operators. Every node with
index `< sep_idx` is a forward node; every node with index `> sep_bwd_idx`
is a backward or optimizer node.

```python
# graph_prof.py  __init__, lines 81-98
for i, node in enumerate(nodes):
    if node.target is torch.ops.separator.sep.default:
        self.sep_node = node
        sep_idx = i                        # end of forward
    elif node.target is torch.ops.separator.sep_backward.default:
        self.sep_bwd_node = node
        sep_bwd_idx = i                    # start of backward

self.forward_nodes  = {n for n in nodes if node_to_idx[n] < sep_idx}
self.backward_nodes = {n for n in nodes if node_to_idx[n] > sep_bwd_idx}
```

**Step 2 — Identify model parameters and optimizer states**

We locate the `_foreach_addcdiv` node (the final weight update operation:
`W = W - lr * m_hat / (v_hat + ε)`). Its first argument list contains
the model parameters; other placeholder nodes that have users in the
backward region are optimizer states (Adam's `exp_avg`, `exp_avg_sq`,
and `step` tensors).

```python
# graph_prof.py  _find_param_and_opt_nodes, lines 192-225
for node in reversed(nodes):
    if node.target is torch.ops.aten._foreach_addcdiv.Scalar:
        flat_args = node.args[0]           # first list = model params
        for n in flat_args:
            if isinstance(n, fx.Node) and n.op == OP.PLACEHOLDER:
                param_nodes.add(n)
        for n in node.args[1]:             # second list = bias-corrected moments
            if isinstance(n, fx.Node) and n.op == OP.PLACEHOLDER:
                opt_state_nodes.add(n)
        break

# Any remaining placeholder that has users in backward = optimizer state
for node in nodes:
    if node.op != OP.PLACEHOLDER:
        break
    users_in_bwd = any(u in self.backward_nodes for u in node.users)
    if users_in_bwd and node not in param_nodes:
        opt_state_nodes.add(node)
```

**Step 3 — Classify every node**

Each node gets one of six types:

| NodeType | Meaning |
|----------|---------|
| `PARAM`  | Model weight placeholder |
| `ACT`    | Activation: created in forward, consumed in backward |
| `GRAD`   | Gradient computation node (in backward region) |
| `OPT`    | Adam optimizer state placeholder |
| `INPUT`  | Input mini-batch placeholder |
| `OTHER`  | Sentinel ops, loss, getitem, output |

The critical rule for `ACT`: a non-placeholder forward node is an
activation if and only if **at least one of its users is in the backward region**.

```python
# graph_prof.py  _classify_node, lines 227-266
def _classify_node(self, node, node_to_idx, sep_idx, sep_bwd_idx,
                   param_nodes, opt_state_nodes):
    idx = node_to_idx[node]

    if node.op == OP.PLACEHOLDER:
        if node in param_nodes:    return NodeType.PARAM
        if node in opt_state_nodes: return NodeType.OPT
        return NodeType.INPUT

    if node.target in (torch.ops.separator.sep.default,
                       torch.ops.separator.sep_backward.default):
        return NodeType.OTHER

    if idx < sep_idx:
        # Forward node is an activation if any user is in backward
        if any(node_to_idx.get(u, -1) > sep_bwd_idx for u in node.users):
            return NodeType.ACT
        return NodeType.OTHER

    if idx > sep_bwd_idx:
        return NodeType.GRAD

    return NodeType.OTHER   # between sep and sep_bwd (loss region)
```

**Step 4 — Record activation lifetimes**

For each activation node `a`, we compute:
- `last_fwd_use(a)` = the forward-region user of `a` with the highest
  graph index (the last time `a` is used before the backward pass starts)
- `first_bwd_use(a)` = the backward-region user of `a` with the lowest
  graph index (the first time `a` is needed in the backward pass)

The gap between these two is the **idle window** — how long the tensor
sits in GPU memory without being used. This is the key input for the
Phase 2 selection algorithm.

```python
# graph_prof.py  __init__, lines 127-138
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
```

#### (b) Runtime Profiling in `run_node`

We override `fx.Interpreter.run_node` to measure every node:

```python
def run_node(self, n):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    mem_before = torch.cuda.memory_allocated()

    start.record()
    result = super().run_node(n)    # actually execute the node
    end.record()

    torch.cuda.synchronize()        # wait for GPU to finish

    self._runtimes[n].append(start.elapsed_time(end))              # ms
    self._mem_delta[n].append(torch.cuda.memory_allocated() - mem_before)  # bytes
    return result
```

Two important design decisions here:

**Why `torch.cuda.Event` and not Python's `time.time()`?**
CUDA kernels execute asynchronously. When `super().run_node(n)` returns,
the GPU kernel may not have finished yet. `time.time()` would measure
near-zero for every node. CUDA Events are timestamps that live on the GPU
timeline, so `elapsed_time` gives the true kernel duration.

**Why `synchronize()` after each node?**
Without it, `elapsed_time` would block eventually anyway, but
`torch.cuda.memory_allocated()` would be read before the kernel has
actually allocated its output tensor, giving a wrong memory delta.

#### (c) Stats Lifecycle: `reset_stats`, `aggregate_stats`, `print_stats`

The profiler runs in three phases controlled by `starter_code.py` and
`benchmarks.py`:

```
GraphProfiler(gm)      → static analysis only
  ↓
run() × 2              → warm-up iterations (stats accumulate but are discarded)
  ↓
reset_stats()          → clears all accumulated lists
  ↓
run() × 3              → real measurement iterations
  ↓
aggregate_stats()      → computes per-node average runtime and memory delta
  ↓
print_stats()          → outputs two tables (see below)
```

The calling code in `benchmarks.py` that drives this lifecycle:

```python
# benchmarks.py  Experiment.graph_transformation, lines 109-124
def graph_transformation(self, gm, args):
    warm_up_iters, profile_iters = 2, 3
    self.profiler = GraphProfiler(gm)

    with torch.no_grad():
        for _ in range(warm_up_iters):
            self.profiler.run(*args)        # warm-up: throw away results
        self.profiler.reset_stats()         # ← clear before real measurement

        for _ in range(profile_iters):
            self.profiler.run(*args)        # real profiling iterations
        self.profiler.aggregate_stats()     # ← compute per-node averages
        self.profiler.print_stats()         # ← print both tables

    self.peak_mem_by_cat = self.profiler.peak_memory_by_category()
    return gm
```

`reset_stats` and `aggregate_stats` themselves:

```python
# graph_prof.py  lines 326-338
def reset_stats(self):
    self._runtimes.clear()
    self._mem_delta.clear()

def aggregate_stats(self):
    for node in self.module.graph.nodes:
        rts = self._runtimes.get(node, [])
        mds = self._mem_delta.get(node, [])
        if rts:
            self.node_avg_runtime[node]   = sum(rts) / len(rts)
            self.node_avg_mem_delta[node] = sum(mds) / len(mds)
```

`print_stats()` produces two outputs:

**Table 1 — Per-node stats** (one row per node):
```
Node                    Op       NodeType   Runtime(ms)  MemDelta(MB)
relu                    call_fun ACT              0.031         0.400
addmm_1                 call_fun OTHER            0.041         0.000
...
Total forward  runtime : 3.551 ms
Total backward runtime : 23.752 ms
Total activation memory: 4.36 MB
```

**Table 2 — Activation lifetime summary** (one row per activation):
```
Activation              Size(MB)    last_fwd_use         first_bwd_use
relu                       0.400         addmm_1               mm_17
relu_1                     0.400         addmm_2               mm_15
...
```

#### (d) Peak Memory Simulation: `peak_memory_by_category`

This method simulates the memory timeline without running the graph again.
It walks nodes in graph order, adds a tensor's byte size to a running total
when its producer executes, and removes it when its last user executes.
At each step it records the per-category running total and tracks the peak
for each category. The result feeds directly into the bar chart generator.

```python
# graph_prof.py  peak_memory_by_category, lines 400-452
for i, node in enumerate(nodes):
    sz = self._tensor_sizes.get(node, 0)
    if sz:
        live[node] = sz                      # tensor comes alive

    # Free tensors whose last user just executed
    to_free = [n for n, lu in last_use.items() if lu == i and n in live]
    for n in to_free:
        del live[n]

    # Snapshot per-category totals and record peak
    by_cat = defaultdict(int)
    for n, b in live.items():
        by_cat[_cat(n)] += b                 # _cat maps NodeType → string key

    for cat, total in by_cat.items():
        if total > peak_by_cat[cat]:
            peak_by_cat[cat] = total         # update per-category peak

return peak_by_cat   # {"params": N, "activations": N, "gradients": N, ...}
```

---

### Change 2: Fixes and Extensions to `benchmarks.py`

**Bug fix — optimizer incompatibility with tracing**

The original code used `fused=True` for both the Transformer and ResNet
optimizers. The `fused=True` Adam inspects `device.type` on its parameter
tensors during `make_fx` tracing — but `make_fx` uses fake tensors whose
device is `None`, causing `AttributeError: 'NoneType' object has no
attribute 'type'`.

Fix: change both to `foreach=True`, which decomposes into standard ATen
element-wise operations that are fully compatible with fake-tensor tracing.

```python
# Before (broken):
self.optimizer = optim.Adam(..., fused=True, capturable=True)

# After (fixed):
self.optimizer = optim.Adam(..., foreach=True, capturable=True)
```

**Added `sweep_batch_sizes` dictionary**

Specifies the batch sizes to sweep for the memory chart for each model:
```python
sweep_batch_sizes = {
    "Transformer": [1, 2, 4, 8],
    "Resnet18":    [4, 8, 16, 32, 64],
    "Resnet50":    [1, 2, 4, 8, 16],
}
```

**Updated `graph_transformation`** to save the `GraphProfiler` instance and
call `peak_memory_by_category()` after profiling, storing the result in
`self.peak_mem_by_cat` for the sweep to read. (Code shown above in section c.)

**Added `run_memory_sweep` function** which iterates over batch sizes,
instantiates an `Experiment` for each, runs the profiler, collects the peak
memory breakdown, and generates a stacked bar chart with matplotlib.

```python
# benchmarks.py  run_memory_sweep, lines 142-194
def run_memory_sweep(model_name, out_dir="records"):
    for bs in sweep_batch_sizes[model_name]:
        try:
            exp = Experiment(model_name, bs)
            exp.init_opt_states()
            compiled_fn = compile(exp.train_step, exp.graph_transformation)
            compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
            mem = exp.peak_mem_by_cat          # filled by graph_transformation
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={bs}, stopping sweep.")
            break
        for cat in categories:
            results[cat].append(mem.get(cat, 0) / 1e9)   # bytes → GB

    # stacked bar chart
    for cat, color in zip(categories, colors):
        ax.bar(x, results[cat], width, bottom=bottoms, label=cat, color=color)
        bottoms = [b + v for b, v in zip(bottoms, results[cat])]

    plt.savefig(f"{out_dir}/peak_memory_{model_name.lower()}.png", dpi=150)
```

---

### Change 3: Minor Fix to `graph_tracer.py`

Added `import warnings` at the top and wrapped the `detect_anomaly` context
with `warnings.catch_warnings()` to suppress the noisy "Anomaly Detection
has been enabled" message that PyTorch prints to stderr on every run.

```python
# graph_tracer.py  (inside _compile)
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="Anomaly Detection has been enabled"
    )
    with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        gm = make_fx(
            functional_call,
            tracing_mode="fake",
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(*args)
```

---

### New File: `phase1_run.py`

A single command-line entry point for all Phase 1 experiments:

```
python phase1_run.py                        # all models, profile + sweep
python phase1_run.py --model Resnet18       # one model only
python phase1_run.py --no-sweep             # profiling stats only
python phase1_run.py --no-profile           # bar charts only
```

It uses a `Tee` pattern to simultaneously print stats to the terminal and
capture them to `records/stats_<model>.txt`.

```python
# phase1_run.py  profile_model, lines 42-75
class Tee:
    def write(self, s):
        original_stdout.write(s)   # print to terminal
        captured.write(s)          # also collect in StringIO buffer
    def flush(self):
        original_stdout.flush()

sys.stdout = Tee()
compiled_fn = compile(exp.train_step, exp.graph_transformation)
compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
sys.stdout = original_stdout

# Write captured output to file
with open(f"records/stats_{model_name.lower()}.txt", "w") as f:
    f.write(captured.getvalue())
```

---

## Part 3: Results

### Profiling Statistics

| Model | Batch | Fwd (ms) | Bwd (ms) | Bwd/Fwd | Activation Mem |
|-------|-------|----------|----------|---------|----------------|
| ResNet18 | 16 | 21.0 | 104.2 | 5.0× | 347 MB |
| ResNet50 | 4  | 51.5 | 238.1 | 4.6× | 350 MB |
| Transformer | 4 | 34.2 | 163.6 | 4.8× | 19 MB |

The backward pass is consistently ~5× slower than forward, as expected
(each layer must compute two gradient directions).

### DummyModel Activation Lifetimes

The profiler found 19 activations on the DummyModel:
- 10 `relu` outputs: 0.40 MB each (1000 × 100 × 4 bytes)
- 9 transposed weight matrices `t_*`: 0.04 MB each (100 × 100 × 4 bytes)

The lifetime table shows the "reverse ladder" pattern clearly:
- `relu` (layer 1) is last used in forward at `addmm_1` but not needed until
  `mm_17` — near the very end of backward. Longest idle window.
- `relu_9` (last layer) is needed at `threshold_backward` — right at the
  start of backward. Shortest idle window.

This directly motivates checkpointing early-layer activations first.

### Peak Memory Bar Charts

Saved in `records/`:
- `peak_memory_resnet18.png`
- `peak_memory_resnet50.png`
- `peak_memory_transformer.png`

All three charts show activations (orange) as the dominant memory consumer,
growing linearly with batch size, while parameters (blue) and optimizer
states (red) remain flat. This confirms that activations are the correct
target for Phase 2's checkpointing algorithm.

---

## Part 4: What Phase 2 Will Use From This

The profiler exposes the following data per activation node, which feeds
directly into the Phase 2 μ-TWO selection algorithm:

| Field | Where it comes from |
|-------|---------------------|
| `activation_nodes` | Set of `fx.Node` objects classified as `ACT` |
| `act_last_fwd_use[a]` | Last forward user of activation `a` |
| `act_first_bwd_use[a]` | First backward user of activation `a` |
| `_tensor_sizes[a]` | Byte size of the activation tensor |
| `node_avg_runtime[n]` | Average ms to execute node `n` |

Given these, Phase 2 will compute a recomputation cost for each activation
(sum of runtimes of the subgraph needed to reproduce it) and compare it
against the memory saved by discarding it. The greedy algorithm from the
μ-TWO paper then selects the subset of activations to checkpoint.

---

## Part 5: How to Interpret the Results

### Reading the Per-Node Stats Table

Each row in the printed table looks like this:

```
Node                    Op       NodeType   Runtime(ms)  MemDelta(MB)
relu                    call_fun ACT              0.031         0.400
addmm_1                 call_fun OTHER            0.041         0.000
t_10                    call_fun ACT              0.002         0.040
```

**Node** — the name `fx` assigned to the operation in the graph.

**NodeType** — what role this node plays:
- `ACT`: this is an activation tensor that Phase 2 might checkpoint. Look at
  its `MemDelta` — that is exactly how much GPU memory would be freed if it
  were discarded.
- `GRAD`: a backward computation. These have non-zero runtimes but allocate
  memory only temporarily (their delta goes to zero by the end of backward).
- `PARAM` / `OPT` / `INPUT`: placeholders. Their `MemDelta` is zero because
  they are pre-allocated before the graph runs.
- `OTHER`: overhead ops (copy_, getitem, sentinel). Usually tiny runtime and
  zero memory delta.

**Runtime(ms)** — average GPU kernel time over the 3 profile iterations.
A node that takes `0.000 ms` is typically a metadata-only op (a `getitem`
slicing into a list result, for example) — no kernel is actually launched.
High-runtime nodes in the forward region are expensive to recompute; you want
to avoid checkpointing those.

**MemDelta(MB)** — how much GPU memory changed after this node executed.
- Positive: the node allocated a new tensor (e.g. `relu` output).
- Zero or negative: the node consumed an existing tensor and the allocator
  reused memory, or the node is an in-place op.
- The sum of all positive `MemDelta` values across `ACT` nodes equals the
  **total activation memory** printed at the bottom of the table.

**Summary lines at the bottom:**

```
Total forward  runtime : 3.551 ms
Total backward runtime : 23.752 ms
Total activation memory: 4.36 MB
```

- The ratio `backward / forward` is normally 2–5×. For the DummyModel it is
  ~6.7×; for the real models it settles around 4.7–5×. A ratio much larger
  than 5 would suggest an unusually expensive backward op worth investigating.
- `Total activation memory` is the peak amount that must be kept alive
  simultaneously for the backward pass. This is the budget Phase 2 is trying
  to reduce.

---

### Reading the Activation Lifetime Table

```
Activation              Size(MB)    last_fwd_use         first_bwd_use
relu                       0.400         addmm_1               mm_17
relu_1                     0.400         addmm_2               mm_15
relu_8                     0.400        addmm_9    threshold_backward_1
relu_9                     0.400       addmm_10      threshold_backward
```

**Size(MB)** — the raw byte cost of keeping this activation alive. Larger
means more memory saved if checkpointed.

**last_fwd_use** — the last operation in the forward pass that reads this
activation. After this node finishes, the activation sits idle until backward
starts. If `last_fwd_use` is the node that produced it (no other forward op
reads it), the idle window starts immediately.

**first_bwd_use** — the first backward operation that needs this activation.
The further this is from the top of the backward region, the longer the
activation has been idle.

**The idle window = graph index of `first_bwd_use` minus graph index of
`last_fwd_use`.** A large idle window means:
1. A long time passes between the activation being created and it being needed
   in backward.
2. If we checkpoint it, we have a large window in which the memory can be
   reclaimed.

In the DummyModel results above, `relu` (layer 1) has `last_fwd_use = addmm_1`
(very early in the graph) and `first_bwd_use = mm_17` (near the very end of
backward). This is the longest possible idle window — `relu` is the best
candidate to checkpoint. By contrast, `relu_9` (last layer) is needed almost
immediately in backward (`threshold_backward`), so checkpointing it would
save memory for almost no time.

The pattern reverses as you go deeper: **earlier layers = longer idle window =
better checkpoint candidates.**

---

### Reading the Peak Memory Bar Charts

Each bar chart (`peak_memory_<model>.png`) shows stacked bars for each
batch size tested:

```
Peak GPU memory (GB)
  │   ┌────────┐
  │   │ other  │
  │   ├────────┤
  │   │opt_stat│
  │   ├────────┤
  │   │ grads  │
  │   ├────────┤
  │   │  acts  │  ← dominant, grows with batch size
  │   ├────────┤
  │   │ params │  ← flat, batch-size independent
  └───┴────────┴── batch size
```

**What the colors mean:**

| Color  | Category    | What it represents |
|--------|-------------|-------------------|
| Blue   | `params`    | Model weight tensors — fixed size regardless of batch |
| Orange | `activations` | Forward-pass intermediate tensors — grows linearly with batch size |
| Green  | `gradients` | Backward-pass tensors — also grows with batch size |
| Red    | `opt_states`| Adam's exp_avg and exp_avg_sq — fixed size regardless of batch |
| Purple | `other`     | Everything else (loss, sentinel ops) — near zero |

**What to look for:**

1. **Activations dominate and grow linearly.** In all three models, the orange
   segment is the tallest and the only one that scales with batch size. This is
   the fundamental motivation for activation checkpointing.

2. **Params and optimizer states are flat.** Their memory is determined by the
   number of model parameters, not the batch size. Doubling the batch size does
   not change them at all. There is nothing to optimize there.

3. **The OOM point.** For ResNet50, the sweep stops before the largest batch
   size because the GPU runs out of memory. The last successful batch size
   shows how close you are to the hardware limit without any checkpointing.
   After Phase 2, the same batch sizes should fit because activation memory
   will be reduced.

4. **Comparing models.** ResNet18 vs ResNet50: ResNet50 has more layers and
   wider feature maps, so its activation memory grows faster per batch sample.
   The Transformer has a much smaller activation footprint because its
   intermediate tensors are embedding-sized, not image-sized.

---

### Quick Sanity Checks

These checks confirm the profiler is working correctly:

| Observation | What it confirms |
|-------------|-----------------|
| `MemDelta > 0` only on forward nodes | Backward reuses buffers; activations are freed as backward consumes them |
| `backward runtime ≈ 4–5× forward runtime` | Each backward layer computes two partial derivatives; rule of thumb is 2–3× compute but memory accesses add overhead |
| `PARAM` and `OPT` nodes all have `Runtime ≈ 0` | Placeholders do not execute any kernel; they are just pointers to pre-allocated tensors |
| Activation count matches layer count × 2 | Each `relu` output plus each transposed weight `t_*` feeds into the corresponding backward `mm` |
| Peak activation memory ÷ batch size is constant | Confirms linear scaling; a non-linear result would indicate a bug in the memory timeline simulation |
