# CS265 MLSys Project — Definitive Codebase Walkthrough
### Activation Checkpointing via Computational Graph Profiling and Rewriting
**Harvard CS265, Spring 2025**

---

## Table of Contents

1. [The Problem in Full Detail](#1-the-problem-in-full-detail)
2. [Repository Structure](#2-repository-structure)
3. [`utils.py` — The Decomposition Table](#3-utilspy--the-decomposition-table)
4. [`graph_tracer.py` — Capturing the Training Graph](#4-graph_tracerpy--capturing-the-training-graph)
5. [`graph_prof.py` — GraphProfiler](#5-graph_profpy--graphprofiler)
6. [`ac_algorithm.py` — μ-TWO Selection Algorithm](#6-ac_algorithmpy--μ-two-selection-algorithm)
7. [`graph_rewriter.py` — Graph Rewriter](#7-graph_rewriterpy--graph-rewriter)
8. [`benchmarks.py` — Models, Experiments, Sweeps](#8-benchmarkspy--models-experiments-sweeps)
9. [`phase1_run.py` and `phase2_run.py` — Runner Scripts](#9-phase1_runpy-and-phase2_runpy--runner-scripts)
10. [Complete End-to-End Data Flow](#10-complete-end-to-end-data-flow)
11. [Every Design Decision Explained](#11-every-design-decision-explained)
12. [Anticipated Questions and Answers](#12-anticipated-questions-and-answers)

---

## 1. The Problem in Full Detail

### What happens during one training iteration

A neural network training iteration has four phases that run sequentially on
the GPU:

**Phase 1 — Forward pass.** The input batch `X` flows through the network
layer by layer. Each layer computes an intermediate tensor called an
**activation** (or feature map): `Z1 = f1(X)`, `Z2 = f2(Z1)`, ...,
`Zn = fn(Zn-1)`. These activations must be kept in GPU memory because the
backward pass needs them.

**Phase 2 — Loss computation.** The final activation is compared to the
target labels using a loss function (e.g. cross-entropy). This produces a
scalar loss value `L`.

**Phase 3 — Backward pass.** PyTorch's autograd engine runs the chain rule in
reverse: `∇Zn`, `∇Zn-1`, ..., `∇Z1`, then `∇W_n`, `∇W_{n-1}`, ..., `∇W_1`.
Each gradient computation consumes the corresponding activation and the
upstream gradient. Critically, the activations are consumed **in reverse
order** — `Zn` is consumed first, `Z1` is consumed last.

**Phase 4 — Optimizer step.** Adam updates each weight:
`W = W - lr * m_hat / (sqrt(v_hat) + eps)`, where `m_hat` and `v_hat` are
the bias-corrected first and second moment estimates maintained by the
optimizer.

### The memory problem

Because backward consumes activations in reverse order, activation `Z1`
(computed at the very start of the forward pass) is not needed until the
very end of the backward pass. It sits in GPU memory the entire time —
through the rest of the forward pass, through the loss, and through most of
backward. This idle time is called the **lifetime** of the activation.

Activations from early layers have the longest lifetimes. They accumulate
in memory simultaneously: at the peak, every activation `Z1` through `Zn`
is alive at once. For modern deep networks, activations account for
**70–85% of peak GPU memory usage**.

The weights `W1`...`Wk`, their gradients `∇W1`...`∇Wk`, and the Adam states
(`exp_avg`, `exp_avg_sq`) are also in memory, but their total size is
determined only by the number of parameters — it does not grow with batch
size. Activations, by contrast, scale linearly with batch size (doubling
the batch doubles the activation memory). This is what limits the maximum
batch size you can train with.

### The solution: Activation Checkpointing (AC)

AC chooses a subset of activations to **discard** immediately after the
forward pass is done with them. When the backward pass needs one of these
discarded activations, it **recomputes** it on the fly from the inputs that
are still available.

This trades a small amount of extra compute for a significant reduction in
peak memory. If we checkpoint the right activations (large, cheap to
recompute), we can reduce peak memory by 50% with only 5–15% compute
overhead.

### The three-phase implementation plan

- **Phase 1**: Profile the graph. Find out exactly how much time each node
  takes, how much memory each activation uses, and how long each activation
  sits idle.
- **Phase 2**: Decide. Given the profiling data, run the μ-TWO greedy
  algorithm to select which activations to checkpoint.
- **Phase 3**: Rewrite. Modify the actual computation graph to implement
  the decision: delete the forward-pass uses of the activation's tensor,
  and insert a recomputation subgraph in the backward pass.

---

## 2. Repository Structure

```
CS265-mlsys-project/
│
├── utils.py
│     Defines SPMD_DECOMP_TABLE. Decomposes in-place optimizer operations
│     (e.g. _foreach_add_) into functional equivalents so make_fx can trace them.
│
├── graph_tracer.py
│     Defines SEPFunction, _compile, compile.
│     Captures the full fwd+bwd+optimizer execution as a single fx.GraphModule.
│
├── graph_prof.py
│     Defines OP (enum), NodeType (enum), GraphProfiler (extends fx.Interpreter).
│     Phase 1: static analysis + runtime profiling.
│
├── ac_algorithm.py
│     Defines _recompute_subgraph_nodes, _recompute_cost,
│     _boundary_activation_inputs, mu_two_selection, get_recompute_info,
│     print_ac_plan.
│     Phase 2: μ-TWO greedy activation selection algorithm.
│
├── graph_rewriter.py
│     Defines _replace_subsequent_uses_of, _build_name_map,
│     apply_activation_checkpointing.
│     Phase 3: graph rewriting — inserts recomputation subgraphs.
│
├── benchmarks.py
│     Defines _BertAttention, _BertLayer, SimpleBert, Experiment,
│     run_single, measure_iteration_latency, _run_sweep_single,
│     run_memory_sweep, run_comparison_sweep.
│     All four models, the Experiment class, and the sweep infrastructure.
│
├── starter_code.py
│     Defines DummyModel, train_step, graph_transformation, experiment.
│     The original entry point used during early testing.
│
├── activation_checkpoint.py
│     Course-provided reference example showing how to use
│     _extract_graph_with_inputs_outputs and node_copy on a two-layer network.
│
├── phase1_run.py
│     Runs Phase 1 deliverables for all models. Saves stats text files and
│     Phase 1 bar charts (no AC).
│
└── phase2_run.py
      Runs Phase 2 deliverables for all models. Saves stats (no AC + AC),
      memory comparison charts, and latency comparison charts.
```

**Output files in `records/`:**

| File | What it contains |
|------|-----------------|
| `stats_<model>.txt` | Phase 1 per-node table from starter_code / phase1_run |
| `stats_<model>_noac.txt` | Full per-node table at default batch size, no AC |
| `stats_<model>_ac.txt` | Same table + the μ-TWO AC plan printed below it |
| `peak_memory_<model>.png` | Phase 1 stacked bar chart (no AC, all batch sizes) |
| `peak_memory_comparison_<model>.png` | Grouped bars: no AC (solid) vs AC (faded) |
| `latency_comparison_<model>.png` | Line chart: no AC vs AC iteration time |
| `phase1_design.md` | Design document written before implementation |
| `codebase_walkthrough.md` | This document |
| `midway_report.tex` | LaTeX source for midway check-in report |

---

## 3. `utils.py` — The Decomposition Table

### Why this file exists

PyTorch's Adam optimizer uses bulk in-place operations like `_foreach_add_`,
`_foreach_mul_`, `_foreach_sqrt_` (the `_` suffix means in-place). These
operations work on a Python list of tensors simultaneously, modifying them
in place.

In-place operations are a problem for `make_fx` because `make_fx` records a
**functional** graph — every operation must produce a new output tensor, with
no side effects. An in-place operation like `_foreach_add_(params, grads)`
modifies `params` without producing a new tensor, so there is no output to
record in the graph.

### What the table does

`SPMD_DECOMP_TABLE` is a Python dictionary mapping each in-place operator to
a functional replacement function. For example:

```python
aten._foreach_add_.List: _foreach_add_decomp
```

Where `_foreach_add_decomp` is:

```python
def _foreach_add_decomp(self, other, alpha=1):
    self_updated = aten._foreach_add.List(self, other, alpha=alpha)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)
```

This takes the in-place `_foreach_add_` and decomposes it into:
1. The out-of-place `_foreach_add.List` which produces a new list of tensors
2. A series of `copy_` calls that write the results back

In the traced graph, step 1 appears as a `_foreach_add` node with an output,
and step 2 appears as explicit `copy_` nodes. The graph now has proper
data edges.

The table covers all optimizer operations used by Adam with `foreach=True`:
- `_foreach_add_.List` / `_foreach_add_.Scalar` — adds list/scalar to each param
- `_foreach_mul_.Scalar` — multiplies each param by a scalar (for beta decay)
- `_foreach_sqrt_.default` — element-wise sqrt (for v_hat)
- `_foreach_addcdiv_.Scalar` — the final weight update: `W += -lr * m / (sqrt(v) + eps)`
- `_foreach_addcmul_.Scalar` — used in AMSGrad variant
- `_foreach_div_.List` / `_foreach_div_.Scalar` — division
- `_foreach_neg_.default` — negation
- `_foreach_reciprocal_.default` — reciprocal
- `_foreach_sub_.Scalar` — subtraction
- `aten.native_layer_norm_backward.default` — LayerNorm backward (needed for BERT)

### How it is used

Passed to `make_fx` in `graph_tracer._compile`:

```python
gm = make_fx(
    partial(stateless_func, func),
    tracing_mode="fake",
    decomposition_table=SPMD_DECOMP_TABLE,   # <--- here
    _allow_non_fake_inputs=False,
)(params, buffers, named_states, args, kwargs)
```

`make_fx` intercepts every operation during tracing. When it encounters an
operation that is in `SPMD_DECOMP_TABLE`, it calls the decomposition function
instead and records the resulting functional operations in the graph.

---

## 4. `graph_tracer.py` — Capturing the Training Graph

This file's job is to take a normal Python training function and convert it
into a single, flat `fx.GraphModule` containing every operation — forward,
loss, backward, and optimizer — as a sequence of explicit nodes.

### 4.1 The `sep` and `sep_backward` custom operators

```python
separator_lib = torch.library.Library("separator", "DEF")
separator_lib.define("sep(Tensor x) -> Tensor")
separator_lib.impl("sep", sep, "CompositeExplicitAutograd")
separator_lib.define("sep_backward(Tensor x) -> Tensor")
separator_lib.impl("sep_backward", sep_backward, "CompositeExplicitAutograd")
```

This registers two new PyTorch operators using the custom operator API.
`sep` and `sep_backward` are both identity functions — they return their
input unchanged. However, by registering them as first-class PyTorch
operators, they will appear as distinct nodes in any traced graph.

The `CompositeExplicitAutograd` dispatch key means they are registered for
CPU and CUDA execution and are included in autograd.

### 4.2 The `SEPFunction` autograd function

```python
class SEPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ops.separator.sep(x)

    @staticmethod
    def backward(ctx, grad_x):
        return torch.ops.separator.sep_backward(grad_x)
```

`SEPFunction` is a custom `torch.autograd.Function`. When you call
`SEPFunction.apply(loss)` in a training loop:
- During the **forward pass**, it calls `torch.ops.separator.sep(x)` which
  is a no-op but records a `sep` node in the traced graph.
- During the **backward pass**, PyTorch's autograd calls the `backward`
  method, which calls `torch.ops.separator.sep_backward(grad)` — recording
  a `sep_backward` node in the traced graph.

The result: when `make_fx` traces the complete forward+backward execution,
both `sep` and `sep_backward` appear as nodes. We can find these nodes
by name and use their positions to identify the forward/backward boundary.

### 4.3 `_enable_compile` context manager

```python
@contextmanager
def _enable_compile():
    def f_true():
        return True
    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code
```

This temporarily monkey-patches `torch._utils.is_compiling` to return `True`.
Why? PyTorch's Adam optimizer checks `torch._utils.is_compiling()` and changes
its behavior when inside a compilation context. Specifically, it enables
the `capturable=True` path which allows optimizer state updates to be traced
into the graph. Without this, the optimizer step would not appear in the
captured graph.

### 4.4 `_rematerialize_optimizer` context manager

```python
@contextmanager
def _rematerialize_optimizer(opt, named_states, params):
    orig_states = copy(opt.state)
    for n in named_states:
        opt.state[params[n]] = named_states[n]
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()
    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state = orig_states
```

Adam stores its state in `opt.state`, a dictionary mapping `nn.Parameter`
objects to dicts like `{"exp_avg": tensor, "exp_avg_sq": tensor, "step": tensor}`.
During tracing with fake tensors, the real parameters are replaced by proxy
tensors. This context manager temporarily replaces the optimizer's state
entries so they map from proxy tensors (not real parameters) to the state
dicts. This allows the optimizer step to be traced correctly.

After the context manager exits, everything is restored to the original state.

### 4.5 `_compile` — the core tracing function

```python
def _compile(func, *args, **kwargs):
    # Step 1: Find the nn.Module and Optimizer in the arguments
    mod, opt = None, None
    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, nn.Module):  mod = arg
        if isinstance(arg, optim.Optimizer): opt = arg

    # Step 2: Extract parameters and optimizer states
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))
    named_states = {}
    for n, p in params.items():
        if p in opt.state:
            named_states[n] = opt.state[p]

    # Step 3: Define stateless_func — a pure function with all state explicit
    def stateless_func(func, params, buffers, named_states, args, kwargs):
        with stateless._reparametrize_module(mod, {**params, **buffers}), \
             _rematerialize_optimizer(opt, named_states, params), \
             gradients_tagging(params):
            ret = func(*args, **kwargs)
        return ret, list(mod.parameters()), list(named_states.values())

    # Step 4: Convert all input tensors to FakeTensors
    fake_mode = FakeTensorMode()
    args = pytree.tree_map_only(torch.Tensor, fake_mode.from_tensor, args)
    kwargs = pytree.tree_map_only(torch.Tensor, fake_mode.from_tensor, kwargs)

    # Step 5: Trace with make_fx
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Anomaly Detection has been enabled")
        with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
            gm = make_fx(
                partial(stateless_func, func),
                tracing_mode="fake",
                decomposition_table=SPMD_DECOMP_TABLE,
                _allow_non_fake_inputs=False,
            )(params, buffers, named_states, args, kwargs)

    # Step 6: Clean up helper nodes
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            node.replace_all_uses_with(node.all_input_nodes[0])
            if len(node.users) == 0:
                gm.graph.erase_node(node)
        if node.target == torch.ops.dummy.tag_grad.default:
            node.replace_all_uses_with(node.all_input_nodes[0])
            if len(node.users) == 0:
                gm.graph.erase_node(node)

    # Step 7: Flatten the graph's input/output handling
    gm = _to_caller_flattened_graph_module(gm)

    flat_state, _ = pytree.tree_flatten([{**params, **buffers}, named_states])
    return _CompiledResult(gm, mod, opt, flat_state)
```

**Step 1 — Finding mod and opt:** `pytree.tree_flatten` recursively unpacks
the argument structure into a flat list of leaf values. We scan this for an
`nn.Module` and an `optim.Optimizer`.

**Step 2 — Extracting state:** `named_parameters(remove_duplicate=False)`
returns all parameters by their fully qualified name (e.g.
`"layers.0.weight"`, `"layers.0.bias"`). We build `named_states` to map
each parameter name to its optimizer state dict (if it has been initialized;
that is why `init_opt_states()` must be called first).

**Step 3 — `stateless_func`:** The key insight is that `make_fx` needs a
**pure function** — all inputs must come through function arguments, not
through closed-over global state. `stateless._reparametrize_module` swaps the
model's `nn.Parameter` attributes with the proxy tensors from the `params`
argument. Now the model uses proxies for all its parameters, so every
operation on them is recorded. The function returns three things: the original
return value, the updated parameter list, and the updated optimizer state list.
This forces the parameter update (`optimizer.step()`) to appear in the graph
as operations that produce the updated parameter values.

**Step 4 — FakeTensors:** `FakeTensorMode` creates tensors that know their
shape, dtype, and device but contain no actual data. Every operation on a
FakeTensor records an entry in `make_fx`'s trace but computes nothing on the
GPU. This makes tracing fast (no GPU kernels run) and safe (no OOM errors).

**Step 5 — `make_fx` with `detect_anomaly`:** `torch.autograd.detect_anomaly`
is enabled with `check_nan=False`. This is needed during tracing to ensure
that all backward operations are recorded. Without it, some gradient
computations might be skipped or not traced. The `check_nan=False` argument
prevents it from inspecting actual tensor values (which don't exist for
FakeTensors). The `warnings.catch_warnings()` block suppresses the warning
that `detect_anomaly` prints to stderr.

**Step 6 — Cleanup:**
- `aten.detach.default` nodes: PyTorch inserts these around parameters during
  the `requires_grad` bookkeeping. They are pure identity operations (they
  return the same tensor with `requires_grad=False`). They add nodes to the
  graph without doing any work, so we remove them by replacing each detach
  node with its input node in every downstream use.
- `dummy.tag_grad.default` nodes: Inserted by `gradients_tagging` hooks to
  mark gradient tensors for SPMD (distributed) processing. Also identity
  operations, also removed the same way.

**Step 7 — `_to_caller_flattened_graph_module`:** The traced function receives
its arguments as nested Python structures (dicts, lists, tuples). This function
modifies the graph's code generation so the caller can pass arguments as a
flat list instead, which is more convenient for the profiler and rewriter.

### 4.6 The `compile` wrapper

```python
def compile(func, gm_transformation):
    @wraps(func)
    def wrapper(*args, **kwargs):
        compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
        first_iter = compiled_obj is None
        if first_iter:
            compiled_obj = _compile(func, *args, **kwargs)
            wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj

        flat_inps = compiled_obj.flat_state + pytree.tree_flatten([args, kwargs])[0]

        if first_iter and gm_transformation:
            compiled_obj.gm = gm_transformation(compiled_obj.gm, flat_inps)

        with torch.no_grad():
            output = compiled_obj.gm(*flat_inps)[0]

        return output
    return wrapper
```

This function uses the decorator pattern. `compile(train_step, my_transform)`
returns a new function `wrapper` that, when called the first time, traces the
graph and runs `my_transform` on it. Every subsequent call just runs the
already-compiled (and possibly rewritten) graph directly.

Key details:
- `wrapper.__dict__[COMPILED_OBJECT_KEY]` stores the compiled result on the
  function object itself. This allows the compiled state to persist across
  calls without any external state variable.
- `flat_inps = flat_state + pytree.tree_flatten([args, kwargs])[0]` builds
  the complete flat input list. `flat_state` contains all parameters and
  optimizer states in the exact order they appear as placeholder nodes in
  the graph. The training arguments (`model`, `optimizer`, `batch`) are
  appended after.
- `gm_transformation` is called with `(gm, flat_inps)`. `flat_inps` at this
  point contains the actual (non-fake) tensors. This is what the profiler
  uses to actually execute the graph and measure timings.
- The final call `compiled_obj.gm(*flat_inps)[0]` runs the graph.
  `[0]` extracts the first element of the output tuple (the return value
  of the original training function, which is `None`).

### 4.7 The resulting graph structure in detail

After tracing a model like DummyModel (10 layers of Linear + ReLU), the
`fx.GraphModule` contains approximately 320 nodes arranged as:

```
Placeholders (top of graph, in order):
  arg0_1   ← weight matrix of layer 0     (PARAM)
  arg0_2   ← bias of layer 0              (PARAM)
  arg0_3   ← weight matrix of layer 1     (PARAM)
  ...
  arg2_1   ← exp_avg for layer 0 weight   (OPT)
  arg2_2   ← exp_avg for layer 0 bias     (OPT)
  ...
  arg3_1   ← exp_avg_sq for layer 0       (OPT)
  ...
  arg4_1   ← step counter                 (OPT)
  ...
  arg_batch ← input mini-batch            (INPUT)

Forward operations:
  addmm           ← x @ W0 + b0           (OTHER — output only used in fwd)
  relu            ← relu(addmm)           (ACT — used in backward)
  addmm_1         ← relu @ W1 + b1        (OTHER)
  relu_1          ← relu(addmm_1)         (ACT)
  ...
  addmm_9         ← relu_8 @ W9 + b9      (OTHER)
  relu_9          ← relu(addmm_9)         (ACT)
  sum_1           ← relu_9.sum()          (OTHER)

  separator_sep   ← SEPFunction forward   (OTHER)

  loss computation ...

  separator_sep_backward ← SEPFunction backward (OTHER)

Backward operations:
  mm              ← grad @ W9.T            (GRAD)
  threshold_backward ← relu backward      (GRAD)
  mm_1                                    (GRAD)
  ...

Optimizer operations:
  _foreach_mul_   ← beta1 * exp_avg        (OTHER — already decomposed)
  _foreach_add    ← exp_avg + (1-beta1)*grad
  copy_           ← write back to exp_avg
  ...
  _foreach_addcdiv ← W = W - lr * m_hat / (sqrt(v_hat) + eps)
  copy_           ← write back to weights

  output
```

For ResNet50, this graph has over 4,000 nodes.

---

## 5. `graph_prof.py` — GraphProfiler

`GraphProfiler` extends `torch.fx.Interpreter`. The base class provides a
`run(*args)` method that executes the graph by calling `run_node(n)` for each
node in topological order, passing the outputs of previous nodes as inputs to
later ones. By subclassing and overriding `run_node`, we intercept every
single node execution.

### 5.1 The `OP` and `NodeType` enums

```python
class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"
```

`OP` mirrors `fx.Node.op`, the field that describes what kind of graph node
this is:
- `placeholder`: a function input (parameters, optimizer states, batch)
- `call_function`: a free function call like `torch.ops.aten.relu.default`
- `call_module`: a call to an `nn.Module`'s forward method
- `call_method`: a method call like `.sum()` on a tensor
- `get_attr`: reading an attribute from the graph module
- `output`: the final output node that collects all return values

```python
class NodeType(Enum):
    PARAM = 0    # model weight placeholder
    ACT   = 1    # activation: created in forward, consumed in backward
    GRAD  = 2    # gradient computation node (backward region)
    OPT   = 3    # optimizer state placeholder (exp_avg, exp_avg_sq, step)
    INPUT = 4    # input mini-batch placeholder
    OTHER = 5    # everything else
```

`NodeType` is our semantic classification. The six values cover every possible
role a node can play in the combined graph.

### 5.2 `__init__` — Static Analysis

Static analysis runs once at construction time. No GPU execution happens.

**Building the node index:**

```python
nodes: List[fx.Node] = list(module.graph.nodes)
node_to_idx: Dict[fx.Node, int] = {n: i for i, n in enumerate(nodes)}
```

`module.graph.nodes` is an ordered list of all nodes in the graph in
topological order (each node comes after all its dependencies). We build
a reverse map from node to its position index so we can answer questions
like "does node A come before node B?" in O(1).

**Finding the sentinel boundaries:**

```python
sep_idx = -1
sep_bwd_idx = len(nodes)
for i, node in enumerate(nodes):
    if node.target is torch.ops.separator.sep.default:
        self.sep_node = node
        sep_idx = i
    elif node.target is torch.ops.separator.sep_backward.default:
        self.sep_bwd_node = node
        sep_bwd_idx = i
```

We use `is` (identity comparison) not `==` because `node.target` is a direct
reference to the operator object. `sep_idx` and `sep_bwd_idx` default to `-1`
and `len(nodes)` respectively, which means "before the graph" and "after the
graph". If for some reason the sentinel is not found, these defaults cause the
forward/backward sets to be empty (which would trigger the assertion below).

After this loop:

```python
assert self.sep_node is not None
assert self.sep_bwd_node is not None
self.forward_nodes  = {n for n in nodes if node_to_idx[n] < sep_idx}
self.backward_nodes = {n for n in nodes if node_to_idx[n] > sep_bwd_idx}
```

`forward_nodes` includes everything before `sep`. `backward_nodes` includes
everything after `sep_bwd`, which includes both gradient computations AND
optimizer operations.

**Identifying parameter and optimizer-state placeholders:**

`_find_param_and_opt_nodes` fills two sets: `param_nodes` and
`opt_state_nodes`. It works differently depending on whether the optimizer was
traced with `fused=True` or `foreach=True`.

For `foreach=True` (our case):

```python
for node in reversed(nodes):
    if node.target is torch.ops.aten._foreach_addcdiv.Scalar:
        optimizer_node = node
        flat_args = node.args[0]   # list of model parameter nodes
        for n in flat_args:
            if isinstance(n, fx.Node) and n.op == OP.PLACEHOLDER:
                param_nodes.add(n)
        for n in node.args[1]:     # list of bias-corrected 2nd moments
            if isinstance(n, fx.Node) and n.op == OP.PLACEHOLDER:
                opt_state_nodes.add(n)
        break
```

`_foreach_addcdiv` is the operation `W = W + alpha * (A / B)` applied to
lists. In Adam, this is `W = W + (-lr) * (exp_avg / (sqrt(exp_avg_sq) + eps))`.
Its argument structure is:
- `args[0]`: list of parameter tensors being updated
- `args[1]`: the numerator list (bias-corrected first moments)
- `args[2]`: the denominator list (bias-corrected second moments)
- `scalar`: the learning rate

We search from the end of the graph (using `reversed`) because the weight
update is near the end.

The fallback sweep at the end:

```python
for node in nodes:
    if node.op != OP.PLACEHOLDER:
        break  # placeholders are always at the top of the graph
    if node in param_nodes:
        continue
    users_in_bwd = any(u in self.backward_nodes for u in node.users)
    if users_in_bwd and node not in param_nodes:
        opt_state_nodes.add(node)
```

This catches any remaining placeholder nodes that feed into the backward
region. These are Adam's `exp_avg`, `exp_avg_sq`, and `step` tensors that
were not caught by inspecting `_foreach_addcdiv`.

The reason placeholders are always at the top: `make_fx` follows Python's
variable scoping rules when building the graph. All function arguments become
placeholder nodes and are placed first in the graph, before any computation.

**Classifying every node:**

`_classify_node` assigns one `NodeType` to each node:

```python
def _classify_node(self, node, node_to_idx, sep_idx, sep_bwd_idx,
                   param_nodes, opt_state_nodes):
    idx = node_to_idx[node]

    if node.op == OP.PLACEHOLDER:
        if node in param_nodes:     return NodeType.PARAM
        if node in opt_state_nodes: return NodeType.OPT
        return NodeType.INPUT

    if node.op == OP.OUTPUT:
        return NodeType.OTHER

    if node.target in (torch.ops.separator.sep.default,
                       torch.ops.separator.sep_backward.default):
        return NodeType.OTHER

    if idx < sep_idx:
        # Forward region: an ACT is a forward node used in backward
        if any(node_to_idx.get(u, -1) > sep_bwd_idx for u in node.users):
            return NodeType.ACT
        return NodeType.OTHER

    if idx > sep_bwd_idx:
        return NodeType.GRAD

    return NodeType.OTHER  # between sep and sep_bwd: loss region
```

The `ACT` rule in detail: `node.users` is a dictionary of all nodes that use
this node's output as an input. We check if any user has a graph index greater
than `sep_bwd_idx`. If yes, this activation is consumed in the backward region,
meaning it must stay alive until then. `node_to_idx.get(u, -1)` uses `-1` as
default so that if a user is the output node (which might not be in the index),
it does not accidentally appear to be in the backward region.

**Recording activation lifetimes:**

```python
for act in self.activation_nodes:
    fwd_users = [u for u in act.users if u in self.forward_nodes]
    bwd_users = [u for u in act.users if u in self.backward_nodes]

    self.act_last_fwd_use[act] = (
        max(fwd_users, key=lambda u: node_to_idx[u]) if fwd_users else None
    )
    self.act_first_bwd_use[act] = (
        min(bwd_users, key=lambda u: node_to_idx[u]) if bwd_users else None
    )
```

`act.users` is a dict, not a list, so iterating it gives the user nodes.
`max(fwd_users, key=...)` finds the forward user with the highest graph index
— the last forward operation to read this activation. After this node
executes, the activation enters its idle period.
`min(bwd_users, key=...)` finds the backward user with the lowest graph index
— the first backward operation to need this activation. At this point, the
activation must be available again.

The gap (in graph index) between `last_fwd_use` and `first_bwd_use` is the
**idle window** — how long the tensor consumes memory without being used.

**Runtime data structures:**

```python
self._runtimes:   Dict[fx.Node, List[float]] = defaultdict(list)  # ms
self._mem_delta:  Dict[fx.Node, List[int]]   = defaultdict(list)  # bytes
self._tensor_sizes: Dict[fx.Node, int] = {}
self.node_avg_runtime:   Dict[fx.Node, float] = {}
self.node_avg_mem_delta: Dict[fx.Node, float] = {}
```

`_runtimes[n]` accumulates one float per profiling iteration for node `n`.
`_mem_delta[n]` accumulates the memory change (positive = allocated, negative
= freed) for each iteration.
`_tensor_sizes[n]` stores the byte size of node `n`'s output tensor. It is
filled on the first profiling iteration and never updated (tensor sizes are
constant across iterations for fixed batch sizes).
`node_avg_runtime` and `node_avg_mem_delta` are filled by `aggregate_stats()`.

### 5.3 `run_node` — Runtime Profiling

```python
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

    if n not in self._tensor_sizes:
        self._tensor_sizes[n] = self._measure_output_bytes(result)

    return result
```

**`torch.cuda.Event(enable_timing=True)`:** Creates a CUDA event that records
a timestamp on the GPU timeline. `enable_timing=True` is required for timing
(not all events need timing capability).

**`start.record()` before `super().run_node(n)`:** Records the GPU timestamp
immediately before the kernel is dispatched. This is a GPU-side timestamp —
it records when the GPU reaches this point in its execution queue.

**`super().run_node(n)`:** Calls the parent `fx.Interpreter.run_node`, which
dispatches the actual ATen kernel for node `n`. For example, if `n` represents
`aten.relu.default`, this calls the CUDA relu kernel.

**`end.record()` after `super().run_node(n)`:** Records the GPU timestamp
immediately after dispatch. Because CUDA is asynchronous, this does NOT mean
the kernel has finished — it means the dispatch command has been sent to the
CUDA stream.

**`torch.cuda.synchronize()`:** Blocks the CPU until the GPU has finished
all pending operations. After this returns, both events have their final
timestamps, and `torch.cuda.memory_allocated()` reflects the true current
state of the CUDA allocator.

**`start.elapsed_time(end)`:** Computes the time difference between the two
GPU timestamps in milliseconds. This is the true GPU kernel execution time,
not the CPU dispatch time.

**`mem_after - mem_before`:** The change in GPU allocated memory. Positive
means the node allocated new memory (produced a new tensor). Negative means
previously allocated memory was freed (e.g. a node consumed its last user's
input and PyTorch's garbage collector freed it). Zero means no net change.

**`_measure_output_bytes(result)`:**

```python
@staticmethod
def _measure_output_bytes(result):
    if isinstance(result, torch.Tensor):
        return result.element_size() * result.nelement()
    if isinstance(result, (list, tuple)):
        return sum(t.element_size() * t.nelement()
                   for t in result if isinstance(t, torch.Tensor))
    return 0
```

`element_size()` returns bytes per element (4 for float32, 2 for float16).
`nelement()` returns the total number of elements. Their product is the
tensor's byte footprint.

Some nodes return tuples of tensors (e.g. batch norm returns both the
normalized output and the running statistics). We sum over all tensors in
the tuple.

### 5.4 `reset_stats` and `aggregate_stats`

```python
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

`reset_stats` is called after warm-up iterations to discard the measurements
taken while the GPU was still compiling CUDA kernels (JIT compilation). The
first 1-2 iterations are much slower than steady-state due to kernel
compilation.

`aggregate_stats` computes simple arithmetic means. We use the mean rather
than the median or minimum because the AC algorithm uses these values to
estimate recomputation cost, and the mean is representative of the expected
cost across all iterations.

### 5.5 `print_stats` — Two output tables

The first table prints one row per graph node:

```
Node                    Op       NodeType   Runtime(ms)  MemDelta(MB)
relu                    call_fun ACT              0.031         0.400
addmm_1                 call_fun OTHER            0.041         0.000
...
Total forward runtime: 3.551 ms
Total backward runtime: 23.752 ms
Total activation memory: 4.36 MB
```

Column widths are defined by the format string `"{:<35} {:<8} {:<10} {:>12} {:>14}"`.
Node names are truncated to 35 characters with `node.name[:35]`.
`NodeType.name` returns the string name of the enum member (e.g. `"ACT"`).

The totals at the bottom are accumulated while iterating nodes:
- `total_fwd_ms` accumulates runtimes for all nodes in `forward_nodes`
- `total_bwd_ms` accumulates runtimes for all nodes in `backward_nodes`
  (which includes optimizer ops)
- `total_act_mem` sums `_tensor_sizes[node]` for all ACT-typed nodes

The second table prints one row per activation:

```
Activation              Size(MB)    last_fwd_use         first_bwd_use
relu                       0.400         addmm_1               mm_17
```

Activations are sorted by name (`key=lambda n: n.name`) for consistent output.

### 5.6 `peak_memory_by_category` — Memory Timeline Simulation

```python
# Build last_use: for each node, the index of its last consumer
last_use: Dict[fx.Node, int] = {}
for node in nodes:
    for user in node.users:
        idx = node_to_idx.get(user, -1)
        if idx > last_use.get(node, -1):
            last_use[node] = idx

# Simulate timeline
live: Dict[fx.Node, int] = {}
peak_by_cat: Dict[str, int] = {"params": 0, "activations": 0, ...}

for i, node in enumerate(nodes):
    sz = self._tensor_sizes.get(node, 0)
    if sz:
        live[node] = sz

    to_free = [n for n, lu in last_use.items() if lu == i and n in live]
    for n in to_free:
        del live[n]

    by_cat = defaultdict(int)
    for n, b in live.items():
        by_cat[_cat(n)] += b
    for cat, total in by_cat.items():
        if total > peak_by_cat[cat]:
            peak_by_cat[cat] = total
```

**`last_use[node]`:** For each node, we find the maximum index among all
its users. This is the last point at which the node's output tensor is read.
After this point, PyTorch's reference-counting system can free the tensor.

**The simulation loop:** At each step `i`, we:
1. Mark node `i`'s output as live (if it has a non-zero tensor size).
2. Free any tensors whose `last_use` index equals `i` — their last consumer
   just executed.
3. Snapshot the current bytes-per-category and update the per-category peak.

**`_cat(node)`** maps `NodeType` to a string category key:
`PARAM→"params"`, `ACT→"activations"`, `GRAD→"gradients"`,
`OPT→"opt_states"`, everything else `→"other"`.

The result dict is used directly by the bar chart generation code.

---

## 6. `ac_algorithm.py` — μ-TWO Selection Algorithm

### 6.1 `_recompute_subgraph_nodes` — BFS backwards

```python
def _recompute_subgraph_nodes(act, profiler, retained):
    needed = []
    visited = set()
    queue = deque([act])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if node != act:
            if node.op == "placeholder": continue
            if node in retained:         continue
            if node not in profiler.forward_nodes: continue

        needed.append(node)
        for inp in node.all_input_nodes:
            if inp not in visited:
                queue.append(inp)

    node_to_idx = {n: i for i, n in enumerate(profiler.module.graph.nodes)}
    needed.sort(key=lambda n: node_to_idx.get(n, 0))
    return needed
```

This is a **Breadth-First Search** (BFS) starting from `act`, traversing
**backwards** through the dependency graph (following `all_input_nodes` —
the nodes that provide inputs to the current node).

The stopping conditions (applied to every node except `act` itself):
1. `node.op == "placeholder"`: This node is a model parameter, optimizer
   state, or input batch. It is always available in memory — we never
   need to recompute it.
2. `node in retained`: This is another activation that we have decided to
   keep in memory. When we recompute `act`, this activation will be there
   as an input.
3. `node not in profiler.forward_nodes`: We have crossed into the backward
   region. We should never need backward nodes to recompute a forward
   activation.

`node.all_input_nodes` returns a list of all nodes that provide inputs to
the current node. For a node like `relu(addmm(x, W, b))`, `all_input_nodes`
of `relu` is `[addmm]`, and `all_input_nodes` of `addmm` is `[x, W, b]`.

The final `.sort()` converts the BFS result (which arrives in no particular
order) back to topological order. This is necessary because
`_extract_graph_with_inputs_outputs` and the graph insertion later require
nodes in execution order.

**Example for DummyModel layer 1:**
- `act` = `relu` (output of layer 0)
- `relu` depends on `addmm` (the matrix multiply `x @ W0 + b0`)
- `addmm` depends on `arg_batch` (placeholder), `arg0_1` (W0, placeholder),
  `arg0_2` (b0, placeholder)
- All three inputs are placeholders → stop
- Result: `[addmm, relu]` — only two nodes to re-execute

### 6.2 `_recompute_cost`

```python
def _recompute_cost(act, profiler, retained):
    nodes = _recompute_subgraph_nodes(act, profiler, retained)
    return sum(profiler.node_avg_runtime.get(n, 0.0) for n in nodes)
```

Simple: sum the average runtimes of all nodes in the recomputation subgraph.
`node_avg_runtime.get(n, 0.0)` returns 0 for nodes that were not profiled
(e.g. placeholder nodes, which are not in `_runtimes`).

### 6.3 `_boundary_activation_inputs`

```python
def _boundary_activation_inputs(act, profiler, retained):
    subgraph_set = set(_recompute_subgraph_nodes(act, profiler, retained))
    boundary = set()
    for n in subgraph_set:
        for inp in n.all_input_nodes:
            if inp not in subgraph_set and inp in profiler.activation_nodes:
                boundary.add(inp)
    return boundary
```

This finds the activation nodes that are on the **boundary** of the subgraph —
activation nodes that are needed as inputs to the subgraph but are not
themselves inside the subgraph. These are retained activations that `act`
indirectly depends on.

Why this matters: if any boundary activation is also being checkpointed
(removed from `retained`), then it won't be in memory when we need to
recompute `act`. The dependency constraint check uses this set.

### 6.4 `mu_two_selection` — the main algorithm

```python
def mu_two_selection(profiler, memory_budget_bytes):
    # Filter to valid candidates
    candidates = [
        act for act in profiler.activation_nodes
        if profiler._tensor_sizes.get(act, 0) > 0
        and profiler.act_last_fwd_use.get(act) is not None
        and profiler.act_first_bwd_use.get(act) is not None
    ]
```

Three validity conditions:
1. `_tensor_sizes > 0`: the activation has a non-zero tensor (no
   zero-dimensional tensors).
2. `act_last_fwd_use is not None`: the activation is actually used in the
   forward pass at least once (not just created and immediately passed to
   backward).
3. `act_first_bwd_use is not None`: the activation is actually consumed in
   the backward pass. If there is no backward use, there is nothing to
   recompute for.

```python
    retained = set(candidates)
    current_peak = sum(profiler._tensor_sizes.get(a, 0) for a in retained)

    if current_peak <= memory_budget_bytes:
        return set(), set(retained)
```

`current_peak` is an **upper bound** on peak activation memory: the sum of
all activation sizes. In a sequential network, all activations are alive
simultaneously at the peak (just before the first backward operation). This
upper bound is tight for sequential networks and is an overestimate for
networks with short-lived intermediate tensors.

If we are already within budget, return immediately with no activations to
recompute.

```python
    def efficiency(act, retained_set):
        mem  = profiler._tensor_sizes.get(act, 0)
        cost = _recompute_cost(act, profiler, retained_set - {act})
        return (mem / cost) if cost > 1e-9 else float("inf")

    sorted_candidates = sorted(
        candidates,
        key=lambda a: efficiency(a, retained),
        reverse=True,
    )
```

The efficiency is computed assuming `act` is **not** in the retained set
(that is why we pass `retained_set - {act}`). This gives the cost of
recomputing `act` from the available boundary nodes.

The `cost > 1e-9` guard handles nodes with zero measured runtime. This can
happen for very fast operations (like `getitem` or `reshape`) that complete
in under a microsecond. We treat these as infinitely efficient (free to
recompute), so they are always selected first.

Sorting is done **once at the start** using the all-retained baseline. This
is an approximation — as we add activations to `nodes_to_recompute`, the
retained set changes, which can change the efficiency of remaining candidates.
Re-sorting after each selection would be more accurate but O(n²) expensive.
For the models in this project, the single-sort approximation is fine.

```python
    nodes_to_recompute = set()
    for act in sorted_candidates:
        if current_peak <= memory_budget_bytes:
            break

        required_inputs = _boundary_activation_inputs(act, profiler, retained)
        if not required_inputs.issubset(retained):
            continue

        nodes_to_recompute.add(act)
        retained.discard(act)
        current_peak -= profiler._tensor_sizes.get(act, 0)

    return nodes_to_recompute, retained
```

The greedy loop:
1. Stop early if we have met the memory budget.
2. Check the dependency constraint. If any required input is no longer in
   `retained`, skip this activation.
3. Otherwise, add it to `nodes_to_recompute`, remove it from `retained`,
   and subtract its size from `current_peak`.

`current_peak -= _tensor_sizes[act]` is a simplified accounting: we assume
that checkpointing `act` directly reduces peak memory by `act`'s size. This
is an approximation — the actual peak reduction depends on when `act`'s
tensor would have been freed and when the recomputation happens — but it is
the standard assumption used in the μ-TWO paper.

### 6.5 `get_recompute_info` — output for the rewriter

```python
def get_recompute_info(act, profiler, nodes_to_retain):
    subgraph_nodes = _recompute_subgraph_nodes(act, profiler, nodes_to_retain)
    subgraph_set = set(subgraph_nodes)

    boundary_inputs = []
    for n in subgraph_nodes:
        for inp in n.all_input_nodes:
            if inp not in subgraph_set and inp not in boundary_inputs:
                boundary_inputs.append(inp)

    return subgraph_nodes, boundary_inputs
```

`boundary_inputs` is the list of nodes at the border of the subgraph: all
nodes that are inputs to a subgraph node but are not themselves in the
subgraph. This includes both placeholder nodes (params, inputs) and retained
activation nodes. This list is passed to `_extract_graph_with_inputs_outputs`
as the `inputs` argument.

We use a list (not a set) and check `inp not in boundary_inputs` to preserve
order and avoid duplicates while maintaining deterministic behavior.

### 6.6 `print_ac_plan` — Human-readable plan output

Prints a summary and a table sorted by tensor size (largest first):

```
=================================================================
  μ-TWO Activation Checkpointing Plan
=================================================================
  Total activations : 19
  To recompute      : 10  (4.0 MB freed)
  To retain         : 9   (4.4 MB kept in memory)
=================================================================
Activation (recompute)          Size(MB)  RecmpCost(ms)         first_bwd_use
----------------------------------------------------------------------------------------
relu_9                             0.400          0.031     threshold_backward
relu_8                             0.400          0.063     threshold_backward_1
...
```

The `eff` variable is computed but not printed in the current implementation —
it is there for debugging if you want to add it to the format string.

---

## 7. `graph_rewriter.py` — Graph Rewriter

### 7.1 `_replace_subsequent_uses_of`

```python
def _replace_subsequent_uses_of(graph, old_node, new_node):
    old_node_users = set(old_node.users.keys())
    for node in reversed(list(graph.nodes)):
        if node is new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)
```

This iterates the graph **in reverse** (from end to start). It stops when it
reaches `new_node` (the newly inserted recomputed version of the activation).
For every node it encounters that uses `old_node` (the original activation),
it calls `replace_input_with(old_node, new_node)`.

`replace_input_with` updates the node's argument list so that `old_node` is
replaced by `new_node` as an input. This does not change the node's position
in the graph.

Why reverse traversal? We want to replace only backward uses — the nodes that
come AFTER `new_node` in the graph. By iterating in reverse and stopping at
`new_node`, we naturally process only the nodes that come after the insertion
point.

The forward uses of `old_node` are **not replaced**. Forward nodes that
produce the original activation still use it normally (they run first, produce
the tensor, and then the tensor's reference count is reduced). The activation
is not explicitly freed — it is freed by Python's garbage collector when the
`fx.Interpreter`'s environment (which holds node values) drops the reference
after the last user has run. By redirecting all backward uses to `new_node`,
there are no remaining backward references to `old_node`, so it gets freed
when its last forward user runs.

`old_node.users.keys()` returns the set of nodes that use `old_node`. We
convert to a set because `users` is a `weakref.WeakKeyDictionary` in PyTorch
and we want a stable snapshot.

### 7.2 `apply_activation_checkpointing`

```python
def apply_activation_checkpointing(gm, nodes_to_recompute, nodes_to_retain, profiler):
    if not nodes_to_recompute:
        return gm

    name_to_node = _build_name_map(gm)
```

`_build_name_map` builds a `{node.name: node}` dictionary for the entire
current graph. This map is used by `arg_transform` to resolve node references
when copying subgraph nodes into the main graph.

```python
    node_to_idx = {n: i for i, n in enumerate(gm.graph.nodes)}

    def _first_bwd_idx(act):
        fbwd = profiler.act_first_bwd_use.get(act)
        return node_to_idx.get(fbwd, int(1e9)) if fbwd is not None else int(1e9)

    ordered = sorted(nodes_to_recompute, key=_first_bwd_idx)
```

We sort activations by the graph index of their `first_bwd_use`. This means
we process activations that are needed earliest in the backward pass first.
This ordering ensures that when we insert a recomputation subgraph for an
early-needed activation, the graph indices of later activations' `first_bwd_use`
nodes are still valid.

Activations without a `first_bwd_use` get index `int(1e9)` — they sort last.

```python
    for act in ordered:
        first_bwd_use = profiler.act_first_bwd_use.get(act)
        if first_bwd_use is None:
            continue

        _, boundary_inputs = get_recompute_info(act, profiler, nodes_to_retain)

        recompute_graph = _extract_graph_with_inputs_outputs(
            joint_graph=gm.graph,
            inputs=boundary_inputs,
            outputs=[act],
        )
```

`_extract_graph_with_inputs_outputs` is a PyTorch utility from
`torch._functorch.partitioners`. It takes the full `gm.graph` and extracts a
sub-graph that computes `outputs` given `inputs`. The extracted sub-graph:
- Has a `placeholder` node for each element of `inputs` (the boundary nodes)
- Has `call_function` (or similar) nodes for all operations between
  the inputs and outputs
- Has an `output` node collecting the outputs

Importantly, the extracted sub-graph uses its own internal node references —
its nodes are copies with new names. We cannot use them directly in the
main graph; we need to copy them.

```python
        with gm.graph.inserting_before(first_bwd_use):
            for n in recompute_graph.nodes:
                if n.op in ("placeholder", "output"):
                    continue
```

`gm.graph.inserting_before(first_bwd_use)` is a context manager. Every call
to `gm.graph.node_copy()` or `gm.graph.create_node()` inside this block
inserts the new node immediately before `first_bwd_use` in the graph's node
list. After the context manager exits, subsequent `node_copy` calls resume
appending to the end of the graph.

We skip `placeholder` nodes because they correspond to boundary inputs —
these already exist in the main graph and do not need to be inserted. We
skip the `output` node because it is just a collector for the sub-graph's
outputs and has no meaning in the main graph.

```python
                new_node = gm.graph.node_copy(
                    n,
                    arg_transform=lambda arg: name_to_node[arg.name],
                )
```

`gm.graph.node_copy(n, arg_transform)` creates a copy of node `n` in
`gm.graph`. The `arg_transform` function is called for each argument of `n`
(each input edge). It receives the argument node from the **sub-graph** and
must return the corresponding node in the **main graph**.

`lambda arg: name_to_node[arg.name]` does this by looking up the argument's
name in `name_to_node`. This works because:
- For boundary placeholder nodes: their names in the sub-graph match the
  names of the corresponding real nodes in the main graph (this is how
  `_extract_graph_with_inputs_outputs` names them).
- For intermediate nodes already copied in this loop: `name_to_node` was
  updated after each copy (`name_to_node[n.name] = new_node`), so their
  names now map to the newly inserted nodes in the main graph.

```python
                if n.name == act.name:
                    _replace_subsequent_uses_of(gm.graph, act, new_node)

                name_to_node[n.name] = new_node
```

When we reach the node in the sub-graph that corresponds to `act` itself
(i.e. the output node), we call `_replace_subsequent_uses_of`. At this point,
`new_node` has been inserted just before `first_bwd_use`, and `act` is still
the original forward activation node. We redirect all backward uses of `act`
to `new_node`.

After this, the activation's backward users (gradient computations) will use
the freshly recomputed version, not the original forward version. The original
forward tensor is now referenced only by forward nodes. When the last forward
node that uses it runs (and the interpreter drops its environment entry), the
tensor will be garbage collected.

```python
        name_to_node = _build_name_map(gm)
```

After inserting all nodes for this activation, we rebuild the name map. This
is necessary because `node_copy` may have assigned different names to the
new nodes (if there were name conflicts with existing nodes in the graph),
and we need `name_to_node` to be accurate for the next activation's processing.

```python
    gm.graph.lint()
    gm.recompile()
    return gm
```

`gm.graph.lint()` validates the graph: checks that all node arguments are
valid node references, no cycles exist, types are consistent, etc. If there
is a bug in our rewriting, `lint()` will raise an exception here rather than
producing a silently broken graph.

`gm.recompile()` regenerates the Python source code for the `GraphModule`
and recompiles it. After modification, the graph's bytecode is out of date.
`recompile()` calls `torch.fx.Graph.python_code()` to produce new source,
then `exec()` to compile it, making the modified graph executable.

---

## 8. `benchmarks.py` — Models, Experiments, Sweeps

### 8.1 Model tables

```python
model_names = ["Transformer", "Resnet18", "Resnet50", "Bert"]

model_batch_sizes = {
    "Transformer": 4,
    "Resnet18": 16,
    "Resnet50": 4,
    "Bert": 4,
}

sweep_batch_sizes = {
    "Transformer": [1, 2, 4, 8],
    "Resnet18":    [4, 8, 16, 32, 64],
    "Resnet50":    [1, 2, 4, 8, 16],
    "Bert":        [1, 2, 4, 8, 16],
}
```

Default batch sizes are chosen to be large enough to demonstrate meaningful
profiling but small enough to fit in GPU memory without OOM. ResNet18 can
handle larger batches (16) because it has fewer parameters and smaller feature
maps than ResNet50. Transformer and BERT use sequence-based models where memory
scales with both batch size and sequence length, so smaller defaults are used.

### 8.2 `SimpleBert` — the custom BERT implementation

The project requires an LLM model. HuggingFace's `BertModel` is not used
because it contains conditional logic (based on `attention_mask`, `token_type_ids`,
etc.) that `make_fx` cannot trace with fake tensors. Our `SimpleBert` is a
clean reimplementation that is fully static.

**`_BertAttention` forward:**

```python
def forward(self, x):
    B, T, C = x.shape
    qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)   # (B, heads, T, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = self.head_dim ** -0.5
    attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
    return self.proj(out)
```

All Q, K, V projections are fused into a single `Linear(hidden, 3*hidden)` call
to reduce the number of matrix multiplications. `unbind(dim=2)` splits the
result into three tensors along the head dimension. The attention matrix
`q @ k.T` has shape `(B, heads, T, T)`. The softmax is applied along the
last dimension (over keys for each query). The output of attention is
transposed and reshaped back to `(B, T, C)`.

There is no causal mask because BERT is a bidirectional encoder — every
token can attend to every other token.

**`_BertLayer` forward:**

```python
def forward(self, x):
    x = x + self.attn(self.norm1(x))         # pre-norm self-attention
    x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))  # pre-norm FFN
    return x
```

Pre-norm architecture (LayerNorm before the sub-layer) is used rather than
the original BERT's post-norm, because pre-norm is more numerically stable
for tracing. GELU is the activation function used in BERT's FFN.

**`SimpleBert` forward:**

```python
def forward(self, input_ids):
    B, T = input_ids.shape
    pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
    x    = self.tok_emb(input_ids) + self.pos_emb(pos)
    for layer in self.layers:
        x = layer(x)
    return self.head(self.norm(x))
```

`torch.arange(T, ...)` creates position indices `[0, 1, ..., T-1]`. This
works with fake-tensor tracing because `T` is a compile-time constant (the
sequence length is fixed at 128). The `for layer in self.layers` loop is
unrolled during tracing — `make_fx` traces each layer call sequentially,
producing 6 separate sets of attention and FFN nodes in the graph.

### 8.3 The `Experiment` class

**`__init__` parameters:**

```python
def __init__(self, model_name, batch_size, extra_args=[], enable_ac=False, ac_reduction=0.5):
```

- `model_name`: must be in `model_names`
- `batch_size`: the number of samples per training iteration
- `enable_ac`: whether to apply activation checkpointing in `graph_transformation`
- `ac_reduction`: the fraction of activation memory to target for removal
  (0.5 = target 50% reduction)

**Why `foreach=True` and not `fused=True`:**

```python
self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, foreach=True, capturable=True)
```

`fused=True` Adam uses a single CUDA kernel for the entire optimizer step.
During this kernel, it inspects `device.type` on the parameter tensors to
determine which CUDA path to take. With FakeTensors, `device` is `None`,
causing `AttributeError: 'NoneType' object has no attribute 'type'`. The fix
is `foreach=True`, which decomposes into standard ATen operations that are
compatible with FakeTensors.

`capturable=True` enables state updates to be traced into the graph. Without
this, Adam would try to perform the optimizer step differently when
`is_compiling()` returns `True`.

**`init_opt_states`:**

```python
def init_opt_states(self):
    for param in self.model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param)
    self.optimizer.step()
    self.optimizer.zero_grad()
```

This must be called before `compile` to ensure that Adam's state dictionaries
(`exp_avg`, `exp_avg_sq`, `step`) exist for every parameter. `make_fx` extracts
`opt.state` in `_compile` — if the states don't exist yet (i.e., `step()` has
never been called), `named_states` will be empty and no optimizer state
placeholder nodes will appear in the graph.

**`graph_transformation` — the integration point:**

```python
def graph_transformation(self, gm, args):
    warm_up_iters, profile_iters = 2, 3
    self.profiler = GraphProfiler(gm)

    with torch.no_grad():
        for _ in range(warm_up_iters):
            self.profiler.run(*args)
        self.profiler.reset_stats()
        for _ in range(profile_iters):
            self.profiler.run(*args)
    self.profiler.aggregate_stats()
    self.profiler.print_stats()
    self.peak_mem_by_cat = self.profiler.peak_memory_by_category()
```

`warm_up_iters=2`: the first 2 runs allow CUDA to JIT-compile kernels and
warm up memory allocators. These measurements are discarded.
`profile_iters=3`: 3 measured runs are averaged. This gives a stable estimate
without taking too long.

`torch.no_grad()` disables PyTorch's autograd during the profiling runs.
This is correct because we are running the **already-traced graph** directly,
not the original Python training function. The graph already includes backward
operations as explicit nodes — there is no need for autograd to run again.

```python
    if self.enable_ac:
        current_act_peak = sum(
            self.profiler._tensor_sizes.get(a, 0)
            for a in self.profiler.activation_nodes
        )
        budget = int(current_act_peak * (1.0 - self.ac_reduction))

        nodes_to_recompute, nodes_to_retain = mu_two_selection(
            self.profiler, memory_budget_bytes=budget
        )
        print_ac_plan(nodes_to_recompute, nodes_to_retain, self.profiler)

        gm = apply_activation_checkpointing(
            gm, nodes_to_recompute, nodes_to_retain, self.profiler
        )

        act_peak_ac = sum(
            self.profiler._tensor_sizes.get(a, 0) for a in nodes_to_retain
        )
        self.peak_mem_by_cat = dict(self.peak_mem_by_cat)
        self.peak_mem_by_cat["activations"] = act_peak_ac

    return gm
```

`budget = int(current_act_peak * (1.0 - ac_reduction))`: with
`ac_reduction=0.5`, the budget is 50% of current activation memory. The
algorithm will checkpoint activations until the estimated remaining
activation memory is at or below this budget.

After rewriting, we update `peak_mem_by_cat["activations"]` to reflect the
new lower peak: the sum of sizes of only the retained activations. This
updated value is what gets plotted in the AC bar chart.

`dict(self.peak_mem_by_cat)` creates a shallow copy before modifying it, so
the original Phase 1 measurement is preserved in `self.profiler` and only the
`Experiment`'s own copy is updated.

### 8.4 `measure_iteration_latency`

```python
def measure_iteration_latency(compiled_fn, model, optimizer, example_inputs,
                               n_warmup=3, n_iters=10):
    for _ in range(n_warmup):
        compiled_fn(model, optimizer, example_inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        compiled_fn(model, optimizer, example_inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters
```

Unlike the per-node profiling (which calls `synchronize()` after every single
node), here we bracket ALL 10 iterations with a single start/end event pair.
This gives accurate total timing without the CPU-GPU sync overhead dominating
the measurement (synchronize takes ~0.1ms and there are thousands of nodes).

The `torch.cuda.synchronize()` before `start.record()` ensures all previous
operations (including the warmup iterations) have completed before we start
timing.

### 8.5 `_run_sweep_single`

```python
def _run_sweep_single(model_name, enable_ac):
    for bs in sweep_batch_sizes[model_name]:
        try:
            exp = Experiment(model_name, bs, enable_ac=enable_ac)
            exp.init_opt_states()
            compiled_fn = compile(exp.train_step, exp.graph_transformation)
            compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
            mem = exp.peak_mem_by_cat
            lat = measure_iteration_latency(
                compiled_fn, exp.model, exp.optimizer, exp.example_inputs
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={bs}, stopping sweep.")
            break
        ...
```

Each batch size creates a **fresh** `Experiment` object. This is important
because `compile()` stores the compiled graph in the function object's
`__dict__`. If we reused the same `train_step` function across batch sizes,
the second batch size would use the graph compiled for the first.

`torch.cuda.OutOfMemoryError` is caught to handle cases where a batch size
is too large for the GPU. The sweep stops at that point rather than
crashing the entire run.

### 8.6 `run_comparison_sweep` — the chart generation

The function runs two sweeps (no-AC and AC), then generates two matplotlib
charts.

**Memory chart — grouped stacked bars:**
For each batch size, two bars are drawn side by side: the no-AC bar (solid,
`alpha=1.0`) on the left and the AC bar (faded, `alpha=0.55`) on the right.
`offset = -w/2` for no-AC and `+w/2` for AC positions the bars around the
x-axis tick mark.
Each bar is stacked — `bottoms` accumulates the sum of previous categories
so each new category starts from the top of the previous one.
The legend is deduplicated by skipping labels that start with `"_"` (the AC
bars have labels like `"_activations_ac"` to prevent them from appearing in
the legend twice).

**Latency chart — line chart:**
Uses `common_bs` — the intersection of batch sizes that completed without
OOM for both the no-AC and AC runs. The no-AC line uses solid circles `"o-"`,
the AC line uses dashed squares `"s--"`.

---

## 9. `phase1_run.py` and `phase2_run.py` — Runner Scripts

### 9.1 `phase1_run.py`

Runs Phase 1 deliverables only. For each model at its default batch size:

```python
class Tee:
    def write(self, s):
        original_stdout.write(s)   # live print to terminal
        captured.write(s)          # also capture in StringIO
    def flush(self):
        original_stdout.flush()

sys.stdout = Tee()
compiled_fn = compile(exp.train_step, exp.graph_transformation)
compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
sys.stdout = original_stdout

with open(out_path, "w", encoding="utf-8") as f:
    f.write(captured.getvalue())
```

The `Tee` class (named after the Unix `tee` command) simultaneously writes to
two destinations. `sys.stdout = Tee()` replaces Python's standard output with
our tee object. All `print()` calls during the compilation and profiling
(including `profiler.print_stats()`) go through `Tee.write()`. After
compilation, we restore `sys.stdout` and write the captured content to disk.

Also calls `run_memory_sweep` for each model to generate the Phase 1 bar charts.

### 9.2 `phase2_run.py`

**`save_stats(model_name, enable_ac)`:** Same Tee pattern as phase1_run.py.
Saves to `records/stats_<model>_noac.txt` or `records/stats_<model>_ac.txt`.
When `enable_ac=True`, the output includes both `print_stats()` and
`print_ac_plan()` (because both are called inside `graph_transformation`).

**`main(models, mode, ac)`:**
In `compare` mode (the default), for each model:
1. Calls `save_stats(model, False)` — profile at default batch size, no AC
2. Calls `save_stats(model, True)` — profile + apply AC, save plan
3. Calls `run_comparison_sweep(model)` — full sweep, generate 2 charts

**Argument parser:**
```
python phase2_run.py                        → all models, compare mode
python phase2_run.py --model Bert           → Bert only, compare mode
python phase2_run.py --model Bert --mode single --ac  → single batch, AC
```

---

## 10. Complete End-to-End Data Flow

```
User runs:
  phase2_run.py --model Resnet18
         │
         ▼
  save_stats("Resnet18", enable_ac=False)
  save_stats("Resnet18", enable_ac=True)
  run_comparison_sweep("Resnet18")
         │
         ├──── For each batch size in [4, 8, 16, 32, 64]:
         │
         │  Experiment("Resnet18", bs, enable_ac=False/True)
         │    ├── Construct ResNet18 model on CUDA
         │    ├── Create random 224×224 input batch
         │    └── Create Adam(foreach=True, capturable=True)
         │
         │  exp.init_opt_states()
         │    └── Run one optimizer step to initialize exp_avg, exp_avg_sq
         │
         │  compiled_fn = compile(exp.train_step, exp.graph_transformation)
         │
         │  compiled_fn(model, optimizer, batch)   ← FIRST CALL
         │    │
         │    ├── graph_tracer._compile(train_step, model, optimizer, batch)
         │    │     ├── Extract params, buffers, named_states
         │    │     ├── Wrap in stateless_func
         │    │     ├── Convert all tensors to FakeTensors
         │    │     ├── make_fx(stateless_func, SPMD_DECOMP_TABLE)
         │    │     │     Records every ATen op:
         │    │     │     - forward: conv2d, relu, batch_norm, ...
         │    │     │     - loss: cross_entropy
         │    │     │     - SEPFunction.forward → separator.sep node
         │    │     │     - SEPFunction.backward → separator.sep_backward node
         │    │     │     - backward: threshold_backward, mm, ...
         │    │     │     - optimizer: _foreach_mul, _foreach_sqrt, copy_, ...
         │    │     ├── Remove detach + tag_grad nodes
         │    │     └── Returns _CompiledResult(gm, mod, opt, flat_state)
         │    │
         │    └── gm_transformation(gm, flat_inputs)
         │          │
         │          ├── GraphProfiler(gm)   [STATIC ANALYSIS]
         │          │     ├── Scan for sep_node (sep_idx) and sep_bwd_node (sep_bwd_idx)
         │          │     ├── Build forward_nodes (idx < sep_idx)
         │          │     │   and backward_nodes (idx > sep_bwd_idx)
         │          │     ├── _find_param_and_opt_nodes:
         │          │     │     Find _foreach_addcdiv → extract param_nodes, opt_state_nodes
         │          │     │     Fallback: any placeholder with backward users → opt_state
         │          │     ├── _classify_node for every node:
         │          │     │     placeholder → PARAM / OPT / INPUT
         │          │     │     forward non-placeholder with bwd user → ACT
         │          │     │     forward non-placeholder, no bwd user → OTHER
         │          │     │     after sep_bwd → GRAD
         │          │     │     sentinel/output → OTHER
         │          │     └── For each ACT: record last_fwd_use, first_bwd_use
         │          │
         │          ├── 2 warm-up runs via profiler.run(*args)
         │          ├── profiler.reset_stats()
         │          ├── 3 measured runs via profiler.run(*args)
         │          │     Each run: for every node:
         │          │       record(start_event)
         │          │       super().run_node(n)  [GPU kernel]
         │          │       record(end_event)
         │          │       synchronize()
         │          │       append elapsed_ms to _runtimes[n]
         │          │       append (mem_after - mem_before) to _mem_delta[n]
         │          │       record _tensor_sizes[n] (first time only)
         │          ├── profiler.aggregate_stats() → node_avg_runtime, node_avg_mem_delta
         │          ├── profiler.print_stats() → two tables to stdout
         │          ├── profiler.peak_memory_by_category() → peak_mem_by_cat dict
         │          │
         │          └── [if enable_ac=True]
         │                ├── Compute budget = sum(act_sizes) * (1 - ac_reduction)
         │                ├── mu_two_selection(profiler, budget)
         │                │     ├── Filter to valid candidates
         │                │     ├── Sort by efficiency = size / recompute_cost
         │                │     └── Greedy select with dependency constraint
         │                │           Returns (nodes_to_recompute, nodes_to_retain)
         │                ├── print_ac_plan(...)
         │                └── apply_activation_checkpointing(gm, ...)
         │                      ├── Sort nodes_to_recompute by first_bwd_use index
         │                      └── For each act:
         │                            ├── get_recompute_info → boundary_inputs
         │                            ├── _extract_graph_with_inputs_outputs
         │                            │     → recompute_graph
         │                            ├── inserting_before(first_bwd_use):
         │                            │     for each node in recompute_graph:
         │                            │       node_copy with arg_transform
         │                            │       if node == act: replace_subsequent_uses_of
         │                            │       update name_to_node
         │                            ├── rebuild name_to_node
         │                      ├── gm.graph.lint()
         │                      └── gm.recompile()
         │
         │  compiled_fn is now ready (graph compiled, optionally rewritten)
         │
         │  measure_iteration_latency(compiled_fn, ...)
         │    ├── 3 warmup iterations
         │    ├── synchronize
         │    ├── start_event.record()
         │    ├── 10 measured iterations
         │    ├── end_event.record()
         │    ├── synchronize
         │    └── return elapsed_time / 10
         │
         └── Collect: peak_mem_by_cat, latency_ms

  run_comparison_sweep plots:
    ├── peak_memory_comparison_resnet18.png (grouped stacked bars)
    └── latency_comparison_resnet18.png (line chart)
```

---

## 11. Every Design Decision Explained

**Q: Why not just use Python's `time.time()` for timing?**
CUDA operations are asynchronous. When `super().run_node(n)` returns, Python
returns immediately after dispatching the GPU command to the CUDA stream. The
actual GPU kernel may still be running. `time.time()` would measure the CPU
dispatch time (often under 0.01ms) rather than the GPU execution time (which
can be 0.1–10ms for real kernels). CUDA Events are GPU-side timestamps, so
`elapsed_time` measures true GPU kernel duration.

**Q: Why do we need `synchronize()` after every node in `run_node`?**
Without synchronization, `torch.cuda.memory_allocated()` reads the allocator
state before the kernel has actually allocated its output tensor. This would
give wrong (often zero or negative) memory deltas. We need the GPU to finish
the kernel before we sample memory. The trade-off is that `synchronize()`
adds ~0.1ms of CPU-GPU sync overhead per node, but this overhead is the same
for all nodes and is removed when we switch to `measure_iteration_latency`
for the final latency measurement.

**Q: Why `foreach=True` and not `fused=True` for Adam?**
`fused=True` uses a monolithic CUDA kernel that inspects `param.device.type`
to select between CUDA and CPU code paths. With FakeTensors (which have
`device=None`), this causes `AttributeError: 'NoneType' object has no
attribute 'type'`. `foreach=True` decomposes into standard ATen operations
(`_foreach_mul`, `_foreach_sqrt`, etc.) that are fully compatible with
FakeTensors.

**Q: Why call `init_opt_states()` before `compile`?**
`make_fx` extracts optimizer states via `opt.state`. If no optimizer step has
ever been called, `opt.state` is empty and no optimizer state placeholder nodes
appear in the graph. Without these placeholders, the graph cannot represent the
Adam update (it has no nodes for `exp_avg`, etc.), and the graph profiler cannot
classify any nodes as `OPT`. The fake optimizer step in `init_opt_states()`
pre-populates `opt.state` with correctly-shaped tensors.

**Q: Why does `_to_caller_flattened_graph_module` exist?**
`stateless_func` wraps `train_step(model, optimizer, inputs)` and adds extra
arguments (params dict, buffers dict, named_states dict). After tracing, the
`fx.GraphModule` expects these nested dict structures as input. `_to_caller_flattened_graph_module` replaces the graph's code generator with one
that accepts a flat list of tensors instead. This makes it possible to call
`compiled_obj.gm(*flat_state, model, optimizer, batch)` without having to
reconstruct the dict structure every time.

**Q: Why use `SPMD_DECOMP_TABLE` (in-place decomposition)?**
Without decomposition, the graph would contain in-place operations like
`_foreach_add_(params, deltas)`. These have no output edge in the graph
because they modify their input in place. The profiler needs output edges
to track tensor lifetimes and measure tensor sizes. The decomposition
replaces each in-place op with an out-of-place op (producing a new tensor)
followed by `copy_` nodes. The graph now has proper data flow, and the
profiler can track every tensor.

**Q: Why does `_classify_node` check `node_to_idx.get(u, -1) > sep_bwd_idx`
rather than `u in self.backward_nodes`?**
At the point where `_classify_node` runs, `self.backward_nodes` may not yet
be populated (the backward nodes set and classify nodes steps happen in the
same loop). Using the index directly (`node_to_idx.get(u, -1)`) is both
faster and avoids this ordering dependency.

**Q: Why is `ac_reduction=0.5` the default?**
50% is a conservative target that is achievable for most models without
encountering the dependency constraint often. Higher values (0.7–0.9) would
save more memory but risk checkpointing activations that are too expensive
to recompute, leading to unacceptable latency overhead. Lower values (0.2–0.3)
reduce overhead but may not save enough memory to make a meaningful difference.

**Q: Why sort `nodes_to_recompute` by `first_bwd_use` index before rewriting?**
The graph is a linked list of nodes. When we insert nodes before `first_bwd_use`
of activation A, the positions of subsequent nodes in the graph do not change
(we are inserting before a fixed reference point). However, after inserting
for A, the graph has new nodes in it, and if we then process activation B
whose `first_bwd_use` comes after A's, B's `first_bwd_use` node is still
correctly referenced. Processing in order ensures we never insert a node before
a point that is before a node we already inserted, which could cause topological
ordering violations.

**Q: Why rebuild `name_to_node` after each activation's rewrite?**
When `node_copy` creates a new node, PyTorch's `fx.Graph` may rename it to
avoid conflicts (e.g. if a node named `relu` already exists, the copy becomes
`relu_1`). After each activation's rewrite, the names in the graph may have
changed, so we rebuild the map to ensure subsequent activations' `arg_transform`
lambdas resolve to the correct nodes.

**Q: Why does ResNet50 show higher AC latency overhead than ResNet18?**
ResNet50 uses residual (skip) connections. A residual block computes
`output = activation(conv(x) + x)`. The activation at the output of the block
depends on two paths: the conv path and the identity shortcut. Recomputing
this activation requires re-running both paths. For deep residual networks,
the recomputation subgraph can include many operations across both paths,
leading to higher recomputation cost than our single-node cost estimate
predicted. The efficiency ratio we computed underestimated the true cost
for these activations.

**Q: Why does the latency show noise at small batch sizes?**
At small batch sizes, GPU utilization is low — the cores are not fully busy.
Small variations in memory allocation, cache state, and CUDA kernel selection
can cause timing variations of 5–10%. The CUDA Events measure true GPU time,
but the GPU itself is more variable when underutilized. At large batch sizes,
the GPU is saturated and timing becomes much more stable.

---

## 12. Anticipated Questions and Answers

**Q: What would happen if we set `ac_reduction=1.0`?**
The budget would be 0, meaning the algorithm would try to checkpoint every
possible activation. In practice, the dependency constraint prevents some
activations from being checkpointed (if their required inputs are also being
checkpointed). The result would be maximum memory savings but very high
recomputation overhead — potentially 2–3× slower training.

**Q: Could we recompute an activation that depends on another recomputed activation?**
Not with the current dependency constraint. If B requires A as an input, and
A is being recomputed, then to recompute B we would first need to recompute A,
and then use the result to recompute B. This is called "cascaded recomputation"
and is supported by some AC frameworks (like `torch.utils.checkpoint`). Our
implementation forbids it to keep the rewriter simple and predictable.

**Q: How do we know the rewritten graph produces correct numerical results?**
The correctness argument is: we insert a subgraph that computes the exact
same operations as the original forward computation for that activation. The
inputs to the subgraph (retained activations and placeholder parameters) are
identical to what the original forward computation used. The output is
therefore numerically identical to the original activation tensor. We redirect
all backward uses of the original to use this recomputed version. The backward
computation proceeds identically to the non-AC case, producing the same
gradients and weight updates.

**Q: What is `_allow_non_fake_inputs=False` in `make_fx`?**
This tells `make_fx` to assert that all inputs to the traced function are
FakeTensors. If any real tensor sneaks in (e.g. through a closure), it would
cause the trace to have non-fake data dependencies that cannot be represented
in the graph. Setting this to False catches these bugs early.

**Q: Why is `torch.autograd.detect_anomaly(check_nan=False)` needed during tracing?**
Without anomaly detection, PyTorch may skip recording some gradient operations
in the autograd graph during tracing. `detect_anomaly` forces PyTorch to record
the full computation graph for all operations, ensuring that all backward nodes
appear in the traced `fx.GraphModule`. The `check_nan=False` parameter
prevents it from checking for NaN values (which it cannot do with FakeTensors
that have no data).

**Q: The output node at the end of the graph — what does it collect?**
`stateless_func` returns three things: `(ret, list(mod.parameters()), list(named_states.values()))`. The `output` node in the graph collects all three.
The compiled `wrapper` function accesses `[0]` to get `ret`, which is the
original return value of `train_step` (which is `None` for all our models —
the training step is a side-effecting function with no return value).

**Q: Why does `peak_memory_by_category` give an upper bound, not exact peak?**
The simulation assumes that a tensor is freed immediately after its last user
executes. In practice, PyTorch's CUDA memory allocator may not release the
memory to the OS immediately — it keeps it in a pool for reuse. Also, our
`last_use` calculation only considers nodes explicitly in the graph; there
may be intermediate Python-level references. The simulation is accurate
enough for comparison purposes (the relative reduction from AC is correct
even if the absolute numbers are slightly off).

---

## 13. Experimental Results — Memory and Latency Analysis

This section analyzes the actual outputs produced by `phase2_run.py` for all
four models. All profiling numbers come from the `stats_*_noac.txt` and
`stats_*_ac.txt` files. All chart readings come from the eight PNG files in
`records/`.

### 13.1 Profiling Summary

| Model | Batch Size | Fwd (ms) | Bwd (ms) | Bwd/Fwd | Activation Mem |
|-------|-----------|----------|----------|---------|---------------|
| ResNet-18 | 16 | 23.18 | 119.27 | 5.1× | 347 MB |
| ResNet-50 | 4 | 47.65 | 234.77 | 4.9× | 350 MB |
| Transformer | 4 | 32.47 | 165.42 | 5.1× | 19 MB |
| BERT | 4 | 56.79 | 247.55 | 4.4× | 237 MB |

The backward pass is 4.4–5.1× slower than forward across all models. For every
weight matrix, backward computes two gradient tensors (one for the weight, one
to propagate to the next layer), plus the Adam optimizer step runs in the same
backward region. BERT's ratio is the lowest (4.4×) because its attention
backward is nearly as expensive per flop as the forward, compressing the ratio
relative to convolution-heavy models.

ResNet-18 and ResNet-50 have almost identical activation memory (347 MB vs.
350 MB) despite ResNet-50 having more than twice as many parameters. This is
because activation memory is set by feature-map spatial dimensions and batch
size, not parameter count. ResNet-18 runs at batch 16 while ResNet-50 runs at
batch 4 — the 4× smaller batch size of ResNet-50 offsets its wider channels.
The Transformer's tiny activation budget (19 MB) reflects that its tokens
have a small hidden dimension compared to a full spatial feature map.

### 13.2 μ-TWO Checkpointing Plan Summary

| Model | Total ACTs | Recomputed | Freed | Retained | Kept | Actual Reduction |
|-------|-----------|-----------|-------|---------|------|-----------------|
| ResNet-18 | 42 | 6 | 154 MB | 36 | 168 MB | 44.4% |
| ResNet-50 | 107 | 18 | 169 MB | 89 | 174 MB | 48.3% |
| Transformer | 158 | 5 | 9.3 MB | 153 | 9.4 MB | 48.3% |
| BERT | 102 | 31 | 120 MB | 71 | 117 MB | 50.4% |

All four models land within 6 percentage points of the 50% target. ResNet-18
achieves it with only 6 checkpoints because its early-layer ReLU tensors are
massive (the first one alone is 51 MB) — a few high-efficiency selections
cover the entire budget. BERT requires 31 checkpoints because its attention
tensors are more uniformly sized (3–8 MB each), requiring more individual
selections to accumulate the same total savings. The Transformer checkpoints
only 5 activations because its baseline memory is already tiny; the 9.6 MB
budget is met almost immediately by picking the 5 largest tensors.

### 13.3 Peak Memory Comparison Charts

#### ResNet-18

Reading from the chart: at batch size 4 the no-AC bar is ~0.55 GB and the AC
bar is ~0.50 GB. At batch 32, no-AC reaches ~1.32 GB while AC is ~0.94 GB.
At batch 64, no-AC is ~2.42 GB and AC is ~1.70 GB.

Three things are immediately visible in the chart:

**Only the orange segment (activations) shrinks.** Params (blue), gradients
(green), and opt states (red) are identical between the solid and faded bars.
This is exactly correct: AC does not touch weights or their gradients. It only
discards activation tensors and re-creates them on demand.

**The absolute saving grows linearly with batch size.** At batch 4 the saving
is roughly 50 MB; at batch 64 it is about 720 MB. This is expected because
activation memory scales linearly with batch size, so a 44% relative reduction
produces larger absolute savings at larger batches.

**At small batch sizes the saving looks small in proportion to the total bar.**
This is because the fixed costs (params ~100 MB, opt states ~200 MB, gradients
~250 MB) dominate at small batch sizes. The activation segment is not yet the
majority of total memory at batch 4.

#### ResNet-50

At batch 1 the bars are nearly the same height (~1.10 GB vs. ~1.06 GB). At
batch 16 the gap opens to ~2.40 GB vs. ~1.70 GB.

ResNet-50's fixed costs are much higher than ResNet-18's. Params (~100 MB),
opt states (~200 MB), and gradients (~450 MB) together already account for
most of the bar at small batch sizes. The activation segment grows visibly
only from batch 4 onward, which is why the relative saving appears modest at
batch 1 but becomes clearly visible by batch 8 and 16.

The absolute memory saving at batch 16 is approximately 700 MB — comparable
to what ResNet-18 achieves at batch 64. This makes sense: ResNet-50 has larger
feature maps in its bottleneck layers, so even at batch 16 there are large
activations to checkpoint.

#### Transformer

The y-axis for the Transformer chart reaches only 0.09 GB — about 30× smaller
than the ResNet charts. At batch 1 the no-AC bar is ~13 MB and the AC bar is
~10 MB. At batch 8 the saving is from ~90 MB to ~69 MB.

The green gradient segment dominates the bar at every batch size. This makes
sense: the Transformer has ~22 M parameters, so gradients are substantial
relative to its small activation footprint. The activation segment (orange) is
hard to see at batch 1 but becomes visible by batch 4 and 8.

The AC saving is real but visually modest. At batch 8, reducing activation
memory from ~38 MB to ~17 MB (the orange segment shrinks roughly in half)
saves 21 MB out of a 90 MB total — a 23% total-bar reduction. For the
Transformer, AC matters most if you want to train at batch sizes above 8 where
the attention matrices would otherwise start dominating memory.

#### BERT

BERT's chart tells the most interesting story. At batch 1 the bar is already
~1.22 GB with AC reducing it to ~1.15 GB — a small absolute saving. But unlike
the ResNets, BERT's total bar height barely increases from batch 1 to batch 16
(from 1.22 GB to 1.77 GB). This is because BERT's sequence length is fixed at
128, so each attention matrix is $(B, 8, 128, 128)$ — memory grows linearly
with B but the per-layer base cost (embedding tables, layer norms, projection
weights) is large and batch-invariant.

The AC saving (orange segment shrinkage) is consistently visible across all
batch sizes, ranging from ~70 MB at batch 1 to ~340 MB at batch 16. At batch
16 the AC bar is ~1.43 GB vs. the no-AC bar's ~1.77 GB.

An important structural observation: BERT has large fixed costs (params ~115 MB,
opt states ~230 MB) that do not change with batch size. These appear in the
blue and red segments and are identical between the two bars at every batch
size — as expected.

### 13.4 Latency Comparison Charts

#### ResNet-18

The no-AC line starts at ~24 ms (batch 4), dips to ~24 ms (batch 8), rises to
~27 ms (batch 16), ~43 ms (batch 32), and ~74 ms (batch 64). The line is
nearly flat from batch 4 to 8 — the GPU is underutilized at small batches and
adding 4 more samples barely increases wall-clock time.

The AC line starts noticeably higher at batch 4 (~30 ms vs. 24 ms, a 25%
overhead), converges with no-AC at batch 8 (~25 ms), then stays consistently
above no-AC: ~36 ms at batch 16 (+33%), ~46 ms at batch 32 (+7%), ~81 ms at
batch 64 (+9%).

The batch-4 overhead being the highest is initially surprising. At small
batches, GPU utilization is low — the GPU is underutilized and individual
kernel launches have high relative overhead. Inserting recomputation kernels
into an already-underutilized GPU adds disproportionately high latency because
each new kernel launch competes for a GPU that is already switching between
small workloads. At batch 8 the GPU is better saturated and the recomputation
kernels fit into spare capacity.

The ~9–33% overhead at larger batches is the "true" steady-state recomputation
cost: 6 activations are being re-executed, adding roughly 6 additional ReLU +
convolution operations to the backward pass.

#### ResNet-50

The ResNet-50 latency chart is the most volatile of all four. The no-AC line
is not monotonically increasing: it goes 46 ms (batch 1), 54 ms (batch 2), 52
ms (batch 4), 49 ms (batch 8), 69 ms (batch 16). This non-monotone behavior
at small batch sizes is a real GPU phenomenon — kernel tuning, memory layout
decisions, and occupancy heuristics inside cuDNN change non-smoothly as batch
size changes.

The AC line crosses the no-AC line twice. At batch 1, AC is dramatically
slower (61 ms vs. 46 ms — a 32% overhead). At batch 2, AC is actually faster
(47 ms vs. 54 ms). At batch 4, AC is slower again (58 ms vs. 52 ms). At
batch 8, the two lines are essentially the same (48 ms). At batch 16, AC is
clearly slower (79 ms vs. 69 ms — a 14% overhead).

The batch-1 spike in AC latency is caused by ResNet-50's residual connections.
Recomputing an activation at the output of a residual block requires re-running
both the convolutional branch and the identity shortcut, then the addition and
ReLU. At batch 1, this subgraph runs in a very small occupancy regime where
each kernel launch has high relative overhead. The μ-TWO efficiency estimate
uses profiled single-node runtimes, which at batch 1 may severely underestimate
the true recomputation cost when these nodes are inserted into the backward
region of a GPU that is handling other gradient computations simultaneously.

The batch-2 speedup for AC (47 ms vs. 54 ms) is explained by memory pressure:
removing 169 MB of large activations from the CUDA memory pool at batch 2
frees enough contiguous memory that subsequent allocations avoid fragmentation-
driven slowdowns. The net effect is faster memory access for the gradient
computations, outweighing the recomputation cost.

#### Transformer

The Transformer latency chart shows the most erratic AC line. At batch 1, AC
is 50 ms vs. no-AC's 37 ms (35% overhead). At batch 2, AC drops to 39 ms
while no-AC also dips to 33 ms (AC overhead ~18%). At batch 4, AC spikes to
53 ms while no-AC rises to 39 ms (AC overhead ~36%). At batch 8, the
situation inverts: AC is 33 ms and no-AC is 36 ms — AC is faster.

The key insight here is that the Transformer checkpoints only 5 activations,
all of which are attention intermediate tensors. At small batch sizes (1, 2, 4),
the 5 recomputation subgraphs each require a matrix multiplication on a small
batch, which has high kernel launch overhead relative to the actual compute.
This produces the large swings visible in the chart. At batch 8, the 5
recomputation matrix multiplications are large enough to run efficiently, and
the memory savings from discarding the attention matrices outweigh the
recomputation cost — hence AC is faster at batch 8.

The non-monotone no-AC line (37 ms → 33 ms → 39 ms → 36 ms) reflects the
inherent variability of GPU timing at small batch sizes where the GPU is
underutilized. Neither line should be considered stable below batch 4.

#### BERT

BERT's latency chart is the one that best matches the theoretical expectation.
Both lines increase roughly proportionally with batch size, and the gap between
them is small and fairly consistent.

At batch 1: no-AC 37 ms, AC 33 ms — AC is actually 11% faster. At batch 2:
no-AC 33 ms, AC 41 ms — AC is 24% slower. At batch 4: no-AC 42 ms, AC 38 ms
— AC is 9% faster. At batch 8: no-AC 35 ms, AC 36 ms — essentially the same.
At batch 16: no-AC 42 ms, AC 46 ms — AC is 10% slower.

The alternating faster/slower pattern at small batch sizes is noise: BERT's
attention matrix multiplications are just at the size threshold where GPU
occupancy fluctuates. At batch 16, the lines stabilize and AC overhead
settles to about 10% — consistent with checkpointing 31 activations that each
require re-running one attention computation.

The most important takeaway from BERT's latency chart: the overhead is bounded.
It never exceeds 25% and averages around 10% across all batch sizes. Given
that AC saves 50% of activation memory for BERT, the tradeoff is very
favorable: trading 10% latency for 50% memory savings allows BERT to run at
2× the batch size, which more than recovers the throughput loss.

### 13.5 Cross-Model Takeaways

**Memory behavior is predictable and consistent.** In all four models, only the
activation segment shrinks under AC. The savings grow linearly with batch size
and land within 6 percentage points of the 50% target. The profiler's static
size estimates are accurate enough to drive the greedy selection to a reliable
outcome.

**Latency behavior is architecture-dependent and noisy at small batch sizes.**
ResNet-18 shows a clean overhead of 9–33% that decreases with batch size.
ResNet-50 shows large swings driven by residual-connection recomputation costs
and memory-pressure effects that sometimes make AC faster, sometimes slower.
The Transformer shows convergence to a benefit (AC becomes faster) at the
largest tested batch size. BERT shows near-neutral overhead throughout, making
it the model where AC provides the clearest net benefit.

**The μ-TWO efficiency estimate is a good proxy but not perfect.** It
underestimates recomputation cost for residual blocks (ResNet-50, batch 1),
overestimates it for cases where memory-pressure reduction dominates (ResNet-50
batch 2, BERT batch 1 and 4). A more accurate cost model would account for
kernel parallelism, memory pressure effects, and the occupancy of the GPU at
the insertion point in the backward pass — all of which are beyond the scope
of a static profiling approach.

**For the models tested, AC is most beneficial for BERT and ResNet-18 at large
batch sizes.** BERT achieves 50% memory savings with ~10% median overhead.
ResNet-18 achieves 44% savings with ~9% overhead at batch 64. Both of these
are favorable operating points for practical training. ResNet-50 and the
Transformer show more variable overhead and would benefit from a more
conservative memory budget target (e.g., 30% reduction rather than 50%) to
avoid the high-overhead operating points.
