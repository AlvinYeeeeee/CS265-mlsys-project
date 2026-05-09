import argparse
import os
from typing import Any, Dict, List, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torchvision.models import resnet18, resnet50
from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile
from ac_algorithm import mu_two_selection, print_ac_plan
from graph_rewriter import apply_activation_checkpointing


model_names: List[str] = [
    "Transformer",
    "Resnet18",
    "Resnet50",
    "Bert",
]

model_batch_sizes: Dict[str, int] = {
    "Transformer": 4,
    "Resnet18": 16,
    "Resnet50": 4,
    "Bert": 4,
}

# Batch sizes to sweep for the peak-memory bar graph (deliverable 4b).
sweep_batch_sizes: Dict[str, List[int]] = {
    "Transformer": [1, 2, 4, 8],
    "Resnet18":    [4, 8, 16, 32, 64],
    "Resnet50":    [1, 2, 4, 8, 16],
    "Bert":        [1, 2, 4, 8, 16],
}


# --------------------------------------------------------------------------- #
# SimpleBert: a traceable BERT-like encoder                                   #
# --------------------------------------------------------------------------- #
# Uses the same architecture as BERT-base (encoder-only, bidirectional
# self-attention, LayerNorm, GELU, MLM-style head) but is implemented from
# scratch so make_fx can trace it without HuggingFace dependencies.

class _BertAttention(nn.Module):
    """Multi-head self-attention (no causal mask — bidirectional like BERT)."""
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        assert hidden % heads == 0
        self.heads    = heads
        self.head_dim = hidden // heads
        self.qkv  = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)           # each (B, T, heads, head_dim)
        q = q.transpose(1, 2)                  # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class _BertLayer(nn.Module):
    """Single BERT encoder layer: self-attention + feed-forward + LayerNorm."""
    def __init__(self, hidden: int, heads: int, ffn: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = _BertAttention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff1   = nn.Linear(hidden, ffn)
        self.ff2   = nn.Linear(ffn, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


class SimpleBert(nn.Module):
    """
    BERT-like encoder with an MLM head.

    Hyper-parameters (defaults match a small but non-trivial config):
      vocab_size   = 8192
      hidden       = 512     (BERT-base uses 768)
      num_layers   = 6       (BERT-base uses 12)
      num_heads    = 8       (BERT-base uses 12)
      max_seq_len  = 128
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        hidden: int     = 512,
        num_layers: int = 6,
        num_heads: int  = 8,
        max_seq_len: int = 128,
    ):
        super().__init__()
        ffn = hidden * 4
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_seq_len, hidden)
        self.layers  = nn.ModuleList(
            [_BertLayer(hidden, num_heads, ffn) for _ in range(num_layers)]
        )
        self.norm    = nn.LayerNorm(hidden)
        self.head    = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T   = input_ids.shape
        pos    = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x      = self.tok_emb(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class Experiment:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        extra_args: list = [],
        enable_ac: bool = False,
        ac_reduction: float = 0.5,
    ):
        assert model_name in model_names, f"Model {model_name} not found in model names {model_names}"
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size
        self.enable_ac = enable_ac
        # Fraction of current activation memory to remove (0.5 = reduce by 50%)
        self.ac_reduction = ac_reduction

        if self.model_name == "Transformer":

            vocab_size = 2048
            bsz, seq_len = self.batch_size, 256
            with torch.device(dev):
                model_args = ModelArgs(
                    n_layers=8,
                    n_heads=4,
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    dropout_p=0.1,
                )
                self.model = Transformer(model_args)
            src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            self.example_inputs = (src, tgt)

            def transformer_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = transformer_train_step
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, foreach=True, capturable=True)

        elif self.model_name in ["Resnet18", "Resnet50"]:
            inp = torch.randn(self.batch_size, 3, 224, 224, device=dev)
            num_classes = 10
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (inp, target)
            with torch.device(dev):
                self.model = resnet18() if self.model_name == "Resnet18" else resnet50()

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, foreach=True, capturable=True)
            self.train_step = resnet_train_step

        elif self.model_name == "Bert":
            vocab_size  = 8192
            seq_len     = 128
            bsz         = self.batch_size
            with torch.device(dev):
                self.model = SimpleBert(
                    vocab_size=vocab_size,
                    hidden=512,
                    num_layers=6,
                    num_heads=8,
                    max_seq_len=seq_len,
                )
            input_ids = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            labels    = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            self.example_inputs = (input_ids, labels)

            def bert_train_step(
                model: nn.Module, optimizer: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(
                    model(example_inputs[0]), example_inputs[1]
                )
                loss = SEPFunction.apply(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.train_step = bert_train_step
            self.optimizer  = optim.Adam(
                self.model.parameters(), lr=1e-4, foreach=True, capturable=True
            )

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def init_opt_states(self):
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
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

        if self.enable_ac:
            # Target: reduce peak activation memory by ac_reduction fraction.
            current_act_peak = sum(
                self.profiler._tensor_sizes.get(a, 0)
                for a in self.profiler.activation_nodes
            )
            budget = int(current_act_peak * (1.0 - self.ac_reduction))

            nodes_to_recompute, nodes_to_retain = mu_two_selection(
                self.profiler, memory_budget_bytes=budget
            )
            print_ac_plan(nodes_to_recompute, nodes_to_retain, self.profiler)
            self.nodes_to_recompute = nodes_to_recompute
            self.nodes_to_retain    = nodes_to_retain

            gm = apply_activation_checkpointing(
                gm, nodes_to_recompute, nodes_to_retain, self.profiler
            )

            # Update peak activation memory estimate: only retained activations
            # are kept alive through the idle window.
            act_peak_ac = sum(
                self.profiler._tensor_sizes.get(a, 0)
                for a in nodes_to_retain
            )
            self.peak_mem_by_cat = dict(self.peak_mem_by_cat)
            self.peak_mem_by_cat["activations"] = act_peak_ac

        return gm

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


def run_single(model_name: str, batch_size: int, enable_ac: bool = False) -> None:
    """Profile one model at one batch size and print stats."""
    ac_label = "with AC" if enable_ac else "no AC"
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}   Batch size: {batch_size}   ({ac_label})")
    print(f"{'='*60}")
    exp = Experiment(model_name, batch_size, enable_ac=enable_ac)
    exp.init_opt_states()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)


def measure_iteration_latency(
    compiled_fn,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    example_inputs: Any,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> float:
    """
    Run compiled_fn for n_iters measured iterations after n_warmup warmup
    calls.  Returns average iteration latency in milliseconds.

    Uses a single pair of CUDA Events bracketing all measured iterations so
    the per-iteration overhead of synchronize() is not included.
    """
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


def _run_sweep_single(
    model_name: str,
    enable_ac: bool,
) -> tuple:
    """
    Internal helper: sweep batch sizes for one model with or without AC.
    Returns (batch_sizes_used, memory_results, latency_results).
    """
    categories = ["params", "activations", "gradients", "opt_states", "other"]
    batch_sizes = list(sweep_batch_sizes[model_name])

    mem_results: Dict[str, List[float]] = {c: [] for c in categories}
    lat_results: List[float] = []
    valid_bs: List[int] = []

    for bs in batch_sizes:
        label = "AC" if enable_ac else "no AC"
        print(f"\n[sweep/{label}] {model_name}  batch_size={bs}")
        try:
            exp = Experiment(model_name, bs, enable_ac=enable_ac)
            exp.init_opt_states()
            compiled_fn = compile(exp.train_step, exp.graph_transformation)
            # First call: traces graph + profile + (optionally) rewrites
            compiled_fn(exp.model, exp.optimizer, exp.example_inputs)

            mem = exp.peak_mem_by_cat
            lat = measure_iteration_latency(
                compiled_fn, exp.model, exp.optimizer, exp.example_inputs
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={bs}, stopping sweep.")
            break

        valid_bs.append(bs)
        for cat in categories:
            mem_results[cat].append(mem.get(cat, 0) / 1e9)   # bytes → GB
        lat_results.append(lat)

    return valid_bs, mem_results, lat_results


def run_memory_sweep(model_name: str, out_dir: str = "records") -> None:
    """
    Sweep batch sizes for one model (no AC), collect peak memory per category,
    and save a stacked bar chart.  Kept for backward compatibility with
    phase1_run.py.
    """
    os.makedirs(out_dir, exist_ok=True)
    batch_sizes, results, _ = _run_sweep_single(model_name, enable_ac=False)

    categories = ["params", "activations", "gradients", "opt_states", "other"]
    colors     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    x       = range(len(batch_sizes))
    width   = 0.6
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = [0.0] * len(batch_sizes)
    for cat, color in zip(categories, colors):
        ax.bar(x, results[cat], width, bottom=bottoms, label=cat, color=color)
        bottoms = [b + v for b, v in zip(bottoms, results[cat])]

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title(f"{model_name} — Peak Memory vs Batch Size (no AC)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, f"peak_memory_{model_name.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved bar chart → {path}")


def run_comparison_sweep(model_name: str, out_dir: str = "records") -> None:
    """
    Run both a no-AC sweep and an AC sweep for one model, then save two
    comparison charts:
      1. peak_memory_comparison_<model>.png  — grouped bars w/ vs w/o AC
      2. latency_comparison_<model>.png      — line chart w/ vs w/o AC
    """
    os.makedirs(out_dir, exist_ok=True)
    categories = ["params", "activations", "gradients", "opt_states", "other"]
    colors     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    print(f"\n{'='*60}")
    print(f"  Comparison sweep (no AC) — {model_name}")
    print(f"{'='*60}")
    bs_no, mem_no, lat_no = _run_sweep_single(model_name, enable_ac=False)

    print(f"\n{'='*60}")
    print(f"  Comparison sweep (with AC) — {model_name}")
    print(f"{'='*60}")
    bs_ac, mem_ac, lat_ac = _run_sweep_single(model_name, enable_ac=True)

    # Use the intersection of batch sizes that succeeded for both runs
    common_bs = [bs for bs in bs_no if bs in bs_ac]
    idx_no = [bs_no.index(bs) for bs in common_bs]
    idx_ac = [bs_ac.index(bs) for bs in common_bs]

    # ------------------------------------------------------------------ #
    # Chart 1: Peak memory comparison — grouped stacked bars             #
    # ------------------------------------------------------------------ #
    n   = len(common_bs)
    x   = list(range(n))
    w   = 0.35                      # half-width of each group
    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 5))

    for gi, (indices, alpha, edge) in enumerate(
        [(idx_no, 1.0, "no AC"), (idx_ac, 0.55, "AC")]
    ):
        bottoms = [0.0] * n
        offset  = -w / 2 if gi == 0 else w / 2
        mem_src = mem_no if gi == 0 else mem_ac
        for ci, (cat, color) in enumerate(zip(categories, colors)):
            vals = [mem_src[cat][i] for i in indices]
            bars = ax.bar(
                [xi + offset for xi in x], vals, w,
                bottom=bottoms, color=color,
                alpha=alpha,
                label=f"{cat} ({edge})" if gi == 0 else f"_{cat}_ac",
            )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in common_bs])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title(f"{model_name} — Peak Memory: no AC (solid) vs AC (faded)")
    # Deduplicate legend to show each category once
    handles, labels = ax.get_legend_handles_labels()
    seen: Dict[str, Any] = {}
    for h, l in zip(handles, labels):
        if not l.startswith("_") and l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)
    plt.tight_layout()
    path1 = os.path.join(out_dir, f"peak_memory_comparison_{model_name.lower()}.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"\nSaved memory comparison → {path1}")

    # ------------------------------------------------------------------ #
    # Chart 2: Latency comparison — line chart                           #
    # ------------------------------------------------------------------ #
    lat_no_common = [lat_no[i] for i in idx_no]
    lat_ac_common = [lat_ac[i] for i in idx_ac]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(common_bs, lat_no_common, "o-", color="#4C72B0", label="No AC")
    ax.plot(common_bs, lat_ac_common, "s--", color="#DD8452", label="With AC")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Iteration latency (ms)")
    ax.set_title(f"{model_name} — Iteration Latency vs Batch Size")
    ax.legend()
    plt.tight_layout()
    path2 = os.path.join(out_dir, f"latency_comparison_{model_name.lower()}.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved latency comparison → {path2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=model_names, default="Resnet18",
        help="Model to profile",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "sweep", "compare"],
        default="single",
        help=(
            "'single'  — profile at one batch size; "
            "'sweep'   — peak-memory bar graph (no AC); "
            "'compare' — side-by-side memory + latency charts w/ and w/o AC"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size for --mode single",
    )
    parser.add_argument(
        "--ac", action="store_true",
        help="Enable activation checkpointing for --mode single",
    )
    args = parser.parse_args()

    if args.mode == "single":
        bs = args.batch_size or model_batch_sizes[args.model]
        run_single(args.model, bs, enable_ac=args.ac)
    elif args.mode == "sweep":
        run_memory_sweep(args.model)
    else:
        run_comparison_sweep(args.model)
