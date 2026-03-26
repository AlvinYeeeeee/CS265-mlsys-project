import argparse
from typing import Any, Dict, List

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


model_names: List[str] = [
    "Transformer",
    "Resnet18",
    "Resnet50",
]

model_batch_sizes: Dict[str, int] = {
    "Transformer": 4,
    "Resnet18": 16,
    "Resnet50": 4,
}

# Batch sizes to sweep for the peak-memory bar graph (deliverable 4b).
sweep_batch_sizes: Dict[str, List[int]] = {
    "Transformer": [1, 2, 4, 8],
    "Resnet18":    [4, 8, 16, 32, 64],
    "Resnet50":    [1, 2, 4, 8, 16],
}


class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        assert model_name in model_names, f"Model {model_name} not found in model names {model_names}"
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size

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
        return gm

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


def run_single(model_name: str, batch_size: int) -> None:
    """Profile one model at one batch size and print stats."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}   Batch size: {batch_size}")
    print(f"{'='*60}")
    exp = Experiment(model_name, batch_size)
    exp.init_opt_states()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)


def run_memory_sweep(model_name: str, out_dir: str = "records") -> None:
    """
    Sweep batch sizes for one model, collect peak memory per category,
    and save a stacked bar chart to records/.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = sweep_batch_sizes[model_name]
    categories  = ["params", "activations", "gradients", "opt_states", "other"]
    colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    results: Dict[str, List[float]] = {c: [] for c in categories}

    for bs in batch_sizes:
        print(f"\n[sweep] {model_name}  batch_size={bs}")
        try:
            exp = Experiment(model_name, bs)
            exp.init_opt_states()
            compiled_fn = compile(exp.train_step, exp.graph_transformation)
            compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
            mem = exp.peak_mem_by_cat
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={bs}, stopping sweep.")
            batch_sizes = batch_sizes[: batch_sizes.index(bs)]
            break

        for cat in categories:
            results[cat].append(mem.get(cat, 0) / 1e9)   # bytes -> GB

    # --- plot ---
    x      = range(len(batch_sizes))
    width  = 0.6
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = [0.0] * len(batch_sizes)

    for cat, color in zip(categories, colors):
        vals = results[cat]
        ax.bar(x, vals, width, bottom=bottoms, label=cat, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=model_names, default="Resnet18",
        help="Model to profile",
    )
    parser.add_argument(
        "--mode", choices=["single", "sweep"], default="single",
        help="'single' profiles at the default batch size; "
             "'sweep' generates the peak-memory bar graph",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size for --mode single",
    )
    args = parser.parse_args()

    if args.mode == "single":
        bs = args.batch_size or model_batch_sizes[args.model]
        run_single(args.model, bs)
    else:
        run_memory_sweep(args.model)
