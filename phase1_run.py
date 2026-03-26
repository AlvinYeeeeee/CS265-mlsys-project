"""
phase1_run.py — Phase 1 deliverable runner

Runs all experiments required for the Phase 1 (midway) check-in:
  1. Profile each model at its default batch size → per-node stats table
     saved to records/stats_<model>.txt
  2. Sweep batch sizes for each model → peak-memory stacked bar chart
     saved to records/peak_memory_<model>.png

Usage
-----
    # Everything (all models, profile + sweep):
    conda run -n cs265 python phase1_run.py

    # One model only:
    conda run -n cs265 python phase1_run.py --model Resnet18

    # Skip the sweep (just profiling stats):
    conda run -n cs265 python phase1_run.py --no-sweep

    # Skip profiling (just bar charts):
    conda run -n cs265 python phase1_run.py --no-profile
"""

import argparse
import os
import sys
from io import StringIO
from typing import List

from benchmarks import (
    Experiment,
    model_names,
    model_batch_sizes,
    run_memory_sweep,
)
from graph_tracer import compile

RECORDS_DIR = "records"


def profile_model(model_name: str) -> None:
    """
    Profile one model at its default batch size and save the stats
    table to records/stats_<model>.txt.
    """
    bs = model_batch_sizes[model_name]
    print(f"\n{'='*60}")
    print(f"  Profiling  {model_name}  (batch_size={bs})")
    print(f"{'='*60}")

    exp = Experiment(model_name, bs)
    exp.init_opt_states()

    # Capture print_stats output so we can tee it to a file
    captured = StringIO()
    original_stdout = sys.stdout

    class Tee:
        def write(self, s):
            original_stdout.write(s)
            captured.write(s)
        def flush(self):
            original_stdout.flush()

    sys.stdout = Tee()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
    sys.stdout = original_stdout

    out_path = os.path.join(RECORDS_DIR, f"stats_{model_name.lower()}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}   Batch size: {bs}\n")
        f.write(captured.getvalue())
    print(f"\n  Stats saved → {out_path}")


def main(models: List[str], do_profile: bool, do_sweep: bool) -> None:
    os.makedirs(RECORDS_DIR, exist_ok=True)

    for model_name in models:
        if do_profile:
            profile_model(model_name)

        if do_sweep:
            print(f"\n{'='*60}")
            print(f"  Memory sweep  {model_name}")
            print(f"{'='*60}")
            run_memory_sweep(model_name, out_dir=RECORDS_DIR)

    print(f"\n{'='*60}")
    print("  Phase 1 complete. Outputs in records/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 deliverable runner")
    parser.add_argument(
        "--model",
        choices=model_names,
        default=None,
        help="Run a single model only (default: all models)",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip per-node profiling stats",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip batch-size memory sweep / bar chart",
    )
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else model_names
    main(
        models=models_to_run,
        do_profile=not args.no_profile,
        do_sweep=not args.no_sweep,
    )
