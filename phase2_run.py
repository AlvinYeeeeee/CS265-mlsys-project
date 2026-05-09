"""
phase2_run.py

Runs all experiments required for the final project check-in:
  1. For each model, run the full comparison sweep (no AC vs with AC):
       - Peak memory comparison bar chart  → records/peak_memory_comparison_<model>.png
       - Iteration latency comparison line chart → records/latency_comparison_<model>.png
  2. Optionally run a single model+batch-size to inspect the AC plan and
     verify the modified graph produces the same results.

Usage
-----
    # All models, full comparison sweep:
    conda run -n cs265 python phase2_run.py

    # One model only:
    conda run -n cs265 python phase2_run.py --model Resnet18

    # Quick single-model sanity check (no charts):
    conda run -n cs265 python phase2_run.py --model Resnet18 --mode single

    # Single check WITH AC enabled:
    conda run -n cs265 python phase2_run.py --model Resnet18 --mode single --ac
"""

import argparse
import os
import sys
from io import StringIO
from typing import List

from benchmarks import (
    model_names,
    model_batch_sizes,
    Experiment,
    run_comparison_sweep,
    run_single,
)
from graph_tracer import compile

RECORDS_DIR = "records"


def save_stats(model_name: str, enable_ac: bool) -> None:
    """
    Profile one model at its default batch size (with or without AC) and
    save the full print_stats + AC plan output to a text file.

    Output path:
      records/stats_<model>_noac.txt   (enable_ac=False)
      records/stats_<model>_ac.txt     (enable_ac=True)
    """
    bs     = model_batch_sizes[model_name]
    label  = "ac" if enable_ac else "noac"
    suffix = "with AC" if enable_ac else "no AC"

    print(f"\n{'='*60}")
    print(f"  Stats  {model_name}  (batch_size={bs}, {suffix})")
    print(f"{'='*60}")

    exp = Experiment(model_name, bs, enable_ac=enable_ac)
    exp.init_opt_states()

    # Tee: write to both terminal and an in-memory buffer simultaneously
    captured        = StringIO()
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

    out_path = os.path.join(RECORDS_DIR, f"stats_{model_name.lower()}_{label}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}   Batch size: {bs}   ({suffix})\n")
        f.write(captured.getvalue())
    print(f"  Stats saved → {out_path}")


def main(models: List[str], mode: str, ac: bool) -> None:
    os.makedirs(RECORDS_DIR, exist_ok=True)

    for model_name in models:
        print(f"\n{'#'*65}")
        print(f"#  {model_name}")
        print(f"{'#'*65}")

        if mode == "single":
            bs = model_batch_sizes[model_name]
            run_single(model_name, bs, enable_ac=ac)

        elif mode == "compare":
            # Save profiling stats at the default batch size (both no-AC and AC)
            save_stats(model_name, enable_ac=False)
            save_stats(model_name, enable_ac=True)
            # Run the full batch-size sweep and generate comparison charts
            run_comparison_sweep(model_name, out_dir=RECORDS_DIR)

    print(f"\n{'='*65}")
    print("  Phase 2 complete.  Outputs in records/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 & 3 deliverable runner. "
            "No arguments = run full comparison sweep on ALL models."
        )
    )
    parser.add_argument(
        "--model",
        choices=model_names,
        default=None,
        help=f"Limit to one model. Choices: {model_names}. Default: all models.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "compare"],
        default="compare",
        help=(
            "'compare' (default) — full batch-size sweep, saves memory + latency "
            "comparison charts for each model; "
            "'single'  — profile at the default batch size and print the AC plan "
            "(useful for a quick sanity check)"
        ),
    )
    parser.add_argument(
        "--ac",
        action="store_true",
        help="Enable AC when using --mode single (ignored for --mode compare).",
    )
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else model_names
    main(models=models_to_run, mode=args.mode, ac=args.ac)
