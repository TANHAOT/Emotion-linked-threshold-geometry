#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "qwen3_32b_reference.json"


def run(cmd):
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def add_runtime_overrides(parser):
    parser.add_argument("--sae-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-id")
    parser.add_argument("--model-dir")
    parser.add_argument("--sae-repo-id")
    parser.add_argument("--sae-local-dir")
    parser.add_argument("--sae-file")
    parser.add_argument("--sae-config-file")
    parser.add_argument("--sae-filename")
    parser.add_argument("--sae-loader", choices=(
        "dictionary_batch_topk", "dictionary_matryoshka_batch_topk", "dictionary_topk", "dictionary_jumprelu",
        "dictionary_relu", "gemma_scope_jumprelu",
    ))
    parser.add_argument("--sae-layer", type=int)
    parser.add_argument("--sae-trainer")
    parser.add_argument("--hook-module")


def runtime_args(args):
    values = ["--sae_config", str(args.sae_config)]
    mapping = {
        "model_id": "--model-id", "model_dir": "--model-dir",
        "sae_repo_id": "--sae-repo-id", "sae_local_dir": "--sae-local-dir",
        "sae_file": "--sae-file", "sae_config_file": "--sae-config-file",
        "sae_filename": "--sae-filename", "sae_loader": "--sae-loader",
        "sae_layer": "--sae-layer", "sae_trainer": "--sae-trainer",
        "hook_module": "--hook-module",
    }
    for attr, flag in mapping.items():
        value = getattr(args, attr)
        if value is not None:
            values.extend([flag, str(value)])
    return values


def main():
    p = argparse.ArgumentParser(description="Unfairness-related SAE screening and steering with a configurable model/SAE profile.")
    p.add_argument("stage", choices=["prepare", "screen", "steer"])
    p.add_argument("--input-dir", type=Path, default=HERE / "inputs")
    p.add_argument("--output-root", type=Path, default=HERE / "results")
    p.add_argument("--subject-start", type=int, default=None)
    p.add_argument("--subject-end", type=int, default=None)
    p.add_argument("--target-id", type=int, default=16217)
    p.add_argument("--multipliers", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2])
    p.add_argument("--steering-method", choices=["replace", "delta"], default="delta")
    p.add_argument("--seed", type=int, default=20260714)
    add_runtime_overrides(p)
    a = p.parse_args()

    if a.stage == "prepare":
        start = 400 if a.subject_start is None else a.subject_start
        end = 420 if a.subject_end is None else a.subject_end
        run([sys.executable, HERE / "prepare_inputs.py", "--start", str(start), "--end", str(end), "--output-dir", a.input_dir])
        return

    runner = HERE / "behavior" / "sae_tpp_runner.py"
    common = [sys.executable, runner, "--input_dir", a.input_dir, "--output_root", a.output_root, "--seed", str(a.seed)] + runtime_args(a)

    if a.stage == "screen":
        start = 400 if a.subject_start is None else a.subject_start
        end = 420 if a.subject_end is None else a.subject_end
        run(common + ["--subjnum", str(start), "--subject_end", str(end)])
        print("Use rank_unfairness_features.py on the baseline result directory created for this SAE profile.")
        return

    start = 0 if a.subject_start is None else a.subject_start
    end = 1017 if a.subject_end is None else a.subject_end
    cmd = common + [
        "--subjnum", str(start), "--subject_end", str(end), "--steer",
        "--target_id", str(a.target_id), "--multipliers", *map(str, a.multipliers),
        "--steering_method", a.steering_method, "--resume",
    ]
    run(cmd)


if __name__ == "__main__":
    main()
