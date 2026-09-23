#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "qwen3_32b_reference.json"


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
    p = argparse.ArgumentParser(description="Run TPP steering for an emotion-related feature using any configured compatible SAE.")
    p.add_argument("--emotion", choices=["anger", "joy"], required=True)
    p.add_argument("--feature-id", type=int, required=True, help="Feature ID selected for this exact model/SAE checkpoint.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1017)
    p.add_argument("--input-dir", type=Path, default=HERE / "inputs")
    p.add_argument("--output-root", type=Path, default=HERE / "results")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--multipliers", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2])
    p.add_argument("--steering-method", choices=["delta", "replace"], default="delta")
    p.add_argument("--seed", type=int, default=20260714)
    add_runtime_overrides(p)
    a = p.parse_args()

    cmd = [
        sys.executable, HERE / "code" / "behavior" / "sae_tpp_runner.py",
        "--subjnum", str(a.start), "--subject_end", str(a.end),
        "--input_dir", a.input_dir, "--output_root", a.output_root,
        "--steer", "--target_id", str(a.feature_id),
        "--multipliers", *map(str, a.multipliers),
        "--steering_method", a.steering_method,
        "--seed", str(a.seed), "--resume", *runtime_args(a),
        "--output_tag", a.emotion,
    ]
    if a.no_report:
        cmd.append("--noemotion")
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
