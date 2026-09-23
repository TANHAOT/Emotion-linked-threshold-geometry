#!/usr/bin/env python3
"""Extract maximal per-text SAE feature activations for a configurable model/SAE pair."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SECTION_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "project_core").is_dir()), None)
if REPO_ROOT is None:
    raise RuntimeError("Could not locate the project root containing project_core/.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from project_core.sae_runtime import (
    add_sae_override_arguments, apply_sae_overrides, describe_profile,
    load_sae_profile, load_sae_runtime,
)


def main(args):
    import torch

    profile = apply_sae_overrides(load_sae_profile(args.sae_config), args)
    print("SAE_CONFIG", describe_profile(profile))
    runtime = load_sae_runtime(profile, SECTION_DIR / "code" / "vendor")
    tokenizer, model, sae, submodule = runtime.tokenizer, runtime.model, runtime.sae, runtime.submodule

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    captured = {}

    def hook(module, inputs, output):
        del module, inputs
        captured["hidden"] = output if isinstance(output, torch.Tensor) else output[0]

    handle = submodule.register_forward_hook(hook)
    row_ids, maximums, filtered_counts = [], [], []
    started = time.time()
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                encoded = tokenizer(
                    [row["text"] for row in batch], padding=True, truncation=True,
                    max_length=args.max_length, return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                special = encoded.pop("special_tokens_mask").bool()
                encoded = encoded.to(runtime.input_device)
                model(**encoded, use_cache=False)
                hidden = captured.pop("hidden")
                content = encoded.attention_mask.to(hidden.device).bool() & ~special.to(hidden.device)
                norms = hidden.float().norm(dim=-1)
                masked_norms = norms.masked_fill(~content, float("nan"))
                medians = torch.nanmedian(masked_norms, dim=1).values.clamp_min(1e-6)
                sinks = (norms > args.sink_ratio * medians[:, None]) & content
                valid = content & ~sinks
                acts = sae.encode(hidden.to(dtype=sae.W_enc.dtype, device=sae.W_enc.device)).float()
                max_acts = acts.masked_fill(~valid.to(acts.device).unsqueeze(-1), 0.0).amax(dim=1)
                row_ids.extend(row["row_id"] for row in batch)
                maximums.append(max_acts.cpu().numpy().astype(np.float16))
                filtered_counts.extend(sinks.sum(dim=1).cpu().tolist())
                print(
                    f"EXTRACT {min(start + len(batch), len(rows))}/{len(rows)} "
                    f"elapsed={time.time() - started:.1f}s sinks={sum(filtered_counts)}",
                    flush=True,
                )
    finally:
        handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        row_id=np.asarray(row_ids, dtype=np.int32),
        max=np.concatenate(maximums),
        filtered_tokens=np.asarray(filtered_counts, dtype=np.int16),
    )
    metadata = {
        "profile_name": runtime.profile_name,
        "profile_tag": runtime.profile_tag,
        "profile_fingerprint": runtime.fingerprint,
        "model": profile["model"].get("repo_id"),
        "sae_loader": profile["sae"].get("loader"),
        "sae_repository": profile["sae"].get("repo_id"),
        "sae_local_file": profile["sae"].get("local_file"),
        "sae_filename": profile["sae"].get("filename"),
        "sae_layer": runtime.layer,
        "sae_trainer": runtime.trainer,
        "feature_count": int(sae.W_enc.shape[-1]),
        "dataset": str(args.dataset),
        "sink_ratio": args.sink_ratio,
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sae-config", "--sae_config", dest="sae_config", type=Path, default=SECTION_DIR / "config" / "qwen3_32b_reference.json")
    add_sae_override_arguments(p)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--sink-ratio", type=float, default=10.0)
    main(p.parse_args())
