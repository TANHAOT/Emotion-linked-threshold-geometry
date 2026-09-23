#!/usr/bin/env python3
"""Model-independent ranking of SAE features for one target emotion."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_auc(labels, values):
    if len(np.unique(labels)) < 2 or np.all(values == values[0]):
        return 0.5
    return float(roc_auc_score(labels, values))


def effect_size(positive, control):
    pooled = np.sqrt((positive.var(axis=0) + control.var(axis=0)) / 2 + 1e-8)
    return (positive.mean(axis=0) - control.mean(axis=0)) / pooled


def main(args):
    records = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    by_id = {row["row_id"]: row for row in records}
    stored = np.load(args.activations)
    order = np.argsort(stored["row_id"])
    activations = stored["max"][order].astype(np.float32)
    metadata = [by_id[int(row_id)] for row_id in stored["row_id"][order]]

    split = np.asarray([row["split"] for row in metadata])
    labels = np.asarray([row["labels"][args.emotion] for row in metadata])
    kinds = np.asarray([row[f"{args.emotion}_kind"] for row in metadata])
    subtypes = np.asarray([row[f"{args.emotion}_subtype"] for row in metadata])
    positive_subtypes = sorted(set(subtypes[labels == 1]))

    discovery_positive = (split == "discovery") & (labels == 1)
    discovery_control = (split == "discovery") & (labels == 0)
    discovery_implicit = discovery_positive & (kinds == "implicit")
    positive = activations[discovery_positive]
    control = activations[discovery_control]
    implicit = activations[discovery_implicit]

    d_all = effect_size(positive, control)
    d_implicit = effect_size(implicit, control)
    coverage = (positive > 0).mean(axis=0)
    delta_all = positive.mean(axis=0) - control.mean(axis=0)
    delta_implicit = implicit.mean(axis=0) - control.mean(axis=0)
    subtype_d, subtype_coverage, subtype_delta = {}, {}, {}
    for subtype in positive_subtypes:
        values = activations[discovery_positive & (subtypes == subtype)]
        subtype_d[subtype] = effect_size(values, control)
        subtype_coverage[subtype] = (values > 0).mean(axis=0)
        subtype_delta[subtype] = values.mean(axis=0) - control.mean(axis=0)

    conservative_d = np.minimum.reduce(
        [d_all, d_implicit, *(subtype_d[name] for name in positive_subtypes)]
    )
    eligible = (
        (coverage >= args.minimum_coverage)
        & (delta_all > 0)
        & (delta_implicit > 0)
        & np.isfinite(conservative_d)
    )
    for subtype in positive_subtypes:
        eligible &= subtype_coverage[subtype] >= args.minimum_subtype_coverage
        eligible &= subtype_delta[subtype] > 0

    feature_ids = np.where(eligible)[0]
    feature_ids = feature_ids[np.argsort(conservative_d[feature_ids])[::-1]][: args.preliminary]
    rows = []
    for feature_id in feature_ids:
        item = {
            "emotion": args.emotion,
            "feature_id": int(feature_id),
            "coverage": float(coverage[feature_id]),
            "cohen_d_all": float(d_all[feature_id]),
            "cohen_d_implicit": float(d_implicit[feature_id]),
            "conservative_cohen_d": float(conservative_d[feature_id]),
        }
        for subtype in positive_subtypes:
            item[f"coverage_{subtype}"] = float(subtype_coverage[subtype][feature_id])
            item[f"cohen_d_{subtype}"] = float(subtype_d[subtype][feature_id])
        for evaluation_split in ("discovery", "heldout"):
            base = split == evaluation_split
            item[f"{evaluation_split}_auc"] = safe_auc(
                labels[base], activations[base, feature_id]
            )
            implicit_mask = base & ((labels == 0) | (kinds == "implicit"))
            item[f"{evaluation_split}_implicit_auc"] = safe_auc(
                labels[implicit_mask], activations[implicit_mask, feature_id]
            )
            for subtype in positive_subtypes:
                subtype_mask = base & ((labels == 0) | (subtypes == subtype))
                item[f"{evaluation_split}_{subtype}_auc"] = safe_auc(
                    labels[subtype_mask], activations[subtype_mask, feature_id]
                )
        nonzero = positive[:, feature_id]
        nonzero = nonzero[nonzero > 0]
        item["natural_scale"] = float(np.median(nonzero)) if len(nonzero) else 1.0
        item["rank_score"] = float(
            min(
                item["discovery_implicit_auc"],
                *(item[f"discovery_{name}_auc"] for name in positive_subtypes),
            )
            + 0.10 * item["discovery_auc"]
        )
        rows.append(item)

    rows.sort(key=lambda item: item["rank_score"], reverse=True)
    selected = rows[: args.top_k]
    for item in selected:
        feature_id = item["feature_id"]
        indices = np.argsort(activations[:, feature_id])[::-1][: args.examples]
        item["top_activating_examples"] = [
            {
                "activation": float(activations[index, feature_id]),
                "text": metadata[index]["text"],
                "source_label": metadata[index]["source_label"],
                "split": metadata[index]["split"],
            }
            for index in indices
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.activations.with_suffix(args.activations.suffix + ".metadata.json")
    activation_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    model_name = args.model_name or activation_metadata.get("profile_name") or activation_metadata.get("model") or "unspecified"
    payload = {
        "model": model_name,
        "activation_metadata": activation_metadata,
        "emotion": args.emotion,
        "dictionary_width": int(activations.shape[1]),
        "positive_subtypes": positive_subtypes,
        "selection_used_heldout_data": False,
        "rankings": selected,
    }
    (args.output_dir / "candidate_rankings.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(
        [{key: value for key, value in item.items() if key != "top_activating_examples"}
         for item in selected]
    ).to_csv(args.output_dir / "candidate_rankings.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", choices=("anger", "joy"), required=True)
    parser.add_argument("--model_name", default=None, help="Optional label; defaults to activation metadata when available.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--minimum_coverage", type=float, default=0.10)
    parser.add_argument("--minimum_subtype_coverage", type=float, default=0.05)
    parser.add_argument("--preliminary", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--examples", type=int, default=20)
    main(parser.parse_args())

