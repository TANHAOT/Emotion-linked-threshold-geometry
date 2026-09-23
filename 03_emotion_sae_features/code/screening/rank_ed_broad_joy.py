#!/usr/bin/env python3
"""Rank Trainer-1 SAE features across three broad-joy subtypes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SUBTYPES = ("joyful", "excited", "content")


def safe_auc(labels, values):
    if len(np.unique(labels)) < 2 or np.all(values == values[0]):
        return 0.5
    return float(roc_auc_score(labels, values))


def effect_size(positive, control):
    pooled = np.sqrt((positive.var(axis=0) + control.var(axis=0)) / 2 + 1e-8)
    return (positive.mean(axis=0) - control.mean(axis=0)) / pooled


def main(args):
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    by_id = {row["row_id"]: row for row in rows}
    stored = np.load(args.activations)
    order = np.argsort(stored["row_id"])
    row_ids = stored["row_id"][order]
    activations = stored["max"][order].astype(np.float32)
    metadata = [by_id[int(row_id)] for row_id in row_ids]
    split = np.asarray([row["split"] for row in metadata])
    labels = np.asarray([row["labels"]["joy"] for row in metadata])
    kinds = np.asarray([row["joy_kind"] for row in metadata])
    subtypes = np.asarray([row["joy_subtype"] for row in metadata])

    discovery_positive = (split == "discovery") & (labels == 1)
    discovery_control = (split == "discovery") & (labels == 0)
    implicit_positive = discovery_positive & (kinds == "implicit")
    positive = activations[discovery_positive]
    control = activations[discovery_control]
    implicit = activations[implicit_positive]

    d_all = effect_size(positive, control)
    d_implicit = effect_size(implicit, control)
    coverage = (positive > 0).mean(axis=0)
    delta_all = positive.mean(axis=0) - control.mean(axis=0)
    delta_implicit = implicit.mean(axis=0) - control.mean(axis=0)
    subtype_d, subtype_coverage, subtype_delta = {}, {}, {}
    for subtype in SUBTYPES:
        values = activations[discovery_positive & (subtypes == subtype)]
        subtype_d[subtype] = effect_size(values, control)
        subtype_coverage[subtype] = (values > 0).mean(axis=0)
        subtype_delta[subtype] = values.mean(axis=0) - control.mean(axis=0)

    conservative_d = np.minimum.reduce(
        [d_all, d_implicit, *(subtype_d[subtype] for subtype in SUBTYPES)]
    )
    eligible_mask = (
        (coverage >= args.minimum_coverage)
        & (delta_all > 0)
        & (delta_implicit > 0)
        & np.isfinite(conservative_d)
    )
    for subtype in SUBTYPES:
        eligible_mask &= subtype_coverage[subtype] >= args.minimum_subtype_coverage
        eligible_mask &= subtype_delta[subtype] > 0
    eligible = np.where(eligible_mask)[0]
    eligible_order = eligible[np.argsort(conservative_d[eligible])[::-1]]
    preliminary = eligible_order[:args.preliminary]
    conservative_rank = {int(feature): rank + 1 for rank, feature in enumerate(eligible_order)}

    def calculate(feature_id):
        item = {
            "emotion": "joy",
            "feature_id": int(feature_id),
            "eligible": bool(eligible_mask[feature_id]),
            "conservative_effect_rank": conservative_rank.get(int(feature_id)),
            "coverage": float(coverage[feature_id]),
            "cohen_d_all": float(d_all[feature_id]),
            "cohen_d_implicit": float(d_implicit[feature_id]),
            "conservative_cohen_d": float(conservative_d[feature_id]),
        }
        for subtype in SUBTYPES:
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
            for subtype in SUBTYPES:
                subtype_mask = base & ((labels == 0) | (subtypes == subtype))
                item[f"{evaluation_split}_{subtype}_auc"] = safe_auc(
                    labels[subtype_mask], activations[subtype_mask, feature_id]
                )
        active_values = positive[:, feature_id]
        active_values = active_values[active_values > 0]
        item["natural_scale"] = float(np.median(active_values)) if len(active_values) else 1.0
        item["rank_score"] = float(
            min(
                item["discovery_implicit_auc"],
                *(item[f"discovery_{subtype}_auc"] for subtype in SUBTYPES),
            )
            + 0.10 * item["discovery_auc"]
        )
        return item

    candidates = [calculate(int(feature_id)) for feature_id in preliminary]
    candidates.sort(key=lambda item: item["rank_score"], reverse=True)
    score_rank = {item["feature_id"]: rank + 1 for rank, item in enumerate(candidates)}
    for item in candidates:
        item["rank_score_rank"] = score_rank[item["feature_id"]]

    selected = candidates[:args.top_k]
    for item in selected:
        feature_id = item["feature_id"]
        indices = np.argsort(activations[:, feature_id])[::-1]
        item["top_activating_examples"] = [
            {
                "activation": float(activations[index, feature_id]),
                "text": metadata[index]["text"],
                "source_label": metadata[index]["source_label"],
                "target_label": int(labels[index]),
                "split": metadata[index]["split"],
                "joy_kind": metadata[index]["joy_kind"],
            }
            for index in indices[:args.examples]
        ]

    probe_rows = []
    for feature_id in args.probe_features:
        probe = next((item for item in candidates if item["feature_id"] == feature_id), None)
        if probe is None:
            probe = calculate(feature_id)
            probe["rank_score_rank"] = None
        probe_rows.append({key: value for key, value in probe.items() if key != "top_activating_examples"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "Qwen3-32B",
        "sae_layer": 48,
        "sae_trainer": 1,
        "dictionary_width": int(activations.shape[1]),
        "corpus": "EmpatheticDialogues-only broad joy",
        "positive_subtypes": list(SUBTYPES),
        "rankings": {"joy": selected},
        "probe_features": probe_rows,
    }
    (args.output_dir / "candidate_rankings.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame([
        {key: value for key, value in item.items() if key != "top_activating_examples"}
        for item in selected
    ]).to_csv(args.output_dir / "candidate_rankings.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(args.output_dir / "probe_features.csv", index=False)
    print(json.dumps({"top5": selected[:5], "probes": probe_rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--minimum_coverage", type=float, default=0.10)
    parser.add_argument("--minimum_subtype_coverage", type=float, default=0.05)
    parser.add_argument("--preliminary", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--probe_features", type=int, nargs="+", default=[3271])
    main(parser.parse_args())
