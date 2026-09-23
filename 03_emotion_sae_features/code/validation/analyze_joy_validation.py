#!/usr/bin/env python3
"""Analyze Hartmann seven-class dose curves for EmpatheticDialogues joy candidates."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests


HARTMANN = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
SUBTYPES = ("joyful", "excited", "content")


def paired_p(group, maximum_dose):
    paired = group.pivot_table(index="word", columns="dose", values="hartmann_joy")
    try:
        return float(wilcoxon(paired[maximum_dose], paired[0.0], alternative="greater").pvalue)
    except (KeyError, ValueError):
        return 1.0


def main(args):
    frame = pd.read_csv(args.input_csv)
    frame = frame[(frame.emotion == "joy") & (frame.dose <= args.max_dose)].copy()
    rankings = pd.read_csv(args.ranking_csv).set_index("feature_id")
    probes = pd.read_csv(args.probe_csv).set_index("feature_id")
    metrics = pd.concat([rankings, probes[~probes.index.isin(rankings.index)]])
    rows = []
    non_target_labels = [label for label in HARTMANN if label != "joy"]
    for feature_id, group in frame.groupby("feature_id"):
        curve = group.groupby("dose").hartmann_joy.mean().sort_index()
        rho, trend_p = spearmanr(curve.index, curve.values)
        non_target = {}
        for label in non_target_labels:
            values = group.groupby("dose")[f"hartmann_{label}"].mean()
            non_target[label] = float(values.loc[args.max_dose] - values.loc[0.0])
        quality = group.groupby("dose").agg(
            word_present_rate=("word_present", "mean"),
            repetition=("trigram_repetition", "mean"),
            word_count=("word_count", "mean"),
        )
        target_delta = float(curve.loc[args.max_dose] - curve.loc[0.0])
        rank = metrics.loc[feature_id]
        row = {
            "feature_id": int(feature_id),
            "heldout_auc": float(rank.heldout_auc),
            "heldout_implicit_auc": float(rank.heldout_implicit_auc),
            "natural_scale": float(group.natural_scale.iloc[0]),
            "baseline_joy_probability": float(curve.loc[0.0]),
            "dose3_joy_probability": float(curve.loc[args.max_dose]),
            "target_delta": target_delta,
            "dose_spearman_rho": float(rho),
            "dose_trend_p": float(trend_p),
            "paired_p": paired_p(group, args.max_dose),
            "maximum_non_target_delta": max(non_target.values()),
            "largest_non_target": max(non_target, key=non_target.get),
            "specificity_margin": target_delta - max(non_target.values()),
            "minimum_word_present_rate": float(quality.word_present_rate.min()),
            "maximum_mean_repetition": float(quality.repetition.max()),
            "maximum_length_ratio": float(
                quality.word_count.max() / max(quality.loc[0.0, "word_count"], 1)
            ),
        }
        for subtype in SUBTYPES:
            row[f"heldout_{subtype}_auc"] = float(rank[f"heldout_{subtype}_auc"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["fdr_q"] = multipletests(summary.paired_p, method="fdr_bh")[1]
    quality = (
        (summary.minimum_word_present_rate >= 0.98)
        & (summary.maximum_mean_repetition <= 0.02)
        & (summary.maximum_length_ratio <= 1.5)
    )
    summary["validation_status"] = np.where(
        (summary.target_delta > 0)
        & (summary.dose_spearman_rho >= 0.70)
        & (summary.fdr_q < 0.05)
        & (summary.specificity_margin > 0)
        & quality,
        "validated",
        "not_validated",
    )
    summary["selection_score"] = (
        summary.target_delta
        + 0.10 * summary.dose_spearman_rho
        + 0.20 * summary.heldout_implicit_auc
        + 0.10 * summary.specificity_margin
    )
    summary = summary.sort_values(
        ["validation_status", "selection_score"], ascending=[False, False]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    selected = summary.head(args.selection_count).feature_id.astype(int).tolist()
    (args.output_dir / "selected_features.json").write_text(json.dumps({
        "model": "Qwen3-32B", "sae_layer": 48, "sae_trainer": 1,
        "corpus": "EmpatheticDialogues-only broad joy",
        "features": {"joy": selected},
    }, indent=2))

    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(10, 6))
    for feature_id, group in frame.groupby("feature_id"):
        curve = group.groupby("dose").hartmann_joy.agg(["mean", "sem"]).reset_index()
        axis.errorbar(
            curve.dose, curve["mean"], yerr=1.96 * curve["sem"],
            marker="o", label=f"Feature {feature_id}",
        )
    axis.set_title("Qwen3-32B Trainer-1 EmpatheticDialogues joy candidate screen")
    axis.set_xlabel("Steering dose x natural activation scale")
    axis.set_ylabel("Hartmann joy probability")
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "joy_candidate_dose_curves.png", dpi=220)
    plt.close(fig)

    chosen = 3271 if 3271 in set(summary.feature_id) else selected[0]
    group = frame[frame.feature_id == chosen]
    fig, axis = plt.subplots(figsize=(10, 6))
    deltas = {}
    for label in HARTMANN:
        curve = group.groupby("dose")[f"hartmann_{label}"].mean().sort_index()
        axis.plot(curve.index, curve.values, marker="o", label=label.capitalize())
        deltas[label] = float(curve.loc[args.max_dose] - curve.loc[0.0])
    axis.set_title(f"Seven-class specificity: Feature {chosen}")
    axis.set_xlabel("Steering dose x natural activation scale")
    axis.set_ylabel("Hartmann probability")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "feature3271_seven_class_curves.png", dpi=220)
    plt.close(fig)
    pd.DataFrame([deltas], index=[chosen]).to_csv(
        args.output_dir / "feature3271_deltas.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--ranking_csv", type=Path, required=True)
    parser.add_argument("--probe_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--selection_count", type=int, default=3)
    parser.add_argument("--max_dose", type=float, default=3.0)
    main(parser.parse_args())
