#!/usr/bin/env python3
"""Dose-response generation validation for features from any compatible SAE profile."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

SECTION_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "project_core").is_dir()), None)
if REPO_ROOT is None:
    raise RuntimeError("Could not locate the project root containing project_core/.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from project_core.sae_runtime import (
    add_sae_override_arguments, apply_sae_overrides, describe_profile,
    load_sae_profile, load_sae_runtime, render_chat,
)

WORDS = [
    "notebook", "window", "bottle", "garden", "ticket", "pencil", "curtain", "bridge",
    "basket", "mirror", "camera", "folder", "kettle", "button", "market", "station",
    "blanket", "calendar", "suitcase", "lantern", "river", "library", "coffee", "bicycle",
    "letter", "kitchen", "painting", "doctor", "teacher", "neighbor", "phone", "table",
    "clock", "train", "paper", "street", "office", "shelf", "jacket", "computer",
    "umbrella", "museum", "pocket", "sandwich", "elevator", "backpack", "newspaper", "harbor",
    "cabinet", "staircase", "restaurant", "photograph", "passport", "bookstore", "airport", "radio",
    "ceiling", "sidewalk", "warehouse", "fountain", "envelope", "cupboard", "telescope", "keyboard",
]


def clean_sentence(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return " ".join(text.split())


def parse_feature(value):
    emotion, feature = value.split(":", 1)
    return emotion, int(feature)


def main(args):
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    profile = apply_sae_overrides(load_sae_profile(args.sae_config), args)
    print("SAE_CONFIG", describe_profile(profile))
    runtime = load_sae_runtime(profile, SECTION_DIR / "code" / "vendor")
    tokenizer, model, sae, submodule = runtime.tokenizer, runtime.model, runtime.sae, runtime.submodule
    tokenizer.padding_side = "left"

    candidates = [parse_feature(value) for value in args.features]
    feature_count = int(sae.W_enc.shape[-1])
    invalid = [feature_id for _, feature_id in candidates if not 0 <= feature_id < feature_count]
    if invalid:
        raise ValueError(f"Feature IDs outside [0, {feature_count}): {invalid}")

    corpus = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata_path = args.activations.with_suffix(args.activations.suffix + ".metadata.json")
    activation_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if activation_metadata and not args.allow_mismatched_activations:
        expected = activation_metadata.get("profile_fingerprint")
        if expected and expected != runtime.fingerprint:
            raise ValueError(
                f"Activation/SAE mismatch: activation fingerprint={expected}, runtime={runtime.fingerprint}. "
                "Use the same SAE profile used for extraction, or pass --allow-mismatched-activations only if this is intentional."
            )
    activation_data = np.load(args.activations)
    order = np.argsort(activation_data["row_id"])
    row_ids = activation_data["row_id"][order]
    maximums = activation_data["max"][order].astype(np.float32)
    by_id = {row["row_id"]: row for row in corpus}
    metadata = [by_id[int(row_id)] for row_id in row_ids]
    scales = {}
    for emotion, feature_id in candidates:
        mask = np.asarray([
            row["split"] == "discovery" and row["labels"].get(emotion) == 1 for row in metadata
        ])
        values = maximums[mask, feature_id]
        values = values[values > 0]
        scales[(emotion, feature_id)] = float(np.median(values)) if len(values) else 1.0

    state = {"feature": None, "delta": 0.0}

    def steering_hook(module, inputs, output):
        del module, inputs
        if state["feature"] is None or state["delta"] == 0:
            return output
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        vector = sae.W_dec[state["feature"]].to(dtype=hidden.dtype, device=hidden.device)
        edited = hidden + torch.as_tensor(state["delta"], dtype=hidden.dtype, device=hidden.device) * vector
        return edited if isinstance(output, torch.Tensor) else (edited,) + output[1:]

    validation_words = WORDS
    if args.words_file is not None:
        validation_words = [line.strip() for line in args.words_file.read_text().splitlines() if line.strip()]
    validation_words = validation_words[:args.words]
    if len(validation_words) < args.words:
        raise ValueError(f"Requested {args.words} words but only found {len(validation_words)}")

    generation_cfg = dict(profile.get("generation", {}))
    validation_chat_kwargs = dict(generation_cfg.get("validation_chat_template_kwargs", {}))
    validation_generation_cfg = {**generation_cfg, "chat_template_kwargs": validation_chat_kwargs}
    prompts = []
    for word in validation_words:
        messages = [
            {"role": "system", "content": "Follow the user's instruction and write only one sentence."},
            {"role": "user", "content": f'Write one short, coherent sentence that naturally includes the word "{word}".'},
        ]
        prompts.append(render_chat(tokenizer, messages, validation_generation_cfg))

    handle = submodule.register_forward_hook(steering_hook)
    raw_rows = []
    started = time.time()
    try:
        with torch.inference_mode():
            for emotion, feature_id in candidates:
                scale = scales[(emotion, feature_id)]
                for dose in args.doses:
                    state.update(feature=feature_id, delta=float(dose) * scale)
                    outputs = []
                    dose_started = time.time()
                    for start in range(0, len(prompts), args.batch_size):
                        batch = prompts[start:start + args.batch_size]
                        encoded = tokenizer(batch, padding=True, return_tensors="pt").to(runtime.input_device)
                        generated = model.generate(
                            **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                        )
                        for index in range(len(batch)):
                            outputs.append(clean_sentence(tokenizer.decode(
                                generated[index, encoded.input_ids.shape[1]:], skip_special_tokens=True,
                            )))
                    elapsed = time.time() - dose_started
                    for word, sentence in zip(validation_words, outputs):
                        tokens = sentence.lower().split()
                        trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
                        repetition = 0.0 if not trigrams else 1 - len(set(trigrams)) / len(trigrams)
                        raw_rows.append({
                            "profile": runtime.profile_tag, "emotion": emotion, "feature_id": feature_id,
                            "natural_scale": scale, "dose": float(dose), "absolute_delta": float(dose) * scale,
                            "word": word, "sentence": sentence, "word_present": word.lower() in sentence.lower(),
                            "word_count": len(tokens), "trigram_repetition": repetition,
                            "generation_seconds_for_dose": elapsed,
                        })
                    print(
                        f"GENERATE profile={runtime.profile_tag} emotion={emotion} feature={feature_id} "
                        f"dose={dose:g} seconds={elapsed:.2f} total={time.time() - started:.1f}", flush=True,
                    )
    finally:
        handle.remove()

    del model, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    classifier_source = str(args.classifier_dir)
    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_source, local_files_only=args.classifier_local_only)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        classifier_source, local_files_only=args.classifier_local_only,
        torch_dtype=torch.float32, device_map="auto",
    ).eval()
    id2label = {int(key): value.lower() for key, value in classifier.config.id2label.items()}
    classifier_device = next(classifier.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(raw_rows), 128):
            batch = raw_rows[start:start + 128]
            encoded = classifier_tokenizer(
                [row["sentence"] for row in batch], padding=True, truncation=True,
                max_length=192, return_tensors="pt",
            ).to(classifier_device)
            probabilities = classifier(**encoded).logits.sigmoid().cpu()
            for row, scores in zip(batch, probabilities):
                for index, score in enumerate(scores.tolist()):
                    row[id2label[index]] = float(score)
                row["predicted_label"] = id2label[int(scores.argmax())]

    if args.secondary_classifier_dir is not None:
        source = str(args.secondary_classifier_dir)
        secondary_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=args.classifier_local_only)
        secondary = AutoModelForSequenceClassification.from_pretrained(
            source, local_files_only=args.classifier_local_only,
            torch_dtype=torch.float32, device_map="auto",
        ).eval()
        secondary_labels = {int(key): value.lower() for key, value in secondary.config.id2label.items()}
        secondary_device = next(secondary.parameters()).device
        with torch.inference_mode():
            for start in range(0, len(raw_rows), 128):
                batch = raw_rows[start:start + 128]
                encoded = secondary_tokenizer(
                    [row["sentence"] for row in batch], padding=True, truncation=True,
                    max_length=192, return_tensors="pt",
                ).to(secondary_device)
                probabilities = secondary(**encoded).logits.softmax(dim=-1).cpu()
                for row, scores in zip(batch, probabilities):
                    for index, score in enumerate(scores.tolist()):
                        row[f"hartmann_{secondary_labels[index]}"] = float(score)
                    row["hartmann_predicted_label"] = secondary_labels[int(scores.argmax())]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(raw_rows)
    frame.to_csv(args.output_dir / "steering_sentences_scored.csv", index=False)
    with (args.output_dir / "steering_sentences.jsonl").open("w", encoding="utf-8") as handle_out:
        for row in raw_rows:
            handle_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    import matplotlib.pyplot as plt
    emotion_order = list(dict.fromkeys(emotion for emotion, _ in candidates))
    columns = 2
    rows = (len(emotion_order) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4.8 * rows), squeeze=False)
    summary_rows = []
    for axis, emotion in zip(axes.flat, emotion_order):
        target_column = "caring" if emotion == "compassion" else emotion
        subset = frame[frame["emotion"] == emotion]
        for feature_id, group in subset.groupby("feature_id"):
            curve = group.groupby("dose")[target_column].agg(["mean", "sem"]).reset_index()
            axis.errorbar(curve["dose"], curve["mean"], yerr=1.96 * curve["sem"], marker="o", label=f"Feature {feature_id}")
            quality = group.groupby("dose").agg(
                word_present_rate=("word_present", "mean"), mean_word_count=("word_count", "mean"),
                mean_trigram_repetition=("trigram_repetition", "mean"),
            ).reset_index()
            for _, row in curve.iterrows():
                q = quality[quality["dose"] == row["dose"]].iloc[0]
                summary_rows.append({
                    "profile": runtime.profile_tag, "emotion": emotion, "feature_id": feature_id,
                    "dose": row["dose"], "target_probability_mean": row["mean"],
                    "target_probability_sem": row["sem"], **q.drop(labels="dose").to_dict(),
                })
        axis.set_title(emotion.capitalize())
        axis.set_xlabel("Steering dose × natural activation scale")
        axis.set_ylabel(f"{target_column.capitalize()} probability")
        axis.legend()
    for axis in axes.flat[len(emotion_order):]:
        axis.axis("off")
    fig.suptitle(f"{runtime.profile_tag}: single-feature semantic dose response (layer {runtime.layer})")
    fig.tight_layout()
    fig.savefig(args.output_dir / "steering_dose_curves.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "steering_summary.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--activations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--features", nargs="+", required=True, help="Emotion:feature pairs, e.g. anger:662 joy:3271. IDs are checkpoint-specific.")
    # Source-script fallback; the README passes the eight-dose reference grid explicitly.
    p.add_argument("--doses", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 3, 4, 5])
    p.add_argument("--words", type=int, default=48)
    p.add_argument("--words-file", type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--sae-config", "--sae_config", dest="sae_config", type=Path, default=SECTION_DIR / "config" / "qwen3_32b_reference.json")
    add_sae_override_arguments(p)
    p.add_argument("--classifier-dir", type=Path, default=Path.home() / "models/roberta-base-go_emotions")
    p.add_argument("--secondary-classifier-dir", type=Path, default=Path.home() / "models/emotion-english-distilroberta-base")
    p.add_argument("--classifier-local-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-mismatched-activations", action="store_true")
    main(p.parse_args())
