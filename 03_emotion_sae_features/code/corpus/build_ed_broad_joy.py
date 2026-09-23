#!/usr/bin/env python3
"""Build an EmpatheticDialogues-only broad-joy corpus matched to the anger design."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


POSITIVE_LABELS = ("joyful", "excited", "content")
CONTROL_LABELS = (
    "angry", "afraid", "sad", "disappointed", "guilty", "caring", "grateful"
)
EXPLICIT = re.compile(
    r"\b(joy|joyful|happy|happiness|excited|excitement|content|contented|"
    r"glad|delighted|cheerful|thrilled)\b",
    re.I,
)


def stable_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_key(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clean(text):
    for old, new in {
        "_comma_": ",", "_period_": ".", "_pipe_": "|", "[NAME]": "someone"
    }.items():
        text = text.replace(old, new)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = " ".join(text.split()).strip(" \t\r\n\"'")
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not 6 <= len(words) <= 80:
        return None
    alpha_ratio = sum(char.isalpha() or char.isspace() for char in text) / max(len(text), 1)
    return text if alpha_ratio >= 0.62 else None


def select(records, limit):
    unique = {}
    for record in records:
        unique.setdefault(normalize_key(record["text"]), record)
    return sorted(unique.values(), key=lambda row: stable_key(row["text"]))[:limit]


def read_records(source_dir):
    records = []
    for source_split, split in (
        ("train", "discovery"), ("valid", "heldout"), ("test", "heldout")
    ):
        conversations = {}
        with (source_dir / f"{source_split}.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                conversations.setdefault(row["conv_id"], row)
        for row in conversations.values():
            text = clean(row["prompt"])
            if text is None:
                continue
            records.append({
                "text": text,
                "source": "EmpatheticDialogues",
                "source_split": source_split,
                "split": split,
                "source_id": row["conv_id"],
                "source_label": row["context"],
            })
    return records


def build(args):
    records = read_records(args.source_dir)
    output = []
    counts = {}
    caps = {
        "discovery": args.discovery_positive_per_label,
        "heldout": args.heldout_positive_per_label,
    }
    for split in ("discovery", "heldout"):
        by_label = defaultdict(list)
        for row in records:
            if row["split"] == split:
                by_label[row["source_label"]].append(row)

        positive_quota = min(
            caps[split], *(len(by_label[label]) for label in POSITIVE_LABELS)
        )
        positives = []
        for label in POSITIVE_LABELS:
            positives.extend(select(by_label[label], positive_quota))

        control_quota = math.ceil(len(positives) / len(CONTROL_LABELS))
        controls = []
        for label in CONTROL_LABELS:
            controls.extend(select(by_label[label], control_quota))

        for label, selected in ((1, positives), (0, controls)):
            for row in selected:
                item = dict(row)
                item["labels"] = {"joy": label}
                item["joy_kind"] = (
                    "explicit" if label == 1 and EXPLICIT.search(item["text"]) else "implicit"
                )
                item["joy_subtype"] = item["source_label"] if label == 1 else "control"
                output.append(item)
        counts[split] = {
            "positive": len(positives),
            "control": len(controls),
            "positive_per_label": dict(Counter(row["source_label"] for row in positives)),
            "control_per_label": dict(Counter(row["source_label"] for row in controls)),
            "implicit_positive": sum(not EXPLICIT.search(row["text"]) for row in positives),
        }

    deduplicated = {}
    for row in output:
        deduplicated.setdefault(normalize_key(row["text"]), row)
    rows = sorted(deduplicated.values(), key=lambda row: stable_key(row["text"]))
    for row_id, row in enumerate(rows):
        row["row_id"] = row_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "rows": len(rows),
        "source": "EmpatheticDialogues only",
        "positive_definition": list(POSITIVE_LABELS),
        "control_definition": list(CONTROL_LABELS),
        "moral_keyword_filter": False,
        "counts": counts,
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovery_positive_per_label", type=int, default=500)
    parser.add_argument("--heldout_positive_per_label", type=int, default=150)
    build(parser.parse_args())
