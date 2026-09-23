from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

# Allow this module to be imported or executed from any working directory.
PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "project_core").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate the project root containing project_core/.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_core.data import character_to_prompt, subject_character, subject_trials
from project_core.models import token_limit
from project_core.prompts import compose_user_prompt, gpt35_system_prompt

PATTERNS = {
    "emotion": re.compile(r"(AA_valence|AA_arousal|choice|AC_valence|AC_arousal)\s*=\s*(-?\d+)", re.I),
    "no_report": re.compile(r"(choice)\s*=\s*(\d+)", re.I),
    "unfairness": re.compile(r"(AA_unfairness|choice|AC_unfairness)\s*=\s*(-?\d+)", re.I),
    "intentionality": re.compile(r"(AA_intentionality|choice|AC_intentionality)\s*=\s*(-?\d+)", re.I),
}

EXPECTED = {
    "emotion": ("AA_valence", "AA_arousal", "choice", "AC_valence", "AC_arousal"),
    "no_report": ("choice",),
    "unfairness": ("AA_unfairness", "choice", "AC_unfairness"),
    "intentionality": ("AA_intentionality", "choice", "AC_intentionality"),
}

BASE_COLUMNS = [
    "subject_index", "id", "trial", "amount_of_allocation", "cost_level", "amount_of_cost",
]

RESULT_COLUMNS = {
    "emotion": [
        "AA_valence", "AA_arousal", "choice", "AC_valence", "AC_arousal",
        "EmoFDBK_valence", "EmoFDBK_arousal", "valid", "Output",
    ],
    "no_report": ["choice", "valid", "Output"],
    "unfairness": [
        "AA_unfairness", "choice", "AC_unfairness", "UnfairnessFDBK", "valid", "Output",
    ],
    "intentionality": [
        "AA_intentionality", "choice", "AC_intentionality", "IntentionalityFDBK", "valid", "Output",
    ],
}


def parse_output(text: str, condition: str) -> dict[str, Any]:
    values: dict[str, Any] = {key: None for key in EXPECTED[condition]}
    for key, raw in PATTERNS[condition].findall(text or ""):
        canonical = next(name for name in EXPECTED[condition] if name.lower() == key.lower())
        values[canonical] = int(raw)

    choice = values.get("choice")
    valid = choice in (0, 1)
    for key, value in values.items():
        if key != "choice":
            valid = valid and isinstance(value, int) and -100 <= value <= 100

    if condition == "emotion":
        values["EmoFDBK_valence"] = None if values["AA_valence"] is None or values["AC_valence"] is None else values["AC_valence"] - values["AA_valence"]
        values["EmoFDBK_arousal"] = None if values["AA_arousal"] is None or values["AC_arousal"] is None else values["AC_arousal"] - values["AA_arousal"]
    elif condition == "unfairness":
        values["UnfairnessFDBK"] = None if values["AA_unfairness"] is None or values["AC_unfairness"] is None else values["AC_unfairness"] - values["AA_unfairness"]
    elif condition == "intentionality":
        values["IntentionalityFDBK"] = None if values["AA_intentionality"] is None or values["AC_intentionality"] is None else values["AC_intentionality"] - values["AA_intentionality"]
    values["valid"] = bool(valid)
    values["Output"] = (text or "").strip().replace("\n", " ")
    return values


def _output_path(
    output_root: Path, model: str, condition: str, subject_index: int,
    temperature: float, use_persona: bool,
) -> Path:
    persona_label = "persona" if use_persona else "nopersona"
    temperature_label = f"{float(temperature):.1f}"
    out_dir = output_root / f"{persona_label}_{condition}_{temperature_label}_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"output_{subject_index}.tsv"


def _ensure_output_file(path: Path, condition: str) -> None:
    """Create the TSV and header before the first API request."""
    if path.exists() and path.stat().st_size > 0:
        return
    fieldnames = BASE_COLUMNS + RESULT_COLUMNS[condition]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def _completed_trials(path: Path) -> set[int]:
    """Return trial numbers already safely written to disk."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        frame = pd.read_csv(path, sep="\t", usecols=["trial"])
    except (pd.errors.EmptyDataError, ValueError):
        return set()
    return {int(value) for value in frame["trial"].dropna().tolist()}


def _append_row(path: Path, condition: str, row: dict[str, Any]) -> None:
    """Append one completed trial and force it to disk immediately."""
    fieldnames = BASE_COLUMNS + RESULT_COLUMNS[condition]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _retryable_exception(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        # Connection errors and timeouts generally have no HTTP status code.
        return True
    try:
        status = int(status)
    except (TypeError, ValueError):
        return True
    return status in {408, 409, 429} or status >= 500


def _request_with_retry(
    client: OpenAI, *, model: str, system_prompt: str, prompt: str,
    temperature: float, max_retries: int, retry_backoff_seconds: float,
):
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max(128, token_limit(model) // 2),
            )
        except Exception as exc:
            if attempt >= max_retries or not _retryable_exception(exc):
                raise
            delay = min(retry_backoff_seconds * (2 ** attempt), 60.0)
            print(
                f"RETRY model={model} attempt={attempt + 1}/{max_retries} "
                f"after {delay:g}s: {exc}"
            )
            time.sleep(delay)
    raise RuntimeError("Retry loop exited unexpectedly.")


def run_subject(
    *, model: str, condition: str, subject_index: int, temperature: float, use_persona: bool,
    demographics: pd.DataFrame, trials: pd.DataFrame, output_root: Path,
    trials_per_subject: int, base_url: str, api_key: str, timeout_seconds: float = 600.0,
    resume: bool = True, max_retries: int = 5, retry_backoff_seconds: float = 2.0,
) -> Path:
    character = subject_character(demographics, subject_index)
    persona_text = character_to_prompt(character)
    settings = subject_trials(trials, subject_index)[:trials_per_subject]
    system_prompt = gpt35_system_prompt(condition) if model == "gpt-3.5-turbo-0125" else ""

    out_path = _output_path(
        output_root, model, condition, subject_index, temperature, use_persona
    )
    completed = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            saved = pd.read_csv(out_path, sep="\t", usecols=["trial", "valid"])
            has_error = (
                not saved["valid"].astype(str).str.strip().str.lower().eq("true").all()
                or saved["trial"].isna().any()
                or saved["trial"].duplicated().any()
                or set(saved["trial"]) != {int(setting["trial"]) for setting in settings}
            )
            if not resume or has_error:
                out_path.unlink()
            else:
                completed = set(saved["trial"].astype(int))
        except (pd.errors.EmptyDataError, ValueError):
            out_path.unlink()
    _ensure_output_file(out_path, condition)

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)

    for round_index, setting in enumerate(settings):
        trial_number = int(setting["trial"])
        if trial_number in completed:
            continue

        prompt = compose_user_prompt(
            persona_text, setting, round_index, condition, use_persona=use_persona
        )
        response = _request_with_retry(
            client,
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        text = response.choices[0].message.content or ""
        parsed = parse_output(text, condition)
        row = {
            "subject_index": subject_index,
            "id": setting["id"],
            "trial": setting["trial"],
            "amount_of_allocation": setting["amount_of_allocation"],
            "cost_level": setting["cost_level"],
            "amount_of_cost": setting["amount_of_cost"],
            **parsed,
        }
        _append_row(out_path, condition, row)
        print(
            f"SAVED model={model} condition={condition} subject={subject_index} "
            f"trial={trial_number} -> {out_path}"
        )

    return out_path
