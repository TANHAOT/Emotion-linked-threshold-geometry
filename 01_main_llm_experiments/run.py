#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "project_core").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Could not locate the project root containing project_core/.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise RuntimeError(
        "python-dotenv is required. Install requirements-api.txt first."
    ) from exc

load_dotenv(ROOT / ".env", override=False)

from config import FORMAL_EXPERIMENTS, FORMAL_RUN, RUN_OPTIONS, TEST, TEST_CONFIG
from project_core.data import load_human_data

VALID_CONDITIONS = {"emotion", "no_report", "unfairness", "intentionality"}


def parse_subjects(spec: str | int | list[int] | tuple[int, ...], n_total: int) -> list[int]:
    if isinstance(spec, int):
        values = [spec]
    elif isinstance(spec, (list, tuple)):
        values = [int(value) for value in spec]
    else:
        text = str(spec).strip().lower()
        if text == "all":
            return list(range(n_total))
        values: list[int] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-", 1))
                if end < start:
                    raise ValueError(f"Invalid subject range {part!r}: end < start")
                values.extend(range(start, end + 1))
            else:
                values.append(int(part))

    values = sorted(set(values))
    if not values:
        raise ValueError("No subjects were selected.")
    if values[0] < 0 or values[-1] >= n_total:
        raise ValueError(f"Subjects must fall within 0-{n_total - 1}")
    return values


def compact(values: list[Any], limit: int = 20) -> str:
    text = [str(value) for value in values]
    if len(text) <= limit:
        return ", ".join(text)
    return ", ".join(text[:10]) + f", ... , {', '.join(text[-3:])}"


def validate_config(name: str, config: dict[str, Any]) -> None:
    required = {
        "models", "conditions", "subjects", "trials_per_subject",
        "use_persona", "temperature", "workers",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration {name!r} is missing: {', '.join(missing)}")
    if not config["models"]:
        raise ValueError(f"Configuration {name!r} has no models.")
    if not config["conditions"]:
        raise ValueError(f"Configuration {name!r} has no conditions.")
    invalid = sorted(set(config["conditions"]) - VALID_CONDITIONS)
    if invalid:
        raise ValueError(f"Configuration {name!r} has invalid conditions: {', '.join(invalid)}")
    if not 1 <= int(config["trials_per_subject"]) <= 60:
        raise ValueError(f"Configuration {name!r}: trials_per_subject must be 1-60.")
    if int(config["workers"]) < 1:
        raise ValueError(f"Configuration {name!r}: workers must be >= 1.")


def print_run_plan(
    *, mode: str, block_name: str, config: dict[str, Any], subjects: list[int],
    human_ids: list[Any], data_dir: Path, output_dir: Path, base_url: str, dry_run: bool,
) -> None:
    models = list(config["models"])
    conditions = list(config["conditions"])
    trials_per_subject = int(config["trials_per_subject"])
    jobs = len(models) * len(conditions) * len(subjects)
    api_calls = jobs * trials_per_subject

    print("\n" + "=" * 78)
    print("01 MAIN LLM EXPERIMENT — RUN CONFIGURATION")
    print("=" * 78)
    print(f"Mode                 : {mode}")
    print(f"Config block         : {block_name}")
    print(f"Models ({len(models)})           : {', '.join(models)}")
    print(f"Conditions ({len(conditions)})       : {', '.join(conditions)}")
    print(f"Subject indices ({len(subjects)}): {compact(subjects)}")
    print(f"Human IDs ({len(human_ids)})        : {compact(human_ids)}")
    print(f"Trials per subject   : {trials_per_subject}")
    print(f"Persona              : {'yes' if config['use_persona'] else 'no'}")
    print(f"Temperature          : {float(config['temperature']):g}")
    print(f"Parallel workers     : {int(config['workers'])}")
    print(f"Subject-level jobs   : {jobs}")
    print(f"Expected API calls   : {api_calls}")
    print(f"Resume existing files: {'yes' if RUN_OPTIONS['resume'] else 'no'}")
    print(f"Max API retries      : {int(RUN_OPTIONS['max_retries'])}")
    print(f"Retry backoff (sec)  : {float(RUN_OPTIONS['retry_backoff_seconds']):g}")
    print(f"Base URL             : {base_url}")
    print(f"Data directory       : {data_dir}")
    persona_label = "persona" if config["use_persona"] else "nopersona"
    temperature_label = f"{float(config['temperature']):.1f}"
    print(f"Output folder pattern: {output_dir / f'{persona_label}_<condition>_{temperature_label}_<model>'}")
    print(f"Dry run              : {'yes' if dry_run else 'no'}")
    print("=" * 78 + "\n")


def execute_block(
    *, mode: str, block_name: str, config: dict[str, Any], demographics,
    trials, data_dir: Path, output_dir: Path, base_url: str, api_key: str,
    dry_run: bool,
) -> None:
    validate_config(block_name, config)
    subjects = parse_subjects(config["subjects"], len(demographics))
    human_ids = demographics.iloc[subjects]["id"].tolist()

    if dry_run:
        return

    from runner import run_subject

    tasks = [
        (model, condition, subject)
        for model in config["models"]
        for condition in config["conditions"]
        for subject in subjects
    ]
    failures: list[tuple[tuple[str, str, int], str]] = []
    with ThreadPoolExecutor(max_workers=int(config["workers"])) as pool:
        future_map = {
            pool.submit(
                run_subject,
                model=model,
                condition=condition,
                subject_index=subject,
                temperature=float(config["temperature"]),
                use_persona=bool(config["use_persona"]),
                demographics=demographics,
                trials=trials,
                output_root=output_dir,
                trials_per_subject=int(config["trials_per_subject"]),
                base_url=base_url,
                api_key=api_key,
                resume=bool(RUN_OPTIONS["resume"]),
                max_retries=int(RUN_OPTIONS["max_retries"]),
                retry_backoff_seconds=float(RUN_OPTIONS["retry_backoff_seconds"]),
            ): (model, condition, subject)
            for model, condition, subject in tasks
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                print(f"DONE {key}: {future.result()}")
            except Exception as exc:
                failures.append((key, repr(exc)))
                print(f"FAILED {key}: {exc}")

    if failures:
        print(f"\n{block_name}: failures")
        for key, error in failures:
            print(key, error)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Part 01 using the explicit TEST / formal dictionaries in "
            "01_main_llm_experiments/config.py."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print and validate the selected configuration without API requests."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Temporarily force TEST mode without editing config.py."
    )
    parser.add_argument(
        "--formal", action="store_true",
        help="Temporarily force formal mode without editing config.py."
    )
    parser.add_argument(
        "--experiment", action="append", choices=tuple(FORMAL_EXPERIMENTS),
        help="In formal mode, run only this named block. Repeat for multiple blocks."
    )
    args = parser.parse_args()

    if args.test and args.formal:
        parser.error("Use only one of --test or --formal.")

    is_test = TEST
    if args.test:
        is_test = True
    elif args.formal:
        is_test = False

    data_dir = ROOT / "data"
    output_dir = Path(__file__).resolve().parent / "result"
    base_url = os.environ.get("LLM_BASE_URL", "https://api.midsummer.work/v1").strip()
    demographics, trials = load_human_data(data_dir)

    if is_test:
        selected = [("TEST_CONFIG", TEST_CONFIG)]
        mode = "TEST"
    else:
        names = args.experiment or FORMAL_RUN
        unknown = [name for name in names if name not in FORMAL_EXPERIMENTS]
        if unknown:
            parser.error(f"Unknown formal experiment(s): {', '.join(unknown)}")
        selected = [(name, FORMAL_EXPERIMENTS[name]) for name in names]
        mode = "FORMAL"

    # Validate and print every block before any API call starts.
    for block_name, config in selected:
        validate_config(block_name, config)
        subjects = parse_subjects(config["subjects"], len(demographics))
        human_ids = demographics.iloc[subjects]["id"].tolist()
        print_run_plan(
            mode=mode,
            block_name=block_name,
            config=config,
            subjects=subjects,
            human_ids=human_ids,
            data_dir=data_dir,
            output_dir=output_dir,
            base_url=base_url,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("Dry run complete. No API requests were sent.")
        return

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    placeholder_keys = {"your_api_key_here", "put_your_api_key_here", "replace_me"}
    if not api_key or api_key.lower() in placeholder_keys:
        parser.error("Set a real LLM_API_KEY in the project-level .env file before API execution.")

    for block_name, config in selected:
        execute_block(
            mode=mode,
            block_name=block_name,
            config=config,
            demographics=demographics,
            trials=trials,
            data_dir=data_dir,
            output_dir=output_dir,
            base_url=base_url,
            api_key=api_key,
            dry_run=False,
        )


if __name__ == "__main__":
    main()
