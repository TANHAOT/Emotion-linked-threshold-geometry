#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "project_core").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Could not locate the project root containing project_core/.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_core.data import load_human_data, write_subject_inputs


def main():
    p = argparse.ArgumentParser(description="Generate character/game JSON inputs for local SAE runners.")
    p.add_argument("--subjects", nargs=2, type=int, metavar=("START", "END"), default=[0, 1017], help="Half-open subject range [START, END).")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "generated_inputs")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = p.parse_args()
    demographics, _ = load_human_data(args.data_dir)
    start, end = args.subjects
    if not 0 <= start < end <= len(demographics):
        p.error(f"Valid range is within [0, {len(demographics)}]")
    write_subject_inputs(args.data_dir, args.output_dir, range(start, end))
    print(f"Wrote subjects {start}..{end-1} to {args.output_dir}")

if __name__ == "__main__":
    main()
