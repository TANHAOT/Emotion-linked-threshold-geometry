#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "project_core").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Could not locate the project root containing project_core/.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_core.data import load_human_data, write_subject_inputs
from project_core.prompts import GAME_DESCRIPTION

def main():
    p=argparse.ArgumentParser(); p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int,default=1017)
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'inputs'); p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    a=p.parse_args(); d,_=load_human_data(a.data_dir)
    if not 0 <= a.start < a.end <= len(d): p.error(f'valid range: 0..{len(d)}')
    write_subject_inputs(a.data_dir,a.output_dir,range(a.start,a.end))
    (a.output_dir/'person_all_game_prompt.json').write_text(json.dumps({'10':['ours','',GAME_DESCRIPTION]},indent=2),encoding='utf-8')
if __name__=='__main__': main()
