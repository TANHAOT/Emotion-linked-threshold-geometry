# 03 — Emotion-related SAE feature identification and targeted steering

This folder implements emotion corpus -> SAE activation extraction -> candidate ranking -> generation validation -> TPP steering. The Qwen reference workflow below follows `emotion_feature_handoff.zip`. Model loading continues to use the current project's JSON SAE profiles.

## Reference settings

`config/qwen3_32b_reference.json` is the runtime profile for Qwen3-32B, layer 48, Trainer 1. `config/qwen_reference.json` is an unchanged copy of the handoff's experiment settings; it is a record of the experiment, not an alternative runtime profile.

| Stage | Reference settings |
| --- | --- |
| Corpus screening | Supplied EmpatheticDialogues anger and joy corpora; content-token maximum; sink ratio 10 |
| Generation validation | Additive steering; deterministic generation; Qwen `enable_thinking=False` |
| Recorded validation grid | 0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3 |
| Recorded validation sample | 8 candidates per emotion; 80 cue words per candidate and dose |
| Validation analysis | Original handoff scripts; `dose <= max_dose`; `max_dose=3.0` |
| TPP intervention | Multiplicative delta steering; 0, 0.5, 1, 1.5, 2; seed 20260714 |

The eight-dose grid above is specified in the handoff's `config/qwen_reference.json` and present in both reference generation CSVs. Its generation script has a separate exploratory fallback grid of `0, 0.25, 0.5, 1, 2, 3, 4, 5`. That fallback is preserved in `validate_generation.py`; the commands below explicitly pass the recorded eight-dose grid. Do not substitute a fixed five-dose filter in either analysis script.

The reference feature IDs and activation scales belong only to the supplied model/SAE checkpoint. Do not transfer them to another model, layer, trainer or dictionary.

## 1. Extract emotion-corpus activations

Run all commands from the project root.

```bash
python 03_emotion_sae_features/code/screening/extract_activations.py \
  --sae_config 03_emotion_sae_features/config/qwen3_32b_reference.json \
  --dataset 03_emotion_sae_features/data/empathetic_dialogues/ed_broad_anger.jsonl \
  --output 03_emotion_sae_features/results/anger_activations.npz

python 03_emotion_sae_features/code/screening/extract_activations.py \
  --sae_config 03_emotion_sae_features/config/qwen3_32b_reference.json \
  --dataset 03_emotion_sae_features/data/empathetic_dialogues/ed_broad_joy.jsonl \
  --output 03_emotion_sae_features/results/joy_activations.npz
```

Each NPZ contains `row_id`, `max` and `filtered_tokens`, with a `.metadata.json` sidecar identifying the model/SAE. Model and SAE weights are not included.

## 2. Reproduce the Qwen candidate ranking

These two scripts are unchanged copies of the exact ranking scripts in the handoff. The model-independent `rank_emotion_features.py` remains available for other model/SAE pairs.

```bash
python 03_emotion_sae_features/code/screening/rank_ed_broad_anger.py \
  --dataset 03_emotion_sae_features/data/empathetic_dialogues/ed_broad_anger.jsonl \
  --activations 03_emotion_sae_features/results/anger_activations.npz \
  --output_dir 03_emotion_sae_features/results/anger_screening

python 03_emotion_sae_features/code/screening/rank_ed_broad_joy.py \
  --dataset 03_emotion_sae_features/data/empathetic_dialogues/ed_broad_joy.jsonl \
  --activations 03_emotion_sae_features/results/joy_activations.npz \
  --output_dir 03_emotion_sae_features/results/joy_screening \
  --probe_features 3271
```

Both scripts write `candidate_rankings.csv` and `candidate_rankings.json`. The joy script also writes `probe_features.csv`; this is the `--probe_csv` input required by the unchanged joy analysis script. Its merge ignores probe IDs already present in the ranking table.

## 3. Generate and score the reference candidate sentences

The following candidates and word counts reproduce the coverage of the supplied reference generation CSVs. They are historical Qwen candidates, not a selection rule for another SAE. Generation uses the same neutral cue words and additive steering as the handoff. The GoEmotions classifier scores and seven-class Hartmann scores are both retained; the candidate-analysis scripts use the `hartmann_*` columns.

```bash
python 03_emotion_sae_features/code/validation/validate_generation.py \
  --sae_config 03_emotion_sae_features/config/qwen3_32b_reference.json \
  --dataset 03_emotion_sae_features/data/empathetic_dialogues/ed_broad_anger.jsonl \
  --activations 03_emotion_sae_features/results/anger_activations.npz \
  --features anger:662 anger:1353 anger:3336 anger:5307 anger:10325 anger:10888 anger:12077 anger:14902 \
  --words-file 03_emotion_sae_features/data/neutral_validation_words.txt --words 80 \
  --doses 0 0.25 0.5 1 1.5 2 2.5 3 \
  --output-dir 03_emotion_sae_features/results/anger_validation

python 03_emotion_sae_features/code/validation/validate_generation.py \
  --sae_config 03_emotion_sae_features/config/qwen3_32b_reference.json \
  --dataset 03_emotion_sae_features/data/empathetic_dialogues/ed_broad_joy.jsonl \
  --activations 03_emotion_sae_features/results/joy_activations.npz \
  --features joy:3271 joy:6718 joy:7435 joy:7565 joy:9355 joy:13121 joy:14013 joy:14456 \
  --words-file 03_emotion_sae_features/data/neutral_validation_words.txt --words 80 \
  --doses 0 0.25 0.5 1 1.5 2 2.5 3 \
  --output-dir 03_emotion_sae_features/results/joy_validation
```

The existing classifier defaults are local directories: `~/models/roberta-base-go_emotions` and `~/models/emotion-english-distilroberta-base`. Set `--classifier-dir` and `--secondary-classifier-dir` to their installed locations. Model, SAE and classifier weights are not included in this package.

## 4. Analyse generation validation

Both analysis scripts preserve the handoff's statistics, thresholds, required arguments and output column names, including `dose3_anger_probability` and `dose3_joy_probability`. Their default `max_dose` is 3.0. Dose 0.25, 1.5 and 2.5 are included when present in the input; there is no fixed-grid exclusion.

### Reanalyse the supplied reference records without loading an LLM

All reference CSV and JSON files below are copied unchanged from the handoff.

```bash
python 03_emotion_sae_features/code/validation/analyze_anger_validation.py \
  --input_csv 03_emotion_sae_features/reference/anger_all_candidates_scored_generations.csv \
  --ranking_csv 03_emotion_sae_features/reference/anger_candidate_rankings.csv \
  --output_dir 03_emotion_sae_features/results/anger_validation_analysis \
  --max_dose 3

python 03_emotion_sae_features/code/validation/analyze_joy_validation.py \
  --input_csv 03_emotion_sae_features/reference/joy_all_candidates_scored_generations.csv \
  --ranking_csv 03_emotion_sae_features/reference/joy_candidate_rankings.csv \
  --probe_csv 03_emotion_sae_features/reference/joy_probe_features.csv \
  --output_dir 03_emotion_sae_features/results/joy_validation_analysis \
  --max_dose 3
```

The archived `reference/anger_candidate_summary.csv` and `reference/joy_candidate_summary.csv` are the original handoff summaries, not five-dose recomputations. The supplied full eight-dose records reproduce those summaries under the original analysis scripts.

### Analyse newly generated records

Use the same commands with these path substitutions:

| Argument | Anger | Joy |
| --- | --- | --- |
| `--input_csv` | `03_emotion_sae_features/results/anger_validation/steering_sentences_scored.csv` | `03_emotion_sae_features/results/joy_validation/steering_sentences_scored.csv` |
| `--ranking_csv` | `03_emotion_sae_features/results/anger_screening/candidate_rankings.csv` | `03_emotion_sae_features/results/joy_screening/candidate_rankings.csv` |
| `--probe_csv` | Not used | `03_emotion_sae_features/results/joy_screening/probe_features.csv` |

Keep generated scores, ranking metrics and probes from the same model/SAE and candidate set. Do not replace newly computed summaries with historical results.

## 5. Run the matched TPP intervention

The behavioural multiplier grid is separate from the generation-validation dose grid and remains `0, 0.5, 1, 1.5, 2`. The current runner's delta steering, missing-value handling, output paths and participant execution remain unchanged.

Generate inputs from authorised participant data:

```bash
python 03_emotion_sae_features/prepare_inputs.py --start 0 --end 1017
```

The public demographic workbook is a redacted template. Supply authorised complete profiles before reproducing persona conditions. The handoff's private participant prompt files have not been copied back into this project.

```bash
python 03_emotion_sae_features/run_behavior_steering.py \
  --emotion anger --feature-id 662 \
  --multipliers 0 0.5 1 1.5 2 --seed 20260714

python 03_emotion_sae_features/run_behavior_steering.py \
  --emotion joy --feature-id 3271 \
  --multipliers 0 0.5 1 1.5 2 --seed 20260714
```

Add `--no-report` for the no-report condition. Preserve the existing model/SAE profile across extraction, validation and behaviour. For a new model/SAE pair, use `config/custom_sae_template.json` and rediscover feature IDs rather than reusing the Qwen IDs.

See `SYNC_CHANGES.md` for the exact source files, parameter changes and verification results.
