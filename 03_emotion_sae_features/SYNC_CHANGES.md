# Synchronisation with emotion_feature_handoff.zip

## Inputs used

Only the two archives supplied with this modification request were used.

| Role | Archive | SHA-256 |
| --- | --- | --- |
| Project to edit | `Emotion-linked-threshold-geometry-main.zip` | `67e941e5cbbcae27914e49b535d6ae878a4f7371253ddde8b90b27fb30057c78` |
| Reference source | `emotion_feature_handoff.zip` | `846aeaba20852587ef91cfcdaa15b0275665f024fbabba751c7e4e33ab20f028` |

Source paths below are relative to `gemma_emotion_feature_replication_handoff_20260901/` inside the second archive. Destination paths are relative to `03_emotion_sae_features/`.

## Modified existing files

| Destination | Before | After / source |
| --- | --- | --- |
| `code/validation/analyze_joy_validation.py` | Forced five-dose `isin` filter, optional probe CSV, maximum dose 2, renamed endpoint column | Exact copy of the second archive's same-named script: `dose <= args.max_dose`, required `--probe_csv`, maximum dose 3, original `dose3_joy_probability` column; all statistics and thresholds unchanged from the source |
| `code/validation/analyze_anger_validation.py` | Maximum dose 2 and renamed endpoint column | Exact copy of the second archive's same-named script: maximum dose 3 and original `dose3_anger_probability` column; original calculations retained |
| `code/validation/validate_generation.py` | Five-dose fallback | Fallback restored to the source `validate_generation_qwen_reference.py`: `[0, 0.25, 0.5, 1, 2, 3, 4, 5]`; existing project runtime interface retained |
| `config/qwen3_32b_reference.json` | No validation-specific chat-template flag | Added `validation_chat_template_kwargs: {enable_thinking: false}` to match the source validation call; behavioural chat settings unchanged |
| `README.md` | Five-dose reanalysis and optional-probe instructions | Complete ranking, generation and analysis commands following the second archive, with required joy probe input and explicit eight-dose reference grid |

The source generation script's exploratory fallback grid is not the recorded reference experiment grid. The handoff's `config/qwen_reference.json` and its scored-generation CSVs specify `0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3`; the README passes this grid explicitly with `--doses`. No fixed five-dose filter is applied. The TPP behaviour grid remains `0, 0.5, 1, 1.5, 2`.

## Added files copied unchanged from the second archive

| Destination | Source |
| --- | --- |
| `code/screening/rank_ed_broad_anger.py` | `code/screening/rank_ed_broad_anger.py` |
| `config/qwen_reference.json` | `config/qwen_reference.json` |
| `reference/anger_candidate_rankings.csv` | `results/qwen_reference/screening/anger_candidate_rankings.csv` |
| `reference/anger_all_candidates_scored_generations.csv` | `results/qwen_reference/validation/anger_all_candidates_scored_generations.csv` |
| `reference/anger_candidate_summary.csv` | `results/qwen_reference/validation/anger_candidate_summary.csv` |
| `reference/joy_candidate_summary.csv` | `results/qwen_reference/validation/joy_candidate_summary.csv` |

This `SYNC_CHANGES.md` is the only additional modification record. The joy exact ranker, joy ranking/probe/generation CSVs, and both ranking JSONs already existed in the newly supplied project and were verified identical to the second archive; their contents have not changed. Corpora, neutral words, the shared ranker's statistical calculations, and the reusable delta-steering code are preserved.

## Files deliberately preserved

All 42 files outside `03_emotion_sae_features/` retain their exact original bytes. Within Part 03, `code/behavior/sae_tpp_runner.py`, `run_behavior_steering.py` and `prepare_inputs.py` are unchanged, including `None` handling, delta steering, the behavioural multiplier grid and seed. No private participant prompt files were copied from the handoff. Existing profile-based loading and checkpoint checks remain in place.

## Verification

- Python syntax: all 54 project Python files passed, including 27 within Part 03.
- Command interfaces: eight Part 03 `--help` checks passed.
- Exact-source comparison: both validation-analysis scripts and both exact Qwen ranking scripts match the second archive byte for byte.
- Reference records: both emotions contain 8 candidates x 8 doses x 80 cue words = 5,120 scored records, and the 80 words match the first 80 supplied neutral cue words.
- Reanalysis: the full original records reproduce every column of both archived candidate-summary CSVs within numerical tolerance (`rtol=1e-10`, `atol=1e-12`).
- Original validation status reproduced: anger Feature 662 has Spearman rho 0.9523809523809524 and `validated`; joy Feature 3271 has rho 0.880952380952381 and `validated`.

These checks reanalyse saved classifier outputs. They do not rerun the LLM, SAE, classifiers or TPP experiment; no model weights were loaded and no API calls were made. There is no new claim of bit-identical model generations on different hardware.
