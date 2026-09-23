# LLM Emotion-Linked Threshold Project — Clean Reproducibility Package


## Part 01 configuration

Part 01 uses an explicit `TEST` switch and dictionaries in `01_main_llm_experiments/config.py`. With `TEST = True`, only `TEST_CONFIG` is executed; with `TEST = False`, the named blocks in `FORMAL_RUN` are taken from `FORMAL_EXPERIMENTS`. Use `python 01_main_llm_experiments/run.py --dry-run` to validate and print the exact model/condition/ID plan without sending API requests.

Part 01 outputs use a flat folder naming scheme directly under `01_main_llm_experiments/`, e.g. `result/persona_emotion_1.0_deepseek-v3/`. The naming rule is `result/<persona|nopersona>_<condition>_<temperature>_<model>`.

## Data

- `data/human_demographics.xlsx` — participant demographics and psychological measures (`n=1017`). Due to ethical requirements, participants' demographic information has been hidden. This file only provides the data structure and contains no real information.

- `data/human_experiment_trials.xlsx` — the 60 third-party-punishment trials per participant. Only experimental conditions are included. Complete results are available at https://osf.io/dh9g5.
 
## Three independent sections

### 01_main_llm_experiments
API-based LLM experiments: main emotion-report condition, no-report control, unfairness-report control, intentionality-report control, persona/no-persona control, and temperature manipulation.

### 02_unfairness_sae
Unfairness-related SAE activation screening and feature steering. The original Qwen3-32B layer-48 Trainer-1 SAE is the default reference and available at https://huggingface.co/adamkarvonen/qwen3-32b-saes, but the runtime accepts alternative compatible base-model/SAE pairs through JSON profiles and CLI overrides.

### 03_emotion_sae_features
Emotion-feature identification and targeted steering: anger/joy corpus construction, activation extraction, candidate ranking, generation validation and steering. The same config-driven SAE runtime is used throughout.

## SAE model compatibility

See `SAE_COMPATIBILITY.md` for the profile schema, supported SAE loaders, hook configuration and rules for switching checkpoints. Each SAE section contains:

```text
config/qwen3_32b_reference.json   # exact supplied Qwen reference
config/custom_sae_template.json   # template for another compatible model/SAE
```

Generic path overrides are available through:

```bash
export BASE_MODEL_DIR=/path/to/base/model
export SAE_LOCAL_DIR=/path/to/sae/repository
```

Feature IDs are checkpoint-specific. When the base model, SAE repository, architecture, width, layer, trainer or hook site changes, feature screening/selection must be rerun before steering.

## Environment

```bash
pip install -r requirements-api.txt
pip install -r requirements-sae.txt
```

The project targets Python 3.12. API credentials for Section 01 are supplied through `LLM_API_KEY`; `LLM_BASE_URL` is optional.

## Prompt/data consistency

`project_core/data.py` is the single source of truth for persona generation. It includes the supplied demographic and psychological fields, including AQ/CESD group labels and SVO choice-count details.

`project_core/prompts.py` contains the game text and the four reporting-condition output prompts.

`project_core/sae_runtime.py` is the shared model/SAE loader for Sections 02 and 03. It keeps the experimental logic separate from checkpoint-specific loading details.

## Import/path handling

All executable scripts now locate the repository root from their own file path before importing `project_core`. They can therefore be launched from the repository root or from another working directory. For IDE/notebook use, you may additionally install the shared package in editable mode:

```bash
pip install -e .
```

This is optional for the supplied command-line entry points.
