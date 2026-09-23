# Configuring alternative SAE models

Parts 02 and 03 use the same config-driven SAE runtime (`project_core/sae_runtime.py`). 

```text
encode(hidden) -> feature activations
decode(feature activations) -> reconstructed hidden state
W_enc and W_dec
```

## Profile structure

Every profile has three blocks:

- `model`: Hugging Face model ID, optional local directory, dtype/device settings.
- `sae`: SAE loader, repository/local file, checkpoint filename, layer, trainer and hook module.
- `generation`: decoding defaults and 0/1 decision-token readout settings.

Use `02_unfairness_sae/config/custom_sae_template.json` or `03_emotion_sae_features/config/custom_sae_template.json` as the starting point.

## Supported SAE loaders

- `dictionary_batch_topk` — dictionary-learning BatchTopK checkpoints (`ae.pt` + `config.json`).
- `dictionary_matryoshka_batch_topk` — Matryoshka BatchTopK dictionary-learning checkpoints.
- `dictionary_topk` — dictionary-learning TopK checkpoints.
- `dictionary_jumprelu` — dictionary-learning JumpReLU checkpoints.
- `dictionary_relu` — dictionary-learning standard ReLU checkpoints.
- `gemma_scope_jumprelu` — Gemma-Scope-style JumpReLU `params.npz` checkpoints.

For Hugging Face SAEs, set `sae.repo_id` and `sae.filename`. For an exact local checkpoint, set `sae.local_file` (and `sae.config_file` when needed); the HF repository fields are then not required.

## Hook selection

`sae.hook_module` can be:

```text
auto
model.layers.{layer}
transformer.h.{layer}
gpt_neox.layers.{layer}
```

or another attribute path valid for the chosen Transformers model. `auto` uses the bundled model-family resolver for Qwen, Gemma, Llama, Mistral and Pythia-like models.

The hook must match the residual-stream location on which the SAE was trained. Matching the layer number alone is not sufficient if the SAE used a different pre/post residual site.

## Paths and overrides

Generic environment variables:

```bash
export BASE_MODEL_DIR=/path/to/base/model
export SAE_LOCAL_DIR=/path/to/sae/repository
```

CLI flags can override the profile without editing JSON:

```text
--model-id
--model-dir
--sae-repo-id
--sae-local-dir
--sae-file
--sae-config-file
--sae-filename
--sae-loader
--sae-layer
--sae-trainer
--hook-module
```

## Required procedure when changing SAE

Changing any of the following invalidates existing feature IDs: base model, SAE checkpoint, SAE architecture, width, layer, trainer, or hook site. Therefore:

1. rerun activation extraction/screening;
2. select new candidate feature IDs;
3. rerun semantic/generation validation where applicable;
4. only then run behavioural steering with the newly selected feature.

The runtime checks the SAE input dimension against the base model hidden size and rejects obvious model/SAE mismatches before the experiment starts.
