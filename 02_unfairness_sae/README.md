# 02 — Unfairness-related SAE feature screening and steering

This section implements the unfairness-related SAE workflow while keeping the original Qwen3-32B experiment as the default reference. The runtime is now **configuration-driven**: the base model, SAE checkpoint, SAE architecture/loader, residual hook, layer and trainer can be changed without editing source code.

## Default reference profile

`config/qwen3_32b_reference.json` reproduces the supplied Qwen setup:

- base model: `Qwen/Qwen3-32B`
- SAE repository: `adamkarvonen/qwen3-32b-saes`
- loader: `dictionary_batch_topk`
- layer: 48
- trainer: `trainer_1`
- hook: `model.layers.{layer}`

The supplement reports pilot candidates 9979, 10018, 11424 and 16217 for **this exact checkpoint**. Feature IDs are local to a model/SAE checkpoint and must not be transferred to another SAE.

## Workflow

```bash
# 1. Generate pilot inputs (default indices 400–419)
python 02_unfairness_sae/run_pipeline.py prepare

# 2. Baseline activation screening with the default Qwen SAE
python 02_unfairness_sae/run_pipeline.py screen

# 3. Rank unfairness-related features from the baseline result directory
python 02_unfairness_sae/rank_unfairness_features.py \
  --inputs 02_unfairness_sae/inputs \
  --results <baseline_result_dir>

# 4. Generate all 1,017 matched inputs before the full intervention
python 02_unfairness_sae/prepare_inputs.py --start 0 --end 1017

# 5. Reference steering experiment
python 02_unfairness_sae/run_pipeline.py steer --target-id 16217
```

## Testing another model/SAE pair

Copy and edit the template:

```bash
cp 02_unfairness_sae/config/custom_sae_template.json \
   02_unfairness_sae/config/my_sae.json
```

Then use the same experimental pipeline:

```bash
python 02_unfairness_sae/run_pipeline.py screen \
  --sae-config 02_unfairness_sae/config/my_sae.json

python 02_unfairness_sae/rank_unfairness_features.py \
  --inputs 02_unfairness_sae/inputs \
  --results <my_sae_baseline_result_dir>

python 02_unfairness_sae/run_pipeline.py steer \
  --sae-config 02_unfairness_sae/config/my_sae.json \
  --target-id <FEATURE_SELECTED_FOR_THIS_SAE>
```

You can also override a profile from the command line, for example:

```bash
python 02_unfairness_sae/run_pipeline.py screen \
  --sae-config 02_unfairness_sae/config/qwen3_32b_reference.json \
  --sae-layer 40 \
  --sae-trainer trainer_2 \
  --sae-filename 'path/resid_post_layer_{layer}/{trainer}/ae.pt'
```

Common overrides include `--model-id`, `--model-dir`, `--sae-repo-id`, `--sae-local-dir`, `--sae-file`, `--sae-config-file`, `--sae-filename`, `--sae-loader`, `--sae-layer`, `--sae-trainer`, and `--hook-module`.

The supported SAE loader families are documented in the root `SAE_COMPATIBILITY.md`.
