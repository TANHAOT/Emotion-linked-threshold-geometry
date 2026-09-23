"""Config-driven loading utilities for base models and sparse autoencoders.

Parts 02 and 03 use this module so the experimental logic is no longer tied to
Qwen3-32B or to one SAE checkpoint. A JSON profile specifies the base model,
SAE loader/checkpoint, hook location, and generation defaults. CLI overrides can
replace any model/SAE location without editing source files.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _expand(value: str | os.PathLike[str] | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "sae_profile"


def load_sae_profile(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"SAE profile must be a JSON object: {path}")
    for section in ("model", "sae"):
        if section not in profile or not isinstance(profile[section], dict):
            raise ValueError(f"SAE profile is missing object '{section}': {path}")
    profile.setdefault("name", path.stem)
    profile.setdefault("generation", {})
    profile["_profile_path"] = str(path)
    return profile




def profile_fingerprint(profile: dict[str, Any]) -> str:
    """Stable short identity for the model/SAE pairing (generation settings excluded)."""
    model = profile.get("model", {})
    sae = profile.get("sae", {})
    payload = {
        "model": {"repo_id": model.get("repo_id") or model.get("local_dir")},
        "sae": {
            "loader": sae.get("loader"),
            "source": sae.get("repo_id") or sae.get("local_file") or sae.get("local_dir"),
            "filename": sae.get("filename"),
            "layer": sae.get("layer"),
            "trainer": sae.get("trainer"),
            "hook_module": sae.get("hook_module"),
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def add_sae_override_arguments(parser) -> None:
    """Add common model/SAE override flags to an argparse parser."""
    parser.add_argument("--model-id", default=None, help="Override model.repo_id from the SAE profile.")
    parser.add_argument("--model-dir", default=None, help="Override the local base-model directory.")
    parser.add_argument("--sae-repo-id", default=None, help="Override sae.repo_id from the SAE profile.")
    parser.add_argument("--sae-local-dir", default=None, help="Override the local SAE repository/cache directory.")
    parser.add_argument("--sae-file", default=None, help="Use an exact local SAE weight file instead of the profile filename.")
    parser.add_argument("--sae-config-file", default=None, help="Config file accompanying --sae-file when required by the loader.")
    parser.add_argument("--sae-filename", default=None, help="Override the Hugging Face filename/template.")
    parser.add_argument("--sae-loader", default=None, choices=(
        "dictionary_batch_topk", "dictionary_matryoshka_batch_topk", "dictionary_topk", "dictionary_jumprelu",
        "dictionary_relu", "gemma_scope_jumprelu",
    ), help="Override the SAE loader implementation.")
    parser.add_argument("--sae-layer", type=int, default=None, help="Override the residual-stream layer.")
    parser.add_argument("--sae-trainer", default=None, help="Override the trainer/checkpoint label used in filename templates.")
    parser.add_argument("--hook-module", default=None, help="Override the module path to hook, e.g. model.layers.{layer}. Use 'auto' for model-family detection.")


def apply_sae_overrides(profile: dict[str, Any], args) -> dict[str, Any]:
    profile = copy.deepcopy(profile)
    model_cfg = profile["model"]
    sae_cfg = profile["sae"]
    mapping = {
        "model_id": (model_cfg, "repo_id"),
        "model_dir": (model_cfg, "local_dir"),
        "sae_repo_id": (sae_cfg, "repo_id"),
        "sae_local_dir": (sae_cfg, "local_dir"),
        "sae_file": (sae_cfg, "local_file"),
        "sae_config_file": (sae_cfg, "config_file"),
        "sae_filename": (sae_cfg, "filename"),
        "sae_loader": (sae_cfg, "loader"),
        "sae_layer": (sae_cfg, "layer"),
        "sae_trainer": (sae_cfg, "trainer"),
        "hook_module": (sae_cfg, "hook_module"),
    }
    for attr, (target, key) in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            target[key] = value
    return profile


def _dtype(name: str):
    import torch
    key = str(name).lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16,
        "fp32": torch.float32, "float32": torch.float32,
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dtype '{name}'. Use bfloat16, float16, or float32.")
    return aliases[key]


def _resolved_local_dir(cfg: dict[str, Any], env_default: str | None = None) -> Path | None:
    env_name = cfg.get("local_dir_env") or env_default
    if env_name and os.environ.get(env_name):
        return _expand(os.environ[env_name])
    return _expand(cfg.get("local_dir"))


def _resolve_submodule_by_path(model, path: str, layer: int):
    formatted = path.format(layer=layer)
    current = model
    for token in formatted.split("."):
        if not token:
            continue
        if token.isdigit():
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current


def resolve_hook_submodule(model, model_utils, hook_module: str | None, layer: int):
    if hook_module in (None, "", "auto"):
        return model_utils.get_submodule(model, layer)
    return _resolve_submodule_by_path(model, hook_module, layer)


def _load_local_sae(*, loader: str, weight_path: Path, config_path: Path | None,
                    model_name: str, layer: int, device, dtype, vendor_modules: dict[str, Any],
                    require_normalized_decoder: bool = True):
    import numpy as np
    import torch

    batch_topk = vendor_modules["batch_topk"]
    topk = vendor_modules["topk"]
    jumprelu = vendor_modules["jumprelu"]
    relu = vendor_modules["relu"]

    if not weight_path.exists():
        raise FileNotFoundError(f"Local SAE weight file not found: {weight_path}")
    if config_path is None:
        candidate = weight_path.with_name("config.json")
        config_path = candidate if candidate.exists() else None
    config = json.loads(config_path.read_text()) if config_path and config_path.exists() else {}

    if loader == "gemma_scope_jumprelu":
        params = np.load(weight_path)
        state = {key: torch.from_numpy(value).cpu() for key, value in params.items()}
        sae = jumprelu.JumpReluSAE(
            d_in=state["W_enc"].shape[0], d_sae=state["W_enc"].shape[1],
            model_name=model_name, hook_layer=layer, device=device, dtype=dtype,
        )
        sae.load_state_dict(state)
    else:
        raw = torch.load(weight_path, map_location="cpu")
        if loader in ("dictionary_batch_topk", "dictionary_matryoshka_batch_topk", "dictionary_topk"):
            if loader == "dictionary_matryoshka_batch_topk":
                raw = dict(raw)
                raw.pop("group_sizes", None)
            if "encoder.weight" in raw:
                mapping = {
                    "encoder.weight": "W_enc", "decoder.weight": "W_dec",
                    "encoder.bias": "b_enc", "bias": "b_dec", "k": "k", "threshold": "threshold",
                }
                state = {mapping.get(key, key): value for key, value in raw.items()}
                state["W_enc"] = state["W_enc"].T
                state["W_dec"] = state["W_dec"].T
            else:
                state = dict(raw)
            trainer_cfg = config.get("trainer", {})
            k = int(trainer_cfg.get("k", config.get("k", state.get("k", 0))))
            if k <= 0:
                raise ValueError("Top-k SAE requires k in config.json or checkpoint state.")
            cls = batch_topk.BatchTopKSAE if loader in ("dictionary_batch_topk", "dictionary_matryoshka_batch_topk") else topk.TopKSAE
            kwargs = dict(
                d_in=state["b_dec"].shape[0], d_sae=state["b_enc"].shape[0], k=k,
                model_name=model_name, hook_layer=layer, device=device, dtype=dtype,
            )
            sae = cls(**kwargs)
            # Runtime-derived k is not a learned parameter for these classes.
            state.pop("k", None)
            # Some checkpoints include a threshold while the instantiated class does not.
            if "threshold" in state and "threshold" not in sae.state_dict():
                state.pop("threshold")
            sae.load_state_dict(state, strict=False)
        elif loader == "dictionary_jumprelu":
            sae = jumprelu.JumpReluSAE(
                d_in=raw["b_dec"].shape[0], d_sae=raw["b_enc"].shape[0],
                model_name=model_name, hook_layer=layer, device=device, dtype=dtype,
            )
            sae.load_state_dict(raw)
        elif loader == "dictionary_relu":
            mapping = {"bias": "b_dec"}
            state = {mapping.get(key, key): value for key, value in raw.items()}
            d_in = state["b_dec"].shape[0]
            d_sae = state.get("b_enc", state.get("encoder.bias")).shape[0]
            sae = relu.ReluSAE(d_in, d_sae, model_name, layer, device, dtype)
            sae.load_state_dict(state)
        else:
            raise ValueError(f"Unsupported local SAE loader: {loader}")

    sae.to(device=device, dtype=dtype)
    if require_normalized_decoder and hasattr(sae, "check_decoder_norms") and not sae.check_decoder_norms():
        if loader == "dictionary_relu" and hasattr(sae, "normalize_decoder"):
            sae.normalize_decoder()
        else:
            raise ValueError("Decoder vectors are not normalized for the selected SAE.")
    return sae


def _load_hf_sae(*, loader: str, repo_id: str, filename: str, local_dir: Path,
                 model_name: str, layer: int, device, dtype, vendor_modules: dict[str, Any]):
    batch_topk = vendor_modules["batch_topk"]
    topk = vendor_modules["topk"]
    jumprelu = vendor_modules["jumprelu"]
    relu = vendor_modules["relu"]
    local_dir.mkdir(parents=True, exist_ok=True)
    common = dict(repo_id=repo_id, filename=filename, model_name=model_name,
                  device=device, dtype=dtype, local_dir=str(local_dir))
    if loader == "dictionary_batch_topk":
        return batch_topk.load_dictionary_learning_batch_topk_sae(**common, layer=layer)
    if loader == "dictionary_matryoshka_batch_topk":
        return batch_topk.load_dictionary_learning_matryoshka_batch_topk_sae(**common, layer=layer)
    if loader == "dictionary_topk":
        return topk.load_dictionary_learning_topk_sae(**common, layer=layer)
    if loader == "dictionary_jumprelu":
        return jumprelu.load_dictionary_learning_jump_relu_sae(**common, layer=layer)
    if loader == "dictionary_relu":
        return relu.load_dictionary_learning_relu_sae(**common, layer=layer)
    if loader == "gemma_scope_jumprelu":
        return jumprelu.load_gemma_scope_jumprelu_sae(
            repo_id=repo_id, filename=filename, layer=layer, model_name=model_name,
            device=device, dtype=dtype, local_dir=str(local_dir),
        )
    raise ValueError(f"Unsupported SAE loader: {loader}")


@dataclass
class SAERuntime:
    profile: dict[str, Any]
    tokenizer: Any
    model: Any
    sae: Any
    submodule: Any
    input_device: Any
    layer: int
    trainer: str
    profile_name: str
    fingerprint: str
    profile_tag: str


def load_sae_runtime(profile: dict[str, Any], vendor_dir: str | os.PathLike[str]) -> SAERuntime:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vendor_dir = Path(vendor_dir).resolve()
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    from interp_tools import model_utils
    import interp_tools.saes.batch_topk_sae as batch_topk
    import interp_tools.saes.topk_sae as topk
    import interp_tools.saes.jumprelu_sae as jumprelu
    import interp_tools.saes.relu_sae as relu

    model_cfg = profile["model"]
    sae_cfg = profile["sae"]
    model_id = model_cfg.get("repo_id")
    if not model_id:
        raise ValueError("model.repo_id is required in the SAE profile.")
    model_dir = _resolved_local_dir(model_cfg, "BASE_MODEL_DIR")
    if model_dir is not None:
        if not model_dir.exists() or not any(model_dir.iterdir()):
            if model_cfg.get("download_to_local_dir", True):
                print(f"Downloading base model {model_id} to {model_dir} ...")
                model_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download(repo_id=model_id, local_dir=str(model_dir))
            else:
                model_dir = None
    model_source = str(model_dir) if model_dir is not None else model_id

    model_dtype = _dtype(model_cfg.get("dtype", "bfloat16"))
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=trust_remote_code,
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        padding_side=model_cfg.get("padding_side", "right"),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": model_dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
    }
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs).eval()

    layer = int(sae_cfg["layer"])
    trainer = str(sae_cfg.get("trainer", ""))
    hook_module = sae_cfg.get("hook_module", "auto")
    submodule = resolve_hook_submodule(model, model_utils, hook_module, layer)
    try:
        hook_device = next(submodule.parameters()).device
    except StopIteration:
        hook_device = next(model.parameters()).device
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device

    sae_dtype = _dtype(sae_cfg.get("dtype", model_cfg.get("dtype", "bfloat16")))
    loader = sae_cfg.get("loader", "dictionary_batch_topk")
    filename_template = sae_cfg.get("filename")
    filename = filename_template.format(layer=layer, trainer=trainer) if filename_template else None
    local_file = _expand(sae_cfg.get("local_file"))
    config_file = _expand(sae_cfg.get("config_file"))
    vendor_modules = {"batch_topk": batch_topk, "topk": topk, "jumprelu": jumprelu, "relu": relu}

    if local_file is not None:
        print(f"Loading local SAE: {local_file}")
        sae = _load_local_sae(
            loader=loader, weight_path=local_file, config_path=config_file,
            model_name=model_id, layer=layer, device=hook_device, dtype=sae_dtype,
            vendor_modules=vendor_modules,
            require_normalized_decoder=bool(sae_cfg.get("require_normalized_decoder", True)),
        )
    else:
        repo_id = sae_cfg.get("repo_id")
        if not repo_id or not filename:
            raise ValueError("SAE profile requires either sae.local_file or both sae.repo_id and sae.filename.")
        local_dir = _resolved_local_dir(sae_cfg, "SAE_LOCAL_DIR") or Path("downloaded_saes").resolve()
        print(f"Loading SAE {repo_id}:{filename} into {local_dir}")
        sae = _load_hf_sae(
            loader=loader, repo_id=repo_id, filename=filename, local_dir=local_dir,
            model_name=model_id, layer=layer, device=hook_device, dtype=sae_dtype,
            vendor_modules=vendor_modules,
        )

    d_in = int(sae.W_enc.shape[0])
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is not None and int(hidden_size) != d_in:
        raise ValueError(
            f"Model/SAE dimension mismatch: model hidden_size={hidden_size}, SAE d_in={d_in}. "
            "Check the base model, hook layer/site, and SAE checkpoint."
        )

    profile_name = _slug(str(profile.get("name", "sae_profile")))
    fingerprint = profile_fingerprint(profile)
    profile_tag = f"{profile_name}_{fingerprint}"
    return SAERuntime(
        profile=profile, tokenizer=tokenizer, model=model, sae=sae, submodule=submodule,
        input_device=input_device, layer=layer, trainer=trainer, profile_name=profile_name,
        fingerprint=fingerprint, profile_tag=profile_tag,
    )


def render_chat(tokenizer, messages: list[dict[str, str]], generation_cfg: dict[str, Any] | None = None) -> str:
    """Render a chat prompt with a safe plain-text fallback for compatible instruct models."""
    generation_cfg = generation_cfg or {}
    kwargs = dict(generation_cfg.get("chat_template_kwargs", {}))
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs,
        )
    except (TypeError, ValueError, AttributeError):
        parts = []
        for message in messages:
            role = message.get("role", "user").upper()
            parts.append(f"{role}: {message.get('content', '')}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)


def token_id_set(tokenizer, texts: list[str]) -> set[int]:
    ids: set[int] = set()
    for text in texts:
        sequence = tokenizer.encode(text, add_special_tokens=False)
        if sequence:
            ids.add(sequence[-1])
    return ids


def extract_binary_choice_logits(tokenizer, generated_ids, scores, generation_cfg: dict[str, Any] | None = None):
    """Locate the generated 0/1 choice after the configured marker and return logits."""
    generation_cfg = generation_cfg or {}
    generated = generated_ids.detach().cpu().tolist() if hasattr(generated_ids, "detach") else list(generated_ids)
    marker = str(generation_cfg.get("choice_marker", "choice"))
    marker_variants = generation_cfg.get("choice_marker_variants", [marker, f" {marker}"])
    marker_sequences = [tokenizer.encode(text, add_special_tokens=False) for text in marker_variants]
    marker_sequences = [seq for seq in marker_sequences if seq]
    zero_ids = token_id_set(tokenizer, generation_cfg.get("zero_tokens", ["0", " 0"]))
    one_ids = token_id_set(tokenizer, generation_cfg.get("one_tokens", ["1", " 1"]))

    start = 0
    for seq in marker_sequences:
        for index in range(0, len(generated) - len(seq) + 1):
            if generated[index:index + len(seq)] == seq:
                start = index + len(seq)
                break
        if start:
            break
    scan_limit = int(generation_cfg.get("choice_scan_tokens", 10))
    for index in range(start, min(len(generated), start + scan_limit)):
        token_id = generated[index]
        if token_id in zero_ids or token_id in one_ids:
            if index >= len(scores):
                return None, None
            logits = scores[index][0]
            zero_logit = max((float(logits[token].item()) for token in zero_ids), default=None)
            one_logit = max((float(logits[token].item()) for token in one_ids), default=None)
            return zero_logit, one_logit
    return None, None


def describe_profile(profile: dict[str, Any]) -> str:
    sae = profile["sae"]
    model = profile["model"]
    return (
        f"profile={profile.get('name')} model={model.get('repo_id')} "
        f"sae_loader={sae.get('loader')} sae_repo={sae.get('repo_id')} "
        f"layer={sae.get('layer')} trainer={sae.get('trainer')} hook={sae.get('hook_module', 'auto')}"
    )
