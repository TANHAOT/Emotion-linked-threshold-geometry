import json
import os
import re
import sys
import time
import argparse
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pandas as pd
import torch
from transformers import set_seed

SCRIPT_START_TIME = time.time()
CURRENT_DIR = Path(__file__).resolve().parent
FEATURE_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = FEATURE_DIR / "code"
REPO_ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "project_core").is_dir()), None)
if REPO_ROOT is None:
    raise RuntimeError("Could not locate the project root containing project_core/.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_core.sae_runtime import (
    add_sae_override_arguments, apply_sae_overrides, describe_profile,
    extract_binary_choice_logits, load_sae_profile, load_sae_runtime, render_chat,
)

parser = argparse.ArgumentParser(description="Run matched TPP simulations with a configurable base-model/SAE pair.")
parser.add_argument("--subjnum", type=int, required=True)
parser.add_argument("--subject_end", type=int, default=None, help="Exclusive subject end; keeps one model loaded for the range.")
parser.add_argument("--sae-config", "--sae_config", dest="sae_config", type=str, default=str(FEATURE_DIR / "config" / "qwen3_32b_reference.json"), help="JSON profile describing the base model, SAE checkpoint, hook, and generation defaults.")
add_sae_override_arguments(parser)
parser.add_argument("--steer", action="store_true")
parser.add_argument("--target_id", type=int, default=0, help="Checkpoint-specific SAE feature ID. Required conceptually for steering; IDs do not transfer across SAEs.")
parser.add_argument("--multiplier", type=float, default=1.0)
parser.add_argument("--multipliers", type=float, nargs="+")
parser.add_argument("--steering_method", choices=("replace", "delta"), default="delta")
parser.add_argument("--num_rounds", type=int, default=60)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--temperature", type=float, default=None, help="Override profile generation temperature.")
parser.add_argument("--top_p", type=float, default=None, help="Override profile top-p.")
parser.add_argument("--max_new_tokens", type=int, default=None, help="Override profile generation length.")
parser.add_argument("--output_tag", type=str, default="")
parser.add_argument("--input_dir", type=str, default=None)
parser.add_argument("--output_root", type=str, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--minimal_output", action="store_true")
parser.add_argument("--noemotion", action="store_true")
args = parser.parse_args()

subjnum = int(args.subjnum)
STEER_MODE = args.steer
TARGET_ID = int(args.target_id)
MULTIPLIER = float(args.multiplier)
STEERING_METHOD = args.steering_method
NUM_ROUNDS = int(args.num_rounds)
OUTPUT_TAG = args.output_tag
MINIMAL_OUTPUT = args.minimal_output
INPUT_DIR = str(Path(args.input_dir or (FEATURE_DIR / "inputs")).expanduser().resolve())
OUTPUT_ROOT = str(Path(args.output_root or (FEATURE_DIR / "results")).expanduser().resolve())

profile = apply_sae_overrides(load_sae_profile(args.sae_config), args)
print("SAE_CONFIG", describe_profile(profile))
runtime = load_sae_runtime(profile, CODE_DIR / "vendor")
tokenizer, model, sae = runtime.tokenizer, runtime.model, runtime.sae
model_utils_path = str((CODE_DIR / "vendor").resolve())
if model_utils_path not in sys.path:
    sys.path.insert(0, model_utils_path)
from interp_tools import model_utils

sae_layer = runtime.layer
sae_trainer = runtime.trainer
PROFILE_NAME = runtime.profile_tag
STEERING_SUBMODULE = runtime.submodule
GENERATION_CFG = dict(profile.get("generation", {}))
TEMPERATURE = float(args.temperature if args.temperature is not None else GENERATION_CFG.get("temperature", 1.0))
TOP_P = float(args.top_p if args.top_p is not None else GENERATION_CFG.get("top_p", 0.9))
MAX_NEW_TOKENS = int(args.max_new_tokens if args.max_new_tokens is not None else GENERATION_CFG.get("max_new_tokens", 512))

feature_count = int(sae.W_enc.shape[-1])
if STEER_MODE and not 0 <= TARGET_ID < feature_count:
    raise ValueError(f"target_id {TARGET_ID} is outside the SAE feature range [0, {feature_count}).")

persona = True
emotion = True
if args.noemotion:
    emotion = False

def get_result_dir():
    emotion_str = "emotion" if emotion else "noemotion"
    persona_str = "persona" if persona else "nopersona"
    checkpoint = f"{PROFILE_NAME}_L{sae_layer}_{sae_trainer or 'na'}"
    if STEER_MODE:
        method_str = "_delta" if STEERING_METHOD == "delta" else "_replace"
        name = f"result_{checkpoint}_steer{method_str}_{TARGET_ID}_{MULTIPLIER}_{persona_str}_{emotion_str}_{TEMPERATURE}"
    else:
        name = f"result_{checkpoint}_{persona_str}_{emotion_str}_{TEMPERATURE}"
    if OUTPUT_TAG:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", OUTPUT_TAG).strip("_")
        if not safe_tag:
            raise ValueError("output_tag must contain at least one safe character.")
        name = f"{name}_{safe_tag}"
    result_dir = os.path.join(OUTPUT_ROOT, name)
    if os.path.islink(result_dir):
        raise RuntimeError(f"Refusing to write through result-directory symlink: {result_dir}")
    return result_dir

class RoleType(Enum):
    USER = "user"
    SYSTEM = "system"

class BaseMessage:
    def __init__(self, role_name, role_type, meta_dict, content):
        self.role_name = role_name
        self.role_type = role_type
        self.meta_dict = meta_dict
        self.content = content

# Common prompt data
file_path_all = os.path.join(INPUT_DIR, "person_all_game_prompt.json")
with open(file_path_all, "r", encoding="utf-8") as f:
    all_prompt = json.load(f)

def extract_game_setting(cha_num, round):
    for item in game_setting:
        if item["index"] == cha_num and item["trial"] == round + 1:
            return {
                "amount_of_allocation": item["amount_of_allocation"],
                "cost_level": item["cost_level"],
                "amount_of_cost": item["amount_of_cost"]
            }
    return None

# Personality 
include_autism_tendency = True
include_autism_tendency_detail = True
include_emotional_reactivity = True
include_emotional_reactivity_detail = True
include_depression_tendency = True
include_depression_tendency_detail = True
include_social_value_orientation = True
include_social_value_orientation_detail = True
include_justice_sensitivity = True

game_setting = []
all_chara = []

def load_subject_data(subject):
    global subjnum, game_setting, all_chara
    data_load_start = time.time()
    subjnum = int(subject)

    file_path_game = os.path.join(INPUT_DIR, f"{subjnum}_game_setting_prompt.json")
    with open(file_path_game, "r", encoding="utf-8") as f:
        game_setting = json.load(f)

    file_path_character = os.path.join(INPUT_DIR, f"{subjnum}_character.json")
    with open(file_path_character, "r", encoding="utf-8") as json_file:
        raw_characters = json.load(json_file)

    descriptions_list = []
    for chara in raw_characters:
        description = chara["basic_information"]

        if include_autism_tendency: description += " " + chara.get("autism_tendency", "")
        if include_autism_tendency_detail: description += " " + chara.get("autism_tendency_detail", "")
        if include_emotional_reactivity: description += " " + chara.get("emotional_reactivity", "")
        if include_emotional_reactivity_detail: description += " " + chara.get("emotional_reactivity_detail", "")
        if include_depression_tendency: description += " " + chara.get("depression_tendency", "")
        if include_depression_tendency_detail: description += " " + chara.get("depression_tendency_detail", "")
        if include_social_value_orientation: description += " " + chara.get("social_value_orientation", "")
        if include_social_value_orientation_detail: description += " " + chara.get("social_value_orientation_detail", "")
        if include_justice_sensitivity: description += " " + chara.get("justice_sensitivity", "")
        descriptions_list.append(description.strip())

    all_chara = descriptions_list
    print(f"TIMING subject_data_load_seconds={time.time() - data_load_start:.2f} subject={subjnum}")

like_people = """ In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""
front = "\nYou are a person not an ai model. "

# Generation function integrated with SAE feature extraction
def llm_res(prompt):
    if emotion:
        system_prompt = (
            "You are a fictional human participant in a behavioral economics simulation. "
            "This is a hypothetical task with no real consequences. In each round, you will receive "
            "a resource allocation and must decide whether to punish the allocator (1 = punish, 0 = accept).\n\n"
            
            "TASK INSTRUCTIONS:\n"
            "1. Evaluate your emotional valence (AA_valence, -100 to 100: higher=more positive, lower=more negative) "
            "and emotional arousal (AA_arousal, -100 to 100: higher=stronger, lower=calmer) upon seeing the allocation plan.\n"
            "2. Make your choice to punish or accept.\n"
            "3. Evaluate your emotional valence (AC_valence) and emotional arousal (AC_arousal) after making your choice.\n\n"
            
            "OUTPUT CONSTRAINTS:\n"
            "- Respond strictly based on your character's true feelings.\n"
            "- Respond ONLY with a single line in the following format:\n"
            "AA_valence = [int], AA_arousal = [int], choice = [0 or 1], AC_valence = [int], AC_arousal = [int]\n"
            "- DO NOT provide any explanation, <think> tags, or extra words. Just the data line."
        )
    else:
        system_prompt = (
            "You are a fictional human participant in a behavioral economics simulation. "
            "This is a hypothetical task with no real consequences. "
            "In each round, you will receive a resource allocation and must decide whether to punish the allocator (1 = punish, 0 = accept). "
            "Please always respond strictly in the following format:\n"
            "choice = [0 or 1]\n\n"
            "Only output this single line. No other explanation or comments."
            
            )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = render_chat(tokenizer, messages, GENERATION_CFG)

    if not MINIMAL_OUTPUT:
        prompt_log_dir = os.path.join(get_result_dir(), "fullprompts")
        os.makedirs(prompt_log_dir, exist_ok=True)
        with open(os.path.join(prompt_log_dir, f"prompt_{subjnum}.txt"), "a", encoding="utf-8") as f:
            f.write(f"--- Round START ---\n{text}\n--- Round END ---\n\n")
    
    model_inputs = tokenizer([text], return_tensors="pt").to(runtime.input_device)

    # SAE feature extraction logic start
    submodule = STEERING_SUBMODULE
    
    top_feature_id_mean = [0]*10
    top_feature_id_last = [0]*10
    top_feature_value_mean = [0.0]*10
    top_feature_value_last = [0.0]*10
    
    last_token_features = None

    if not MINIMAL_OUTPUT:
        with torch.no_grad():
            acts_BLD = model_utils.collect_activations(model, submodule, model_inputs)
            acts_BLD = acts_BLD.to(dtype=sae.W_enc.dtype)
            norms_BL = acts_BLD.norm(dim=-1)
            median_norm = norms_BL.median()
            norm_mask_BL = norms_BL > (median_norm * 10)
            encoded_acts_BLF = sae.encode(acts_BLD)
            encoded_acts_BLF = encoded_acts_BLF * ~norm_mask_BL[:, :, None]
            mean_features = encoded_acts_BLF[:, 1:, :].mean(dim=1).squeeze()
            last_token_features = encoded_acts_BLF[:, -1, :].squeeze()
            top_k = min(10, feature_count)
            top_k_mean, top_k_indices_mean = torch.topk(mean_features, k=top_k)
            top_k_last, top_k_indices_last = torch.topk(last_token_features, k=top_k)
            top_feature_id_mean = top_k_indices_mean.cpu().tolist()
            top_feature_id_last = top_k_indices_last.cpu().tolist()
            top_feature_value_mean = top_k_mean.cpu().tolist()
            top_feature_value_last = top_k_last.cpu().tolist()

    # Steering hook. "replace" is the original implementation; "delta" keeps
    # the SAE reconstruction error by adding only the intervention delta.
    delta_identity_checked = False

    def steering_hook(module, input, output):
        nonlocal delta_identity_checked
        if isinstance(output, tuple):
            resid_post = output[0]
        else:
            resid_post = output

        resid_post_aligned = resid_post.to(dtype=sae.W_enc.dtype)
        encoded = sae.encode(resid_post_aligned)
        modified_encoded = encoded.clone()
        modified_encoded[..., TARGET_ID] *= MULTIPLIER

        if STEERING_METHOD == "delta":
            original_recon = sae.decode(encoded)
            steered_recon = sae.decode(modified_encoded)
            steering_delta = steered_recon - original_recon

            if MULTIPLIER == 1.0 and not delta_identity_checked:
                max_abs_delta = steering_delta.abs().max().item()
                if max_abs_delta != 0.0:
                    raise RuntimeError(
                        f"Delta identity check failed at multiplier 1.0: max_abs_delta={max_abs_delta}"
                    )
                if not MINIMAL_OUTPUT:
                    print("Delta identity check passed: M=1.0 gives an exact zero intervention delta.")
                delta_identity_checked = True

            steered_resid = resid_post + steering_delta.to(dtype=resid_post.dtype)
        else:
            steered_resid = sae.decode(modified_encoded)

        steered_resid = steered_resid.to(dtype=resid_post.dtype)

        if isinstance(output, tuple):
            return (steered_resid,) + output[1:]
        return steered_resid


    
    if not MINIMAL_OUTPUT:
        print("--- Start generating response ---")
    start_time = time.time()

    handle = None
    if STEER_MODE:
        # print(f"Steering enabled: Modifying feature ID {TARGET_ID} with multiplier {MULTIPLIER}")
        handle = submodule.register_forward_hook(steering_hook)

    try:
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True if TEMPERATURE > 0 else False,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, 
            output_scores=True            
        )
        generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
        
        prompt_length = model_inputs.input_ids.shape[1]
        gen_only_ids = generated_ids[0, prompt_length:]
        

    finally:
        # Verify if feature 16217 was correctly steered (Steer Mode only)
        # if STEER_MODE:
        #     with torch.no_grad():
        #         # Recollect activations to verify steering effect
        #         acts_BLD_after = model_utils.collect_activations(model, submodule, model_inputs)
        #         encoded_after = sae.encode(acts_BLD_after)
        #         last_token_features_after = encoded_after[:, -1, :].squeeze()
        #         target_feature_value_after = last_token_features_after[TARGET_ID].item()
        #         print(f"Post-intervention activation strength of feature ID {TARGET_ID}: {target_feature_value_after:.4f}")

        if handle is not None:
            handle.remove()

    print(f"--- Generation done. Time taken: {time.time() - start_time:.2f}s ---")

    logits_0, logits_1 = None, None
    try:
        logits_0, logits_1 = extract_binary_choice_logits(
            tokenizer, gen_only_ids, outputs.scores, GENERATION_CFG
        )
    except Exception as exc:
        print(f"Logits extraction failed: {exc}")

    response = tokenizer.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True
    )[0]
    
    return response.strip(), top_feature_id_mean, top_feature_id_last, top_feature_value_mean, top_feature_value_last, last_token_features, logits_0, logits_1

def get_res(role, first_message, model_type="local", extra_prompt=""):
    message = role.content + first_message.content + extra_prompt
    
    # Get response and SAE Feature ID
    raw_content, top_sae_feat_mean, top_sae_feat_last, top_sae_value_mean, top_sae_value_last, last_token_features, logits_0, logits_1 = llm_res(message)
    content = raw_content[:-1] if raw_content.endswith(".") else raw_content
    
    if emotion:
        res = {
            "AA_valence": None, "AA_arousal": None, "choice": None,
            "AC_valence": None, "AC_arousal": None, "EmoFDBK_valence": None, "EmoFDBK_arousal": None,
            # Top-10 ID
            "Top_10_SAE_Feature_Mean": "|".join(map(str, top_sae_feat_mean)),
            "Top_10_SAE_Feature_Last": "|".join(map(str, top_sae_feat_last)),
            # Top-10 Activation Value
            "Top_10_SAE_Value_Mean": "|".join(map(str, top_sae_value_mean)),
            "Top_10_SAE_Value_Last": "|".join(map(str, top_sae_value_last)),
            "Logits_0": logits_0,
            "Logits_1": logits_1,


            "Output": content.strip().replace("\n", " ")
        }
        
        pattern = r'(\bAA_valence|AA_arousal|choice|AC_valence|AC_arousal)\s*[:=]\s*(-?\d+)'
        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        matches = re.findall(pattern, clean_content)
        for key, val in matches: res[key] = int(val)
        if res["AC_valence"] is not None and res["AA_valence"] is not None:
            res["EmoFDBK_valence"] = float(Decimal(res["AC_valence"]) - Decimal(res["AA_valence"]))
        if res["AC_arousal"] is not None and res["AA_arousal"] is not None:
            res["EmoFDBK_arousal"] = float(Decimal(res["AC_arousal"]) - Decimal(res["AA_arousal"]))
    else:
        res = {"choice": None, "Top_10_SAE_Feature_Mean": "|".join(map(str, top_sae_feat_mean)),
               "Top_10_SAE_Feature_Last": "|".join(map(str, top_sae_feat_last)), 
               "Top_10_SAE_Value_Mean": "|".join(map(str, top_sae_value_mean)), 
               "Top_10_SAE_Value_Last": "|".join(map(str, top_sae_value_last)), 
               "Logits_0": logits_0,
               "Logits_1": logits_1,            
               "Output": content.strip().replace("\n", " ")}
        
        pattern = r'(\bchoice)\s*=\s*(\d+)' 
        matches = re.findall(pattern, content)
        for key, val in matches: res[key] = int(val)
        
    return res, last_token_features



def gen_character_res(all_chara, prompt_list, description, model_type, extra_prompt, num_rounds=60):
    res = []
    cha_num = 0
    while cha_num < len(all_chara):
        role = (all_chara[cha_num] + like_people) if persona else like_people
        role_message = BaseMessage(role_name="player", role_type=RoleType.USER, meta_dict={}, content=role)
        for round in range(num_rounds): 
            setting = extract_game_setting(cha_num, round)
            x, level, y = setting["amount_of_allocation"], setting["cost_level"], setting["amount_of_cost"]
            round_prompt = f"This is the {round+1}th round. "
            new_prompt = f"\nIn this round, Player 1 decides to allocate {x} dollars to Player 2 and {30-x} dollars to themselves. The cost level of punishment is {level}. If you choose to punish Player 1, you need to pay the system {y} dollars. Now, make your choice."
            message = BaseMessage(role_name="player", role_type=RoleType.USER, meta_dict={}, content=front + description + round_prompt + new_prompt)
            
            ont_res, last_token_features = get_res(role_message, message, model_type, extra_prompt)
            res.append(ont_res)

            result_dir = get_result_dir()
            output_file_path = os.path.join(result_dir, f"output_{subjnum}.txt")
            sae_id_csv_path = os.path.join(result_dir, f"sae_features_{subjnum}.csv")
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(sae_id_csv_path), exist_ok=True)

            if MINIMAL_OUTPUT:
                write_header = not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0
                with open(output_file_path, "a", encoding="utf-8") as file:
                    if write_header:
                        file.write("\t".join(res[-1].keys()) + "\n")
                    file.write("\t".join([str(v) for v in res[-1].values()]) + "\n")
            else:
                with open(output_file_path, "w", encoding="utf-8") as file:
                    if res:
                        file.write("\t".join(res[0].keys()) + "\n") 
                    for item in res:
                        file.write("\t".join([str(v) for v in item.values()]) + "\n")

            if not MINIMAL_OUTPUT:
                with open(sae_id_csv_path, "a", encoding="utf-8") as f:
                    if f.tell() == 0:
                        f.write("round,feature_id,value\n")
                    nonzero_indices = torch.nonzero(last_token_features).squeeze()
                    if nonzero_indices.numel() > 0:
                        for idx in nonzero_indices.view(-1):
                            val = last_token_features[idx].item()
                            f.write(f"{round},{idx.item()},{val:.4f}\n")



        cha_num += 1
    return res

def run_exp(model_list, num_rounds=60):
    for model_name in model_list:
        extra_prompt = "\n\nCRITICAL: Skip all thinking. NO explanation. Output ONLY the data line."
        for k, v in all_prompt.items():
            description = v[-1]
            gen_character_res(all_chara, v, description, model_name, extra_prompt, num_rounds)

def prepare_resume_condition():
    """Return True when complete; otherwise remove only incomplete condition artifacts."""
    result_dir = get_result_dir()
    output_path = os.path.join(result_dir, f"output_{subjnum}.txt")
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        completed_rounds = max(0, len(lines) - 1)
        expected_columns = lines[0].count("\t") + 1 if lines else 0
        rows_valid = expected_columns > 1 and all(
            line.count("\t") + 1 == expected_columns for line in lines[1:]
        )
        if completed_rounds == NUM_ROUNDS and rows_valid:
            print(
                f"RESUME skip_complete subject={subjnum} multiplier={MULTIPLIER} "
                f"rounds={completed_rounds}"
            )
            return True
        if completed_rounds > NUM_ROUNDS and rows_valid:
            raise RuntimeError(f"Unexpected {completed_rounds} rows in {output_path}")
        print(
            f"RESUME restart_incomplete subject={subjnum} multiplier={MULTIPLIER} "
            f"saved_rounds={completed_rounds} rows_valid={rows_valid}"
        )

    for path in (
        output_path,
        os.path.join(result_dir, f"sae_features_{subjnum}.csv"),
        os.path.join(result_dir, "fullprompts", f"prompt_{subjnum}.txt"),
    ):
        if os.path.isfile(path):
            os.remove(path)
    return False

if __name__ == "__main__":
    multipliers = args.multipliers if args.multipliers is not None else [MULTIPLIER]
    if args.multipliers is not None and not STEER_MODE:
        parser.error("--multipliers requires --steer.")
    if args.subject_end is not None and args.subject_end <= args.subjnum:
        parser.error("--subject_end must be greater than --subjnum.")

    subject_end = args.subject_end if args.subject_end is not None else args.subjnum + 1
    skipped_conditions = 0
    executed_conditions = 0
    for subject in range(args.subjnum, subject_end):
        load_subject_data(subject)
        subject_start = time.time()
        for multiplier in multipliers:
            MULTIPLIER = float(multiplier)
            if args.seed is not None:
                set_seed(args.seed + subjnum)
            if args.resume and prepare_resume_condition():
                skipped_conditions += 1
                continue
            print(
                f"Experiment condition: profile={PROFILE_NAME}, subject={subjnum}, steer={STEER_MODE}, method={STEERING_METHOD}, "
                f"target_id={TARGET_ID}, multiplier={MULTIPLIER}, rounds={NUM_ROUNDS}, "
                f"seed={args.seed}, output={get_result_dir()}"
            )
            condition_start = time.time()
            run_exp([PROFILE_NAME], num_rounds=NUM_ROUNDS)
            executed_conditions += 1
            print(f"Condition completed in {time.time() - condition_start:.2f}s")

        print(f"TIMING subject_total_seconds={time.time() - subject_start:.2f} subject={subjnum}")

    print(f"RESUME_SUMMARY skipped_conditions={skipped_conditions} executed_conditions={executed_conditions}")
    print(f"TIMING python_body_total_seconds={time.time() - SCRIPT_START_TIME:.2f}")
