from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

AVERAGES = {
    "total_AQ_score": 65.17109145,
    "social_skill_score": 15.75221239,
    "routine_score": 9.418879056,
    "switching_score": 9.529006883,
    "imagination_score": 16.97443461,
    "numbers_patterns_score": 13.49655851,
    "social_behavior_score": 51.67453294,
    "total_ERS_score": 61.95968535,
    "duration_score": 12.33333333,
    "sensitivity_score": 28.633235,
    "intensity_score": 20.99311701,
    "total_CESD_score": 35.40117994,
    "positive_affect_score": 8.212389381,
    "depression_score": 13.58603736,
    "interpersonal_score": 3.000983284,
    "somatic_score": 10.60176991,
    "counts[P]": 5.152409046,
    "counts[I]": 3.382497542,
    "counts[C]": 0.465093412,
    "JSI_score": 32.32546706,
}

PERSONA_FIELDS = (
    "basic_information",
    "autism_tendency",
    "autism_tendency_detail",
    "emotional_reactivity",
    "emotional_reactivity_detail",
    "depression_tendency",
    "depression_tendency_detail",
    "social_value_orientation",
    "social_value_orientation_detail",
    "justice_sensitivity",
)


def _is_all_chinese(value: object) -> bool:
    text = "" if value is None else str(value)
    return all("\u4e00" <= ch <= "\u9fff" for ch in text)


def _pinyin_place(value: object, suffix: str) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""
    if not _is_all_chinese(text):
        return text
    if text.endswith(("市", "省")):
        text = text[:-1]
    from pypinyin import Style, pinyin
    words = pinyin(text, style=Style.NORMAL)
    romanized = "".join(word[0].capitalize() if i == 0 else word[0] for i, word in enumerate(words))
    return f"{romanized} {suffix}"


def build_character(row: pd.Series) -> dict:
    city = _pinyin_place(row.get("city", ""), "city")
    province = _pinyin_place(row.get("province", ""), "province")
    basic_information = f"You are a {row['age']}-year-old {str(row['gender']).lower()} from {city}, {province}. "

    autism_tendency = (
        f"Your total score for autism tendency is {row['total_AQ_score']} "
        f"(societal average: {AVERAGES['total_AQ_score']}). Higher scores indicate a higher risk of autism. "
    )
    autism_tendency_detail = (
        "According to the AQ cutoff score, a score above 70 is considered high risk. "
        f"You are in the {row['AQ_2g']} risk group (divided into normal, high) and the {row['AQ_3g']} risk group "
        "(divided into low, medium, high). Your autism tendency sub-dimension scores are as follows: "
        f"Social skills: {row['social_skill_score']} (societal average: {AVERAGES['social_skill_score']}); "
        f"Routine: {row['routine_score']} (societal average: {AVERAGES['routine_score']}); "
        f"Switching ability: {row['switching_score']} (societal average: {AVERAGES['switching_score']}); "
        f"Imagination: {row['imagination_score']} (societal average: {AVERAGES['imagination_score']}); "
        f"Numbers and patterns: {row['numbers_patterns_score']} (societal average: {AVERAGES['numbers_patterns_score']}); "
        f"Social behavior: {row['social_behavior_score']} (societal average: {AVERAGES['social_behavior_score']}). "
    )
    emotional_reactivity = (
        f"Your total score for emotional reactivity is {row['total_ERS_score']} "
        f"(societal average: {AVERAGES['total_ERS_score']}). Higher scores indicate stronger emotional reactivity. "
    )
    emotional_reactivity_detail = (
        "Your emotional reactivity sub-dimension scores are as follows: "
        f"Emotional duration: {row['duration_score']} (societal average: {AVERAGES['duration_score']}); "
        f"Emotional sensitivity: {row['sensitivity_score']} (societal average: {AVERAGES['sensitivity_score']}); "
        f"Emotional intensity: {row['intensity_score']} (societal average: {AVERAGES['intensity_score']}). "
    )
    depression_tendency = (
        f"Your total score for depression tendency is {row['total_CESD_score']} "
        f"(societal average: {AVERAGES['total_CESD_score']}). Higher scores indicate a higher risk of depression. "
    )
    depression_tendency_detail = (
        "According to the CESD cutoff score, a score above 40 is considered high risk. "
        f"You are in the {row['CESD_2g']} risk group (divided into normal, high) and the {row['CESD_3g']} risk group "
        "(divided into low, medium, high). Your depression tendency sub-dimension scores are as follows: "
        f"Positive affect: {row['positive_affect_score']} (societal average: {AVERAGES['positive_affect_score']}); "
        f"Depression: {row['depression_score']} (societal average: {AVERAGES['depression_score']}); "
        f"Interpersonal relationships: {row['interpersonal_score']} (societal average: {AVERAGES['interpersonal_score']}); "
        f"Somatic symptoms: {row['somatic_score']} (societal average: {AVERAGES['somatic_score']}). "
    )

    svo = str(row["SVO"])
    if svo == "prosocial":
        svo_meaning = "prosocial (among prosocial, proself, and undifferentiated), indicating a preference for maximizing joint outcomes and fairness. "
    elif svo == "proself":
        svo_meaning = "proself (among prosocial, proself, and undifferentiated), indicating a preference for maximizing personal outcomes and self-interest. "
    else:
        svo_meaning = "undifferentiated (among prosocial, proself, and undifferentiated), indicating no clear preference for prosocial or proself behavior. "
    social_value_orientation = f"Your social value orientation is classified as {svo_meaning}."
    social_value_orientation_detail = (
        "Your detailed personality choice counts are as follows: "
        f"Prosocial choices: {row['counts[P]']} times (societal average: {AVERAGES['counts[P]']}); "
        f"Individual choices: {row['counts[I]']} times (societal average: {AVERAGES['counts[I]']}); "
        f"Competitive choices: {row['counts[C]']} times (societal average: {AVERAGES['counts[C]']}). "
    )
    justice_sensitivity = (
        f"Your justice sensitivity (observer subscale) score is {row['JSI_score']} "
        f"(societal average: {AVERAGES['JSI_score']}). Higher scores indicate greater sensitivity to injustice."
    )
    return {
        "id": row["id"],
        "basic_information": basic_information,
        "autism_tendency": autism_tendency,
        "autism_tendency_detail": autism_tendency_detail,
        "emotional_reactivity": emotional_reactivity,
        "emotional_reactivity_detail": emotional_reactivity_detail,
        "depression_tendency": depression_tendency,
        "depression_tendency_detail": depression_tendency_detail,
        "social_value_orientation": social_value_orientation,
        "social_value_orientation_detail": social_value_orientation_detail,
        "justice_sensitivity": justice_sensitivity,
    }


def character_to_prompt(character: dict) -> str:
    return " ".join(str(character.get(field, "")).strip() for field in PERSONA_FIELDS if character.get(field)).strip()


def load_human_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    demographics = pd.read_excel(data_dir / "human_demographics.xlsx", sheet_name="n=1017")
    trials = pd.read_excel(data_dir / "human_experiment_trials.xlsx", sheet_name="Sheet1")
    demographics["province"] = demographics["province"].fillna("")
    demographics["city"] = demographics["city"].fillna("")
    return demographics, trials


def subject_character(demographics: pd.DataFrame, subject_index: int) -> dict:
    if not 0 <= subject_index < len(demographics):
        raise IndexError(f"subject_index {subject_index} outside [0, {len(demographics) - 1}]")
    return build_character(demographics.iloc[subject_index])


def subject_trials(trials: pd.DataFrame, subject_index: int, rounds: int = 60) -> list[dict]:
    start, end = rounds * subject_index, rounds * (subject_index + 1)
    block = trials.iloc[start:end][["id", "trial", "amount_of_allocation", "cost_level", "amount_of_cost"]].copy()
    if len(block) != rounds:
        raise ValueError(f"Expected {rounds} trials for subject {subject_index}, found {len(block)}")
    block.insert(0, "index", 0)
    return block.to_dict(orient="records")


def write_subject_inputs(data_dir: Path, output_dir: Path, subjects: Iterable[int], rounds: int = 60) -> None:
    demographics, trials = load_human_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        character = [subject_character(demographics, subject)]
        settings = subject_trials(trials, subject, rounds=rounds)
        (output_dir / f"{subject}_character.json").write_text(json.dumps(character, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / f"{subject}_game_setting_prompt.json").write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
