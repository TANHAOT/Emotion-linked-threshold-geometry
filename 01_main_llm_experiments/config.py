from __future__ import annotations

from project_core.models import CONTROL_4, MAIN_22, TEMPERATURE_7

# -----------------------------------------------------------------------------
# Switch between a cheap TEST run and the formal experiment configuration.
# Change only this flag for normal use.
# -----------------------------------------------------------------------------
TEST = True

# API/network resilience shared by TEST and formal runs.
RUN_OPTIONS = {
    "resume": True,
    "max_retries": 5,
    "retry_backoff_seconds": 2.0,
}

# -----------------------------------------------------------------------------
# TEST configuration
# Edit this one dictionary when TEST = True.
# `subjects` uses 0-based row indices from human_demographics.xlsx.
# Examples: "0", "0-9", "0,3,7", "all".
# -----------------------------------------------------------------------------
TEST_CONFIG = {
    "models": ["deepseek-v3"],
    "conditions": ["emotion"],
    "subjects": "0",
    "trials_per_subject": 10,
    "use_persona": True,
    "temperature": 1.0,
    "workers": 1,
}

# -----------------------------------------------------------------------------
# Formal experiment configuration used when TEST = False.
# These blocks reproduce the experiments described in the Supplementary
# Materials, excluding Section 4.6 as requested.
#
# main_emotion:
#   22 models, emotion report, persona, temperature = 1.
# reporting_controls:
#   the three additional report conditions for the four control models.
#   The emotion-report baseline is already produced by main_emotion.
# persona_control:
#   no-persona emotion-report condition for the same four control models.
# temperature_control:
#   temperature = 0 for the seven temperature-control models. The temperature = 1
#   baseline is already produced by main_emotion.
# -----------------------------------------------------------------------------
FORMAL_EXPERIMENTS = {
    "main_emotion": {
        "models": MAIN_22,
        "conditions": ["emotion"],
        "subjects": "all",
        "trials_per_subject": 60,
        "use_persona": True,
        "temperature": 1.0,
        "workers": 16,
    },
    "reporting_controls": {
        "models": CONTROL_4,
        "conditions": ["no_report", "unfairness", "intentionality"],
        "subjects": "all",
        "trials_per_subject": 60,
        "use_persona": True,
        "temperature": 1.0,
        "workers": 16,
    },
    "persona_control": {
        "models": CONTROL_4,
        "conditions": ["emotion"],
        "subjects": "all",
        "trials_per_subject": 60,
        "use_persona": False,
        "temperature": 1.0,
        "workers": 16,
    },
    "temperature_control": {
        "models": TEMPERATURE_7,
        "conditions": ["emotion"],
        "subjects": "all",
        "trials_per_subject": 60,
        "use_persona": True,
        "temperature": 0.0,
        "workers": 16,
    },
}

# When TEST = False, run only the named formal blocks below.
# Remove entries if you want to execute only part of the formal experiment.
FORMAL_RUN = [
    "main_emotion",
    "reporting_controls",
    "persona_control",
    "temperature_control",
]
