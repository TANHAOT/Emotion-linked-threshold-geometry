# 01 — Main LLM experiments

Experiment selection is controlled in **one file**:

```text
01_main_llm_experiments/config.py
```

The project-root `.env` contains only credentials / local paths. It no longer contains model, condition, subject, trial, persona, temperature, or worker settings.

## TEST mode

At the top of `config.py`:

```python
TEST = True
```

When `TEST = True`, only `TEST_CONFIG` is used:

```python
TEST_CONFIG = {
    "models": ["deepseek-v3"],
    "conditions": ["emotion"],
    "subjects": "0",
    "trials_per_subject": 10,
    "use_persona": True,
    "temperature": 1.0,
    "workers": 1,
}
```

Edit this single dictionary to choose exactly which model(s), reporting condition(s), participant row indices, number of trials, persona setting, temperature and worker count to test.

Then run:

```bash
python 01_main_llm_experiments/run.py
```

To validate the selection without using the API:

```bash
python 01_main_llm_experiments/run.py --dry-run
```

## Formal mode

Set:

```python
TEST = False
```

The formal experiment blocks are defined in `FORMAL_EXPERIMENTS`:

- `main_emotion`: 22 models, emotion report, persona, temperature 1, 1,017 IDs × 60 trials.
- `reporting_controls`: the four control models under `no_report`, `unfairness`, and `intentionality`. 
- `persona_control`: the four control models, emotion report, without persona information.
- `temperature_control`: the seven temperature-control models at temperature 0. 

`FORMAL_RUN` specifies which formal blocks are executed:

```python
FORMAL_RUN = [
    "main_emotion",
    "reporting_controls",
    "persona_control",
    "temperature_control",
]
```

To run only the reporting controls, change it to:

```python
FORMAL_RUN = ["reporting_controls"]
```

You can also temporarily select a formal block from the command line without editing the file:

```bash
python 01_main_llm_experiments/run.py --formal --experiment reporting_controls
```

## API configuration

Project-root `.env`:

```text
LLM_API_KEY=your_real_key
LLM_BASE_URL=https://new.midsummer.work/v1
```

## Output

All Part 01 results are stored under one `result/` directory:

```text
01_main_llm_experiments/
└── result/
    ├── persona_emotion_1.0_claude-3-7-sonnet-20250219/
    │   ├── output_0.tsv
    │   ├── output_1.tsv
    │   └── ...
    ├── persona_unfairness_1.0_deepseek-v3/
    ├── persona_intentionality_1.0_deepseek-v3/
    ├── persona_no_report_1.0_deepseek-v3/
    ├── nopersona_emotion_1.0_deepseek-v3/
    └── persona_emotion_0.0_deepseek-v3/
```

The naming rule is:

```text
<persona|nopersona>_<condition>_<temperature>_<model>
```

### Incremental resume

`config.py` contains shared network-safety settings:

```python
RUN_OPTIONS = {
    "resume": True,
    "max_retries": 5,
    "retry_backoff_seconds": 2.0,
}
```

With `resume=True`, the runner checks the existing TSV for each participant under the same condition and model before processing that participant. If any saved response is invalid (`valid=False` or a missing validity flag), trial numbers are missing or duplicated, or the TSV cannot be parsed, the participant's existing results are deleted and all configured trials for that participant are rerun. Complete and valid results are preserved, so unaffected participants are not rerun. If a participant's results are incomplete but all saved records are valid, the runner will also rerun all trials.

These checks occur when each participant is started. Invalid responses generated during the current run are saved and trigger a full participant rerun on the next launch, rather than an immediate restart. Transient connection, timeout, rate-limit and server-side errors are handled separately through retries with exponential backoff, subject to the configured retry limit.

Before execution, the runner prints the selected models and conditions, participant indices and human IDs, trials per participant, persona setting, temperature, worker count, expected API-call count, and retry/resume settings.
