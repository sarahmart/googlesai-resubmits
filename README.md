# epihub-resubmits

An external rebuild of the weekly pipeline behind the `Google_SAI` influenza and
RSV ensemble submissions to CDC's [FluSight](https://github.com/cdcepi/FluSight-forecast-hub)
and [RSVHub](https://github.com/CDCgov/rsv-forecast-hub), reproducing Martinson
et al. (2026), [arXiv:2605.16238](https://arxiv.org/abs/2605.16238).

[COVIDHub is out of scope: that submission is already live and not treated here.]

The published GoogleHub archive supplies the forecasting models but not the pipeline
around them. This repository supplies it, from public sources only.

## Layout

| Path | Purpose |
|---|---|
| `models/upstream/` | Pinned, unmodified copy of the ERA model archive (`models/UPSTREAM_SHA`) |
| `models/patched/` | Generated runnable models — gitignored, never hand-edit |
| `pipeline/` | Data preparation, model runner, ensemble, validation |
| `config/` | Per-hub settings and the ensemble component list |
| `tests/` | Regression tests |

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python pipeline/patch_models.py
```

`requirements.txt` pins every dependency exactly, `pandas==2.3.3` among them 
(required for the originally-generated models).

## Weekly run

```bash
# 1. Inputs. Run only AFTER the Wednesday NHSN release.
python pipeline/build_inputs.py --pathogen flu --out_dir data/inputs/flu
python pipeline/build_ilinet.py --out_dir data/inputs/flu \
    --locations data/inputs/flu/locations.csv

# 2. One subprocess per component model.
python pipeline/run_model.py --hub flu --model Google_SAI-Hybrid_3 \
    --reference_date 2026-11-21 \
    --input_dir data/inputs/flu --output_dir data/outputs/flu

# 3. Ensemble the components named in config/flu.yaml for the current phase.
python pipeline/create_ensemble.py --pathogen flu --reference_date 2026-11-21 \
    --submission_dir data/outputs/flu --out_dir data/ensemble/flu

# 4. Gate before pushing.
python pipeline/validate_submission.py --pathogen flu --reference_date 2026-11-21 \
    --submission data/ensemble/flu/2026-11-21-Google_SAI-FluEns.csv
```

## Runtime configuration

Patched models read their configuration from the environment and raise on
anything missing, so a mis-wired run fails at startup rather than forecasting
from the wrong inputs.

| Variable | Meaning |
|---|---|
| `EPI_INPUT_DIR` | Directory holding `dataset.csv`, `locations.csv`, the ILINet extracts |
| `EPI_OUTPUT_DIR` | Where forecast CSVs are written |
| `EPI_REFERENCE_DATE` | Forecast reference date; must be a Saturday |
| `EPI_RUN_VALIDATION` | `1` to run each model's in-built rolling validation (default off) |
| `EPI_PLOTTING` | `1` to emit diagnostic plots (default off) |

