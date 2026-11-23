# Vendored ERA model archive

`upstream/` is an unmodified copy of `epi_forecasts/{flu_hub,rsv_hub}/model_py/`
from [google-research/google-research](https://github.com/google-research/google-research),
pinned at commit `bb18d9b0cb460fee1a080fa2eeb4109fe3805275` (see `UPSTREAM_SHA`). Licence: Apache-2.0; the original
headers are retained in every file.

Accompanying paper: Martinson et al. (2026), *Prospective multi-pathogen disease
forecasting using autonomous LLM-guided tree search*, [arXiv:2605.16238](https://arxiv.org/abs/2605.16238).

Contents: 35 influenza models, 2 RSV models, and three shared modules per hub
(`constant_defs.py`, `epi_utils.py`, `plotting_utils.py`).

