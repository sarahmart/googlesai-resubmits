"""Pin the ensemble rule against forecasts actually submitted in 2025-26.

Network access is required. A download failure fails the test rather than
skipping it: an unverifiable ensemble rule is not a passing condition.
"""

import io
import pathlib
import sys

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'pipeline'))
import create_ensemble  # noqa: E402

REFERENCE_DATE = '2026-02-07'
COMPONENTS = ['Google_SAI-Adapted_9', 'Google_SAI-Adapted_21',
              'Google_SAI-Hybrid_1', 'Google_SAI-Hybrid_5',
              'Google_SAI-Hybrid_6']
GOOGLEHUB = ('https://raw.githubusercontent.com/google-research/google-research/'
             'master/epi_forecasts/flu_hub/model_output')
FLUSIGHT = ('https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/'
            'main/model-output/Google_SAI-FluEns')
KEY = ['location', 'horizon', 'output_type_id']


def _fetch(url):
  response = requests.get(url, timeout=120)
  response.raise_for_status()
  return pd.read_csv(io.StringIO(response.text), dtype={'location': str})


def _quantiles(frame):
  frame = frame[frame.output_type == 'quantile'].copy()
  frame['output_type_id'] = frame.output_type_id.astype(float).round(3)
  return frame.set_index(KEY)['value'].sort_index()


@pytest.fixture(scope='module')
def submitted():
  return _quantiles(_fetch(f'{FLUSIGHT}/{REFERENCE_DATE}-Google_SAI-FluEns.csv'))


@pytest.fixture(scope='module')
def components():
  return {m: _fetch(f'{GOOGLEHUB}/{m}/{REFERENCE_DATE}-{m}.csv')
          for m in COMPONENTS}


def test_equal_weight_quantile_mean_reproduces_submission(submitted, components):
  """The submitted ensemble is the unweighted mean of its components."""
  frames = {}
  for name, frame in components.items():
    frame = frame[frame.output_type == 'quantile'].copy()
    frame['output_type_id'] = frame.output_type_id.astype(float).round(3)
    frames[name] = frame

  ensemble, _ = create_ensemble.build_ensemble(frames)
  rebuilt = _quantiles(ensemble)

  assert rebuilt.index.equals(submitted.index), 'task space differs'
  assert (rebuilt - submitted).abs().max() < 1e-6


def test_rule_is_a_mean_not_a_median(submitted, components):
  """A median over five components would land on a component value every time."""
  stacked = pd.DataFrame({n: _quantiles(f) for n, f in components.items()})
  coincides = (stacked.sub(submitted, axis=0).abs() < 1e-9).any(axis=1)
  assert coincides.mean() < 0.10, (
      'the submitted value coincides with a component value too often for the '
      'rule to be a mean')
