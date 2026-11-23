"""Run one patched ERA model and write its CDC-format submission.

Executed as a subprocess, one per model, so that a crash, hang or memory blow-up
in a single model cannot take down a weekly run. The caller enforces the
timeout; this script just runs the model and reports how long it took.

The model's own `main()` is invoked unchanged: it reads the configured inputs,
refits on data up to `reference_date - 1 week`, forecasts all four horizons for
all 52 jurisdictions, and writes `<reference_date>-<model>.csv`.
"""

import argparse
import datetime
import importlib.util
import os
import pathlib
import sys
import time

import pandas as pd

HUB_DIRS = {'flu': 'flu_hub', 'rsv': 'rsv_hub'}


def check_data_freshness(input_dir, reference_date, allow_stale=False):
  """Refuse to forecast from data staler than the reference date assumes.

  A reference date R is a Saturday, and the most recent complete surveillance
  week ends on R - 7 days. If the inputs stop earlier than that - typically
  because the job ran before the Wednesday NHSN release - the models do not
  fail. They silently produce forecasts that are wrong by an order of
  magnitude, and those forecasts pass every schema and hub validation check.

  Measured on 2026-11-18 with flu Adapted_8: with the last observation one week
  before the reference date the horizon-0 national median was 10,350, against
  an observed 9,393 and rising. With the same data one week staler it was 418.

  Data extending beyond R - 7 is a backtest, not an error: the run then sees
  revised values the real-time forecast could not have, which flatters scores
  but does not break alignment.
  """
  dataset = pd.read_csv(pathlib.Path(input_dir) / 'dataset.csv',
                        usecols=['target_end_date'])
  latest = pd.to_datetime(dataset.target_end_date).max().date()
  reference = datetime.date.fromisoformat(reference_date)
  required = reference - datetime.timedelta(days=7)

  if latest < required:
    message = (
        f'inputs end {latest} but reference date {reference} requires data '
        f'through {required} ({(required - latest).days // 7} week(s) missing). '
        'Forecasting from this would silently produce order-of-magnitude '
        'errors. Refresh the inputs after the Wednesday NHSN release.')
    if not allow_stale:
      raise RuntimeError(message)
    print(f'WARNING (--allow_stale_data): {message}')
  elif latest > required:
    print(f'NOTE: inputs extend to {latest}, beyond the {required} a real-time '
          f'run would have seen. This is a backtest on revised data.')
  return latest


def run(repo_root, hub, model, reference_date, input_dir, output_dir,
        run_validation=False, allow_stale=False):
  check_data_freshness(input_dir, reference_date, allow_stale)
  hub_dir = repo_root / 'models' / 'patched' / HUB_DIRS[hub]
  module_path = hub_dir / f'{model}.py'
  if not module_path.is_file():
    raise FileNotFoundError(f'no such model: {module_path}')

  os.environ.update({
      'EPI_INPUT_DIR': str(input_dir),
      'EPI_OUTPUT_DIR': str(output_dir),
      'EPI_REFERENCE_DATE': reference_date,
      'EPI_RUN_VALIDATION': '1' if run_validation else '0',
      'EPI_PLOTTING': '0',
  })
  sys.path.insert(0, str(hub_dir))

  started = time.monotonic()
  spec = importlib.util.spec_from_file_location(model.replace('-', '_'),
                                                module_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)          # reads inputs at import time
  module.main([])                          # argv is discarded by the model
  elapsed = time.monotonic() - started

  expected = pathlib.Path(output_dir) / f'{reference_date}_{model}.csv'
  if not expected.is_file():
    raise RuntimeError(
        f'{model} finished but wrote no submission at {expected}')
  return elapsed, expected


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  root = pathlib.Path(__file__).resolve().parent.parent
  parser.add_argument('--hub', choices=sorted(HUB_DIRS), required=True)
  parser.add_argument('--model', required=True)
  parser.add_argument('--reference_date', required=True)
  parser.add_argument('--input_dir', type=pathlib.Path, required=True)
  parser.add_argument('--output_dir', type=pathlib.Path, required=True)
  parser.add_argument('--repo_root', type=pathlib.Path, default=root)
  parser.add_argument('--run_validation', action='store_true')
  parser.add_argument('--allow_stale_data', action='store_true',
                      help='downgrade the freshness precondition to a warning')
  args = parser.parse_args()

  elapsed, path = run(args.repo_root, args.hub, args.model,
                      args.reference_date, args.input_dir, args.output_dir,
                      args.run_validation, args.allow_stale_data)
  print(f'OK {args.model} {elapsed:.1f}s -> {path}')


if __name__ == '__main__':
  main()
