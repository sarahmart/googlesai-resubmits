"""Combine component submissions into the Google_SAI ensemble forecast.

The aggregation is an equal-weight mean of each quantile across components,
grouped on (reference_date, horizon, location, target_end_date), followed by
isotonic regression to enforce non-negative, monotonically increasing quantiles.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import yaml
from sklearn import isotonic

GROUPING = ['reference_date', 'target', 'horizon', 'target_end_date', 'location']


def load_config(repo_root, pathogen):
  config = yaml.safe_load(
      (repo_root / 'config' / f'{pathogen}.yaml').read_text())
  ensemble = config['ensemble']
  if ensemble['method'] != 'quantile_mean':
    raise ValueError(f'unsupported ensemble method {ensemble["method"]!r}')

  # Two shapes, because the pathogens differ
  by_phase = ensemble.get('components_by_phase')
  flat = ensemble.get('components')
  if (by_phase is None) == (flat is None):
    raise ValueError(
        f'config/{pathogen}.yaml must set exactly one of `components` or'
        ' `components_by_phase` under `ensemble`.')
  if flat is not None:
    return config['hub'], list(flat), 'all phases'

  phase = ensemble['phase']
  if phase not in by_phase:
    raise ValueError(
        f'phase {phase!r} has no component list; have {sorted(by_phase)}')
  return config['hub'], by_phase[phase], phase


def load_components(submission_dir, components, reference_date):
  """Read each component's submission, failing loudly on anything unexpected."""
  frames, missing = {}, []
  for model in components:
    # Models write `<date>_<model>.csv`; hubs use `<date>-<model>.csv`.
    candidates = [submission_dir / f'{reference_date}_{model}.csv',
                  submission_dir / f'{reference_date}-{model}.csv']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
      missing.append(model)
      continue
    frame = pd.read_csv(path, dtype={'location': str})
    frame = frame[frame.output_type == 'quantile'].copy()
    if frame.empty:
      raise ValueError(f'{path} contains no quantile rows')
    frame['output_type_id'] = frame.output_type_id.astype(float).round(3)
    frames[model] = frame
  return frames, missing


def ensure_monotonicity(group):
  regressor = isotonic.IsotonicRegression(
      y_min=0, increasing=True, out_of_bounds='clip')
  group = group.sort_values('output_type_id')
  group['value'] = regressor.fit_transform(
      group.output_type_id.values, group.value.values)
  return group


def build_ensemble(frames):
  """Equal-weight mean of component quantiles, then enforce monotonicity."""
  combined = pd.concat(frames.values(), ignore_index=True)
  counts = combined.groupby(GROUPING + ['output_type_id']).size()
  if counts.nunique() != 1:
    raise ValueError(
        'components cover different task sets: each (location, horizon,'
        f' quantile) must appear {counts.max()} times, found {sorted(counts.unique())}.'
        ' Averaging these would weight tasks unequally.')

  mean = (combined.groupby(GROUPING + ['output_type_id'], as_index=False)
          ['value'].mean())
  mean['output_type'] = 'quantile'
  # Grouped explicitly rather than with groupby().apply(): pandas >= 3 drops the
  # grouping columns from the frames passed to apply.
  monotonic = pd.concat(
      [ensure_monotonicity(group) for _, group in mean.groupby(GROUPING)],
      ignore_index=True)
  shifted = float(np.abs(
      monotonic.sort_values(GROUPING + ['output_type_id']).value.values
      - mean.sort_values(GROUPING + ['output_type_id']).value.values).sum())
  return monotonic[GROUPING[:1] + ['target', 'horizon', 'target_end_date',
                                   'location', 'output_type', 'output_type_id',
                                   'value']], shifted


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  root = pathlib.Path(__file__).resolve().parent.parent
  parser.add_argument('--pathogen', required=True, choices=['flu', 'rsv'])
  parser.add_argument('--reference_date', required=True)
  parser.add_argument('--submission_dir', type=pathlib.Path, required=True)
  parser.add_argument('--out_dir', type=pathlib.Path, required=True)
  parser.add_argument('--repo_root', type=pathlib.Path, default=root)
  parser.add_argument('--min_components', type=int, default=0,
                      help='0 means every configured component is required')
  args = parser.parse_args()

  hub, components, phase = load_config(args.repo_root, args.pathogen)
  frames, missing = load_components(
      args.submission_dir, components, args.reference_date)
  floor = args.min_components or len(components)
  if len(frames) < floor:
    raise RuntimeError(
        f'only {len(frames)} of {len(components)} components produced a'
        f' submission for {args.reference_date} (missing: {missing});'
        f' --min_components is {floor}.')
  if missing:
    print(f'WARNING: proceeding without {missing} '
          f'({len(frames)}/{len(components)} components, floor {floor})')

  ensemble, shifted = build_ensemble(frames)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  out = args.out_dir / f'{args.reference_date}-{hub["model_id"]}.csv'
  ensemble.to_csv(out, index=False)
  print(f'{hub["model_id"]} [{phase}] {len(frames)} components, '
        f'{len(ensemble):,} rows')
  print(f'  isotonic adjustment, total absolute shift: {shifted:.4f}')
  print(f'  wrote {out}')


if __name__ == '__main__':
  main()
