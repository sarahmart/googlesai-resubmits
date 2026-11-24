"""Check a submission against the hub's requirements before it is pushed.

These are the checks encoded in the hubs' own `hub-config/` — `tasks.json` for
the task space and value types, `validations.yml` for the custom bounds.
"""

import argparse
import io
import pathlib

import pandas as pd
import requests

QUANTILES = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
             0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
COLUMNS = ['reference_date', 'target', 'horizon', 'target_end_date', 'location',
           'output_type', 'output_type_id', 'value']
HUB_LOCATIONS = {
    'flu': 'https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/'
           'main/auxiliary-data/locations.csv',
    'rsv': 'https://raw.githubusercontent.com/CDCgov/rsv-forecast-hub/'
           'main/auxiliary-data/locations.csv',
}
# Taken from each hub's own hub-config/validations.yml. 
# FluSight configures cstm_check_tbl_value_max_popn_frac with
# popn_frac 0.005. RSVHub enforces no population-fraction bound.
POPULATION_FRACTION_CAP = {'flu': 0.005, 'rsv': None}
TARGETS = {'flu': 'wk inc flu hosp', 'rsv': 'wk inc rsv hosp'}


def check(frame, pathogen, reference_date, populations):
  failures = []

  def require(condition, message):
    if not condition:
      failures.append(message)

  require(list(frame.columns) == COLUMNS,
          f'columns are {list(frame.columns)}, expected {COLUMNS}')
  require(set(frame.target.unique()) == {TARGETS[pathogen]},
          f'unexpected targets {sorted(set(frame.target.unique()))}')
  require(set(frame.output_type.unique()) == {'quantile'},
          f'unexpected output types {sorted(set(frame.output_type.unique()))}')
  require((frame.reference_date.astype(str) == reference_date).all(),
          'reference_date column does not match the requested date')

  horizons = sorted(frame.horizon.unique())
  require(horizons == [0, 1, 2, 3], f'horizons are {horizons}, expected [0,1,2,3]')

  # target_end_date = reference_date + horizon * 7 days
  expected_end = (pd.to_datetime(frame.reference_date)
                  + pd.to_timedelta(frame.horizon * 7, unit='D'))
  require((pd.to_datetime(frame.target_end_date) == expected_end).all(),
          'target_end_date is not reference_date + horizon weeks')

  quantiles = sorted(frame.output_type_id.round(3).unique())
  require(quantiles == QUANTILES,
          f'{len(quantiles)} distinct quantiles, expected {len(QUANTILES)}')

  n_locations = frame.location.nunique()
  expected_rows = n_locations * 4 * len(QUANTILES)
  require(len(frame) == expected_rows,
          f'{len(frame)} rows for {n_locations} locations, expected {expected_rows}')
  require(not frame.duplicated(
      ['reference_date', 'horizon', 'location', 'output_type_id']).any(),
          'duplicate (location, horizon, quantile) rows')

  require(frame.value.notna().all(), 'submission contains null values')
  require((frame.value >= 0).all(), 'submission contains negative values')

  ordered = frame.sort_values(['location', 'horizon', 'output_type_id'])
  monotonic = ordered.groupby(['location', 'horizon'])['value'].apply(
      lambda s: s.is_monotonic_increasing)
  require(monotonic.all(),
          f'{int((~monotonic).sum())} (location, horizon) groups are not monotonic')

  merged = frame.merge(populations, on='location', how='left')
  require(merged.population.notna().all(),
          'a submitted location is absent from the hub locations file')
  over_population = merged.value > merged.population
  require(not over_population.any(),
          f'{int(over_population.sum())} values exceed the location population')
  popn_frac = POPULATION_FRACTION_CAP[pathogen]
  if popn_frac is not None:
    over_cap = merged.value > merged.population * popn_frac
    require(not over_cap.any(),
            f'{int(over_cap.sum())} values exceed {popn_frac} x population '
            f'(max ratio {(merged.value / merged.population).max():.6f})')
  return failures


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pathogen', required=True, choices=sorted(TARGETS))
  parser.add_argument('--submission', type=pathlib.Path, required=True)
  parser.add_argument('--reference_date', required=True)
  args = parser.parse_args()

  frame = pd.read_csv(args.submission, dtype={'location': str})
  locations = pd.read_csv(
      io.StringIO(requests.get(HUB_LOCATIONS[args.pathogen], timeout=120).text),
      dtype={'location': str})[['location', 'population']]

  failures = check(frame, args.pathogen, args.reference_date, locations)
  print(f'{args.submission.name}: {len(frame):,} rows, '
        f'{frame.location.nunique()} locations')
  if failures:
    for failure in failures:
      print(f'  FAIL {failure}')
    raise SystemExit(1)
  print('  all checks passed')


if __name__ == '__main__':
  main()
