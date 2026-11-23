"""Build the input files the patched ERA models read from EPI_INPUT_DIR.

Every archived model opens with bare `read_csv` calls against files the model
archive does not produce. Upstream has since published its own preparation step
at `epi_forecasts/pipeline/data_prep/`; this module does the same job from
public sources only, and was checked against it on 2026-11-23. The cleaning and
interpolation agree step for step. Three differences are deliberate: NHSN is
fetched over the Socrata API rather than read from a manually downloaded export,
jurisdictions are joined on FIPS code rather than on name, and the duplicate-row
case upstream records in a comment is asserted here.

Per Methods 1 of the paper, preparation is: extract jurisdiction-level weekly
admission counts, clean non-numeric entries, and linearly interpolate missing
weeks to a complete grid across all 52 jurisdictions.
"""

import argparse
import io
import json
import pathlib

import pandas as pd
import requests

# Preliminary NHSN Weekly Hospital Respiratory Data. The finalised series
# (ua7e-t2fy) lags by a week and would leave no data for the reference week.
NHSN_RESOURCE = 'https://data.cdc.gov/resource/mpgq-jmmr.csv'
NHSN_METADATA = 'https://data.cdc.gov/api/views/mpgq-jmmr.json'
HUB_LOCATIONS = {
    'flu': 'https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/'
           'main/auxiliary-data/locations.csv',
    'rsv': 'https://raw.githubusercontent.com/CDCgov/rsv-forecast-hub/'
           'main/auxiliary-data/locations.csv',
}
ADMISSION_COLUMNS = {
    'flu': 'totalconfflunewadm',
    'covid': 'totalconfc19newadm',
    'rsv': 'totalconfrsvnewadm',
}
TARGET_NAMES = {
    'flu': 'Total Influenza Admissions',
    'covid': 'Total COVID-19 Admissions',
    'rsv': 'Total RSV Admissions',
}
# Each pathogen's dataset.csv carries these admission series. RSV models use
# concurrent influenza and COVID-19 admissions as auxiliary inputs, and the
# RSV main() also reads the influenza column directly.
DATASET_COLUMNS = {'flu': ('flu',), 'rsv': ('rsv', 'flu', 'covid')}
IMPLEMENTED_PATHOGENS = ('flu',)
REQUEST_TIMEOUT = 120


def _get(url, params=None):
  response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
  response.raise_for_status()
  return response


def load_locations(pathogen, required_locations):
  """The hub jurisdiction table, unfiltered, plus the 52-row forecast subset.

  The file written for the models must keep the hub's own layout, including the
  `US` row. Each model's `main()` does its own
  `locations['location'].isin(REQUIRED_CDC_LOCATIONS)` against *string* FIPS
  codes after a plain `pd.read_csv`. With `US` present the column is object
  dtype and the filter works; drop it and the column is pure numeric, so `01`
  parses as integer `1`, the filter matches nothing, and the model builds an
  empty forecast scaffold without ever raising.

  `required_locations` is `constant_defs.REQUIRED_CDC_LOCATIONS`: the 50 states
  plus DC and Puerto Rico, with the national aggregate deliberately absent.
  """
  full = pd.read_csv(io.StringIO(_get(HUB_LOCATIONS[pathogen]).text),
                     dtype={'location': str})
  missing = set(required_locations) - set(full.location)
  if missing:
    raise ValueError(f'hub locations.csv is missing FIPS codes {sorted(missing)}')
  subset = full[full.location.isin(required_locations)].copy()
  if len(subset) != len(required_locations):
    raise ValueError(
        f'expected {len(required_locations)} jurisdictions, got {len(subset)}')
  for column in ('abbreviation', 'location_name', 'population'):
    if subset[column].isna().any():
      raise ValueError(f'hub locations.csv has nulls in {column}')
  return full, subset[['location', 'abbreviation', 'location_name', 'population']]


def nhsn_provenance():
  """Dataset vintage, so a submission can be traced to the data it used."""
  metadata = _get(NHSN_METADATA).json()
  return {
      'resource': NHSN_RESOURCE,
      'name': metadata.get('name'),
      'rows_updated_at': pd.Timestamp(
          metadata['rowsUpdatedAt'], unit='s', tz='UTC').isoformat(),
      'fetched_at': pd.Timestamp.now(tz='UTC').isoformat(),
  }


def fetch_admissions(locations):
  """Weekly confirmed admissions per jurisdiction for all three pathogens.

  Socrata caps a response at `$limit` and returns the truncated page without
  error, so the row count is checked against a server-side count first: a
  partial download would otherwise look like a short time series and quietly
  degrade every forecast.
  """
  columns = ['weekendingdate', 'jurisdiction'] + list(ADMISSION_COLUMNS.values())
  expected = int(_get(NHSN_RESOURCE.replace('.csv', '.json'),
                      params={'$select': 'count(1)'}).json()[0]['count_1'])
  frame = pd.read_csv(io.StringIO(_get(NHSN_RESOURCE, params={
      '$select': ','.join(columns),
      '$limit': str(expected + 1),
  }).text))
  if len(frame) != expected:
    raise ValueError(
        f'NHSN returned {len(frame):,} rows but reports {expected:,};'
        ' the response was truncated or the dataset changed mid-fetch.')
  frame['target_end_date'] = pd.to_datetime(frame.weekendingdate).dt.date

  known = set(locations.abbreviation)
  frame = frame[frame.jurisdiction.isin(known)].copy()
  covered = set(frame.jurisdiction)
  if covered != known:
    raise ValueError(
        'NHSN is missing jurisdictions present in the hub locations file: '
        f'{sorted(known - covered)}')
  return frame.merge(locations, left_on='jurisdiction', right_on='abbreviation')


def build_grid(admissions, locations, pathogen, start_date):
  """Complete (jurisdiction x week) grid with missing weeks interpolated.

  The grid is imposed first and interpolation applied within each jurisdiction
  afterwards, so a week absent for one state is filled from that state's own
  neighbours rather than from the panel mean.
  """
  frame = admissions[admissions.target_end_date >= start_date]
  observed = sorted(frame.target_end_date.unique())
  if not observed:
    raise ValueError(f'no NHSN weeks at or after {start_date}')

  # Impose the complete Saturday axis rather than only the weeks that happen to
  # be present, matching the upstream preprocessing. Taking the observed weeks
  # gives the same answer whenever the series is complete, but if NHSN ever
  # drops a week for every jurisdiction at once it would silently close the gap
  # and shift every lag and week-over-week difference the models compute. Built
  # this way the absent week is created and interpolated like any other.
  weeks = [d.date() for d in
           pd.date_range(observed[0], observed[-1], freq='W-SAT')]
  off_axis = sorted(set(observed) - set(weeks))
  if off_axis:
    raise ValueError(
        f'{len(off_axis)} NHSN week(s) do not fall on the Saturday grid and'
        f' would be dropped from it, e.g. {off_axis[:3]}. Check whether the'
        ' series convention has changed.')

  index = pd.MultiIndex.from_product(
      [locations.location.tolist(), weeks], names=['location', 'target_end_date'])
  grid = pd.DataFrame(index=index).reset_index()
  grid = grid.merge(
      frame[['location', 'target_end_date'] + list(ADMISSION_COLUMNS.values())],
      on=['location', 'target_end_date'], how='left')

  out_columns = []
  for name in DATASET_COLUMNS[pathogen]:
    source, target = ADMISSION_COLUMNS[name], TARGET_NAMES[name]
    # Non-numeric entries (suppression markers, thousands separators) become
    # NaN, then interpolate within each jurisdiction.
    values = pd.to_numeric(
        grid[source].astype(str).str.replace(',', '', regex=False),
        errors='coerce')
    filled = (values.groupby(grid.location)
              .transform(lambda s: s.interpolate(limit_direction='both')))
    # Interpolation cannot fill a jurisdiction that never reported this series
    # at all. Diagnose that here: the cast below would otherwise raise
    # IntCastingNaNError, which names neither the column nor the jurisdiction.
    if filled.isna().any():
      empty = sorted(grid.loc[filled.isna(), 'location'].unique())
      raise ValueError(
          f'{int(filled.isna().sum())} cells of {target!r} are still missing'
          f' after interpolation; jurisdiction(s) {empty[:5]} appear to have no'
          ' data at all for this period.')
    # Truncate to integer.
    grid[target] = filled.astype('int64')
    out_columns.append(target)

  grid = grid.merge(locations, on='location', how='left')
  grid['location'] = grid.location.astype(int)
  result = grid[['target_end_date', 'location', 'location_name', 'population']
                + out_columns].sort_values(['target_end_date', 'location'])

  expected = len(locations) * len(weeks)
  if len(result) != expected:
    raise ValueError(f'grid has {len(result)} rows, expected {expected}')
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pathogen', choices=IMPLEMENTED_PATHOGENS, required=True)
  parser.add_argument('--out_dir', type=pathlib.Path, required=True,
                      help='becomes EPI_INPUT_DIR for the patched models')
  parser.add_argument('--start_date', default='2020-10-01',
                      help='first NHSN week to retain')
  args = parser.parse_args()

  import constant_defs
  required = constant_defs.REQUIRED_CDC_LOCATIONS
  if constant_defs.TARGET_STR != TARGET_NAMES[args.pathogen]:
    raise ValueError(
        f'constant_defs on sys.path is for {constant_defs.TARGET_STR!r}, not'
        f' {TARGET_NAMES[args.pathogen]!r}; point PYTHONPATH at the right hub.')

  hub_locations, locations = load_locations(args.pathogen, required)
  admissions = fetch_admissions(locations)
  dataset = build_grid(admissions, locations, args.pathogen,
                       pd.to_datetime(args.start_date).date())

  args.out_dir.mkdir(parents=True, exist_ok=True)
  # The hub's own file, US row included - see load_locations for why.
  hub_locations.to_csv(args.out_dir / 'locations.csv', index=False)
  dataset.to_csv(args.out_dir / 'dataset.csv', index=False)
  weeks = sorted(dataset.target_end_date.unique())
  provenance = nhsn_provenance()
  provenance.update({'pathogen': args.pathogen, 'rows': len(dataset),
                     'jurisdictions': len(locations),
                     'first_week': str(weeks[0]), 'last_week': str(weeks[-1])})
  (args.out_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
  print(f'{args.pathogen}: {len(dataset):,} rows, {len(locations)} jurisdictions, '
        f'{len(weeks)} weeks {weeks[0]}..{weeks[-1]}')
  print(f'  wrote {args.out_dir / "locations.csv"}')
  print(f'  wrote {args.out_dir / "dataset.csv"}')
  print(f'  NHSN vintage: {provenance["rows_updated_at"]}')


if __name__ == '__main__':
  main()
