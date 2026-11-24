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
import sys

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

# NSSP emergency-department visit trajectories, the source of the
# `percent_visits_rsv` column the RSV models read. 
NSSP_RESOURCE = 'https://data.cdc.gov/resource/rdmq-nq56.csv'
NSSP_METADATA = 'https://data.cdc.gov/api/views/rdmq-nq56.json'
NSSP_COLUMN = 'percent_visits_rsv'
# Jurisdiction rows, as opposed to the sub-state regions in the same dataset.
NSSP_WHERE = "county = 'All'"
IMPLEMENTED_PATHOGENS = ('flu', 'rsv')
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


def fetch_nssp(locations):
  """NSSP emergency-department RSV visit percentages, one row per jurisdiction."""
  expected = int(_get(NSSP_RESOURCE.replace('.csv', '.json'),
                      params={'$select': 'count(1)',
                              '$where': NSSP_WHERE}).json()[0]['count_1'])
  frame = pd.read_csv(io.StringIO(_get(NSSP_RESOURCE, params={
      '$select': f'week_end,geography,{NSSP_COLUMN}',
      '$where': NSSP_WHERE,
      '$limit': str(expected + 1),
  }).text))
  if len(frame) != expected:
    raise ValueError(
        f'NSSP returned {len(frame):,} rows but reports {expected:,};'
        ' the response was truncated or the dataset changed mid-fetch.')

  frame['target_end_date'] = pd.to_datetime(frame.week_end).dt.date
  days = set(pd.to_datetime(frame.week_end).dt.dayofweek)
  if days != {5}:
    raise ValueError(
        f'NSSP week_end values are not all Saturdays (weekdays seen: {days});'
        ' they would not align with the NHSN weekly grid.')

  # Upstream joins NSSP `geography` to the hub's `location_name`.
  merged = frame.merge(locations[['location', 'location_name']],
                       left_on='geography', right_on='location_name',
                       how='inner')
  if merged.duplicated(['location', 'target_end_date']).any():
    n = int(merged.duplicated(['location', 'target_end_date']).sum())
    raise ValueError(f'NSSP has {n} duplicate (jurisdiction, week) rows')
  return merged[['location', 'target_end_date', NSSP_COLUMN]]


def attach_nssp(dataset, locations):
  """Add the NSSP column to the NHSN grid, left so the grid stays intact.

  Upstream outer-joins. A left join is equivalent while the two series cover the
  same weeks, and it keeps the (jurisdiction x week) grid this module asserts on
  from being silently extended by an NSSP-only week.

  Jurisdictions absent from NSSP are returned rather than raised on: NSSP
  publishes no Puerto Rico, which the RSV hub nonetheless solicits.
  """
  nssp = fetch_nssp(locations)
  # build_grid leaves `location` as an integer FIPS code; the hub file and so
  # the NSSP join key are zero-padded strings.
  nssp = nssp.assign(location=nssp.location.astype(int))
  before = len(dataset)
  merged = dataset.merge(nssp, on=['location', 'target_end_date'], how='left')
  if len(merged) != before:
    raise ValueError(
        f'attaching NSSP changed the row count from {before:,} to'
        f' {len(merged):,}; the join is not one-to-one.')

  covered = set(nssp.location)
  absent = sorted(int(x) for x in set(dataset.location) - covered)
  weeks_with_data = (merged.loc[merged[NSSP_COLUMN].notna(), 'target_end_date']
                     .nunique())
  return merged, absent, weeks_with_data


HUB_DIRS = {'flu': 'flu_hub', 'rsv': 'rsv_hub'}


def load_constant_defs(repo_root, pathogen):
  """The hub's own `constant_defs`, so the jurisdiction list has one source.

  Resolved from the patched tree here rather than left to the caller's
  PYTHONPATH, matching `run_model.py`: the required-locations list and the
  target name must come from the same module the models themselves import, or
  the inputs can disagree with the code that reads them.
  """
  hub_dir = repo_root / 'models' / 'patched' / HUB_DIRS[pathogen]
  if not (hub_dir / 'constant_defs.py').is_file():
    raise FileNotFoundError(
        f'{hub_dir / "constant_defs.py"} not found. Run'
        ' `python pipeline/patch_models.py` first to generate models/patched/.')
  sys.path.insert(0, str(hub_dir))
  import constant_defs
  if constant_defs.TARGET_STR != TARGET_NAMES[pathogen]:
    raise ValueError(
        f'{hub_dir} supplies constant_defs for {constant_defs.TARGET_STR!r},'
        f' not {TARGET_NAMES[pathogen]!r}.')
  return constant_defs


def main():
  root = pathlib.Path(__file__).resolve().parent.parent
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pathogen', choices=IMPLEMENTED_PATHOGENS, required=True)
  parser.add_argument('--out_dir', type=pathlib.Path, required=True,
                      help='becomes EPI_INPUT_DIR for the patched models')
  parser.add_argument('--start_date', default='2020-10-01',
                      help='first NHSN week to retain')
  parser.add_argument('--repo_root', type=pathlib.Path, default=root)
  args = parser.parse_args()

  required = load_constant_defs(args.repo_root,
                                args.pathogen).REQUIRED_CDC_LOCATIONS

  hub_locations, locations = load_locations(args.pathogen, required)
  admissions = fetch_admissions(locations)
  dataset = build_grid(admissions, locations, args.pathogen,
                       pd.to_datetime(args.start_date).date())

  nssp = None
  if args.pathogen == 'rsv':
    dataset, absent, nssp_weeks = attach_nssp(dataset, locations)
    nssp = {'resource': NSSP_RESOURCE, 'column': NSSP_COLUMN,
            'weeks_with_values': nssp_weeks,
            'jurisdictions_without_nssp': absent}

  args.out_dir.mkdir(parents=True, exist_ok=True)
  # The hub's own file, US row included - see load_locations for why.
  hub_locations.to_csv(args.out_dir / 'locations.csv', index=False)
  dataset.to_csv(args.out_dir / 'dataset.csv', index=False)
  weeks = sorted(dataset.target_end_date.unique())
  provenance = nhsn_provenance()
  provenance.update({'pathogen': args.pathogen, 'rows': len(dataset),
                     'jurisdictions': len(locations),
                     'first_week': str(weeks[0]), 'last_week': str(weeks[-1])})
  if nssp is not None:
    provenance['nssp'] = nssp
  (args.out_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
  print(f'{args.pathogen}: {len(dataset):,} rows, {len(locations)} jurisdictions, '
        f'{len(weeks)} weeks {weeks[0]}..{weeks[-1]}')
  print(f'  wrote {args.out_dir / "locations.csv"}')
  print(f'  wrote {args.out_dir / "dataset.csv"}')
  print(f'  NHSN vintage: {provenance["rows_updated_at"]}')
  if nssp is not None:
    print(f'  NSSP {NSSP_COLUMN}: values for {nssp["weeks_with_values"]} of'
          f' {len(weeks)} weeks')
    if nssp['jurisdictions_without_nssp']:
      named = ', '.join(
          sorted(locations.loc[locations.location.astype(int).isin(
              nssp['jurisdictions_without_nssp']), 'location_name']))
      print(f'  NSSP has no data for {named}; those jurisdictions are still'
            ' forecast, with the models substituting 0.0 for this feature.')


if __name__ == '__main__':
  main()
