"""Build the three ILINet extracts the patched ERA models read at import time."""

import argparse
import datetime
import pathlib

import pandas as pd
import requests

EPIDATA = 'https://api.delphi.cmu.edu/epidata/fluview/'
CUTOFF = datetime.date(2022, 10, 15)
REQUEST_TIMEOUT = 120

# Delphi column -> cdcfluview column, as consumed by the models.
COLUMN_MAP = {
    'ili': 'unweighted_ili',
    'wili': 'weighted_ili',
    'num_ili': 'ilitotal',
    'num_patients': 'total_patients',
    'num_providers': 'num_of_providers',
}
HHS_REGIONS = [f'hhs{i}' for i in range(1, 11)]


def mmwr_week_start(year, week):
  """Sunday beginning MMWR week `week` of `year`.

  MMWR week 1 is the first week with at least four days in the calendar year,
  i.e. the Sun-Sat week containing 4 January.
  """
  jan4 = datetime.date(year, 1, 4)
  week1_start = jan4 - datetime.timedelta(days=(jan4.weekday() + 1) % 7)
  return week1_start + datetime.timedelta(weeks=week - 1)


SEASON_START_WEEK = 40


def season_label(year, week):
  """ILINet season label in the 'YYYY/YY' form the models expect.

  Format taken from `convert_ilinet_season_to_id` in flu Adapted_4, which
  documents "Converts 'YYYY/YY' ILINet season format" and reads the start year
  from `season_str.split('/')[0]`. Adapted_7 carries the column through and
  drops it later, so only the start year is load-bearing.

  The boundary is MMWR week 40, the CDC convention for influenza surveillance
  seasons.
  """
  start = year if week >= SEASON_START_WEEK else year - 1
  return f'{start}/{(start + 1) % 100:02d}'


def _check_mmwr():
  """Anchors from the CDC MMWR calendar, including both year-boundary cases.

  2014w1 begins in the previous calendar year because 1-4 January 2014 is four
  days, meeting the four-day rule. 2015w1 does not, because the week beginning
  28 December 2014 has only three days in 2015 and is 2014w53 instead.
  """
  for (year, week), expected in {
      (2022, 40): datetime.date(2022, 10, 2),
      (2010, 40): datetime.date(2010, 10, 3),
      (2014, 1): datetime.date(2013, 12, 29),
      (2015, 1): datetime.date(2015, 1, 4),
  }.items():
    got = mmwr_week_start(year, week)
    if got != expected:
      raise AssertionError(
          f'MMWR {year}w{week}: expected {expected}, got {got}')


def fetch(regions):
  """All available ILINet weeks for `regions`, as a frame."""
  response = requests.get(EPIDATA, params={
      'regions': ','.join(regions), 'epiweeks': '190001-205201',
  }, timeout=REQUEST_TIMEOUT)
  response.raise_for_status()
  payload = response.json()
  if payload.get('result') != 1:
    raise RuntimeError(
        f'Delphi returned result={payload.get("result")}:'
        f' {payload.get("message")} (regions={regions})')
  frame = pd.DataFrame(payload['epidata'])
  missing = set(regions) - set(frame.region)
  if missing:
    raise ValueError(f'Delphi returned no rows for regions {sorted(missing)}')
  return frame


def to_cdcfluview(frame, region_type, region_names):
  """Translate Delphi output into the layout the models expect."""
  frame = frame.rename(columns=COLUMN_MAP).copy()
  frame['region'] = frame.region.map(region_names)
  if frame.region.isna().any():
    raise ValueError('a Delphi region code had no display name mapping')
  frame['region_type'] = region_type
  frame['year'] = frame.epiweek // 100
  frame['week'] = frame.epiweek % 100
  frame['week_start'] = [
      mmwr_week_start(y, w) for y, w in zip(frame.year, frame.week)]
  frame['season'] = [season_label(y, w) for y, w in zip(frame.year, frame.week)]
  frame = frame[frame.week_start < CUTOFF]
  if frame.empty:
    raise ValueError(f'no {region_type} rows remain before {CUTOFF}')
  columns = ['region_type', 'region', 'year', 'week', 'week_start', 'season',
             'weighted_ili', 'unweighted_ili', 'ilitotal', 'num_of_providers',
             'total_patients']
  return frame[columns].sort_values(['region', 'week_start'])


def build(out_dir, state_names):
  _check_mmwr()
  national = to_cdcfluview(fetch(['nat']), 'National', {'nat': 'National'})
  hhs = to_cdcfluview(
      fetch(HHS_REGIONS), 'HHS Regions',
      {f'hhs{i}': f'Region {i}' for i in range(1, 11)})
  states = to_cdcfluview(
      fetch(sorted(state_names)), 'States', state_names)

  out_dir.mkdir(parents=True, exist_ok=True)
  written = {}
  for name, frame in (('ilinet_before_20221015.csv', national),
                      ('ilinet_hhs_before_20221015.csv', hhs),
                      ('ilinet_state_before_20221015.csv', states)):
    frame.to_csv(out_dir / name, index=False)
    written[name] = frame
  return written


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--out_dir', type=pathlib.Path, required=True)
  parser.add_argument('--locations', type=pathlib.Path, required=True,
                      help='locations.csv written by build_inputs.py')
  args = parser.parse_args()

  locations = pd.read_csv(args.locations, dtype={'location': str})
  # Delphi keys states by lowercase postal code; the models key ILINet rows by
  # the full state name, which is how they join to the target series.
  # ILINet has no national-aggregate state code; the hub file's `US` row and
  # any territory Delphi does not cover are excluded here.
  state_names = {row.abbreviation.lower(): row.location_name
                 for row in locations.itertuples()
                 if row.location != 'US'}

  written = build(args.out_dir, state_names)
  for name, frame in written.items():
    span = f'{frame.week_start.min()}..{frame.week_start.max()}'
    print(f'  {name}: {len(frame):,} rows, {frame.region.nunique()} regions, {span}')


if __name__ == '__main__':
  main()
