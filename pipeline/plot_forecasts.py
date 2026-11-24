"""Draw the weekly check plot: every jurisdiction, observed series and plume.

  python pipeline/plot_forecasts.py --pathogen flu --reference_date 2026-11-21 \
      --input_dir data/inputs/flu \
      --submission data/ensemble/flu/2026-11-21-Google_SAI-FluEns.csv \
      --out data/figures/flu_2026-11-21.png

Look for: 
- fan that starts where the observed series ends
- medians continuing the recent trend
- and intervals that widen with horizon
"""

import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402
from matplotlib.patches import Patch       # noqa: E402

TARGET_COLUMN = {'flu': 'Total Influenza Admissions',
                 'rsv': 'Total RSV Admissions'}
PATHOGEN_LABEL = {'flu': 'influenza', 'rsv': 'RSV'}
MODEL_ID = {'flu': 'Google_SAI-FluEns', 'rsv': 'Google_SAI-RSVEns'}
FOCAL = '#1b6ca8'
GREY = '#7a7a7a'
# The quantiles drawn: median, 50% interval, 95% interval.
Q = (0.025, 0.25, 0.5, 0.75, 0.975)


def load(pathogen, input_dir, submission, season_start):
  """Observed series since `season_start`, the fan, and last season's peaks."""
  target = TARGET_COLUMN[pathogen]
  obs = pd.read_csv(pathlib.Path(input_dir) / 'dataset.csv')
  obs['date'] = pd.to_datetime(obs.target_end_date)
  obs = obs.rename(columns={target: 'value'})

  fc = pd.read_csv(submission, dtype={'location': str})
  fc['date'] = pd.to_datetime(fc.target_end_date)
  # Submissions carry zero-padded FIPS; dataset.csv carries integer codes.
  fc['loc'] = fc.location.astype(int)
  missing = sorted(set(fc.loc[:, 'loc']) - set(obs.location))
  if missing:
    raise ValueError(f'submission has locations absent from dataset.csv: {missing}')

  start = pd.Timestamp(season_start)
  prior = obs[(obs.date >= start - pd.DateOffset(years=1)) & (obs.date < start)]
  peaks = prior.groupby('location')['value'].max().to_dict()
  names = obs.drop_duplicates('location').set_index('location')['location_name'].to_dict()

  fan = (fc[fc.output_type_id.isin(Q)]
         .pivot_table(index=['loc', 'date'], columns='output_type_id',
                      values='value')
         .reset_index())
  return obs[obs.date >= start], fan, peaks, names


def draw(pathogen, reference_date, obs, fan, peaks, names, out):
  order = sorted(fan['loc'].unique(), key=lambda l: names[l])
  ncol = 8
  nrow = int(np.ceil(len(order) / ncol))
  fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.78 * nrow))
  axes = np.atleast_1d(axes).ravel()
  for ax in axes[len(order):]:
    ax.axis('off')

  off_scale = 0
  for ax, loc in zip(axes, order):
    o = obs[obs.location == loc].sort_values('date')
    f = fan[fan['loc'] == loc].sort_values('date')
    if o.empty or f.empty:
      raise ValueError(f'no data to plot for location {loc}')
    last = o.iloc[-1]
    # Join the fan to the final observation so it starts where the data ends.
    dates = np.concatenate([[last.date.to_datetime64()], f.date.values])
    def series(q):
      return np.concatenate([[last.value], f[q].values])

    ax.fill_between(dates, series(0.025), series(0.975), color=FOCAL,
                    alpha=0.18, lw=0)
    ax.fill_between(dates, series(0.25), series(0.75), color=FOCAL,
                    alpha=0.38, lw=0)
    ax.plot(dates, series(0.5), color=FOCAL, lw=1.4, zorder=4)
    ax.plot(o.date, o.value, color='0.15', lw=1.2, marker='o', ms=2.2, zorder=5)
    # The latest week is provisional and routinely revised upward.
    ax.plot([last.date], [last.value], marker='o', ms=3.6, mfc='white',
            mec='0.15', mew=1.0, ls='none', zorder=6)

    top = max(o.value.max(), f[0.975].max()) * 1.10
    ax.set_ylim(0, top if top > 0 else 1)
    peak = peaks.get(loc)
    if peak and peak > 0:
      if peak <= top:
        ax.axhline(peak, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
      else:
        off_scale += 1
        label = f'{peak / 1000:.1f}k' if peak >= 1000 else f'{peak:.0f}'
        ax.annotate(f'\u2191 {label}', xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=5.5, color=GREY)

    ax.set_title(names[loc], loc='left', pad=2.5, fontsize=8)
    ax.margins(x=0.04)
    ax.tick_params(labelsize=6, length=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
  for i, ax in enumerate(axes[:len(order)]):
    if i + ncol < len(order):
      ax.set_xticklabels([])

  handles = [
      Line2D([], [], color='0.15', lw=1.2, marker='o', ms=2.2,
             label='Reported admissions'),
      Line2D([], [], color='0.15', marker='o', ms=3.6, mfc='white', mew=1.0,
             ls='none', label='Latest week (provisional)'),
      Line2D([], [], color=FOCAL, lw=1.4, label='Ensemble median'),
      Patch(facecolor=FOCAL, alpha=0.38, label='50% interval'),
      Patch(facecolor=FOCAL, alpha=0.18, label='95% interval'),
      Line2D([], [], color=GREY, lw=0.7, ls=(0, (3, 2)),
             label='Previous season peak (\u2191 n = above panel)'),
  ]
  fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.995, 0.010),
             frameon=False, fontsize=8)
  ends = sorted(fan.date.unique())
  fig.suptitle(
      f'{MODEL_ID[pathogen]}, reference date {reference_date}: '
      f'weekly {PATHOGEN_LABEL[pathogen]} hospital admissions, '
      f'{len(order)} jurisdictions\n'
      f'Horizons 0-{len(ends) - 1} = weeks ending '
      + ', '.join(pd.Timestamp(d).strftime('%d %b') for d in ends)
      + f'. Last reported week ending {obs.date.max():%d %b %Y}.',
      x=0.006, ha='left', y=0.997, va='top', fontsize=10)
  fig.supylabel('Weekly admissions', fontsize=9)
  fig.tight_layout(rect=(0.012, 0.0, 1, 0.955))
  out = pathlib.Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=200)
  plt.close(fig)
  return len(order), off_scale


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pathogen', choices=sorted(TARGET_COLUMN), required=True)
  parser.add_argument('--reference_date', required=True)
  parser.add_argument('--input_dir', required=True)
  parser.add_argument('--submission', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--season_start', default=None,
                      help='first week shown; defaults to 1 August of the '
                           'reference season')
  args = parser.parse_args()

  ref = pd.Timestamp(args.reference_date)
  season_start = args.season_start or f'{ref.year if ref.month >= 8 else ref.year - 1}-08-01'
  obs, fan, peaks, names = load(args.pathogen, args.input_dir, args.submission,
                                season_start)
  n, off = draw(args.pathogen, args.reference_date, obs, fan, peaks, names,
                args.out)
  print(f'  wrote {args.out} ({n} panels, {off} with the previous peak off-scale)')


if __name__ == '__main__':
  main()
