"""Choose ensemble components by greedy forward selection on log WIS.

Scores candidate ensembles directly against the archived 2025-26 forecasts and
truth from `epi-hub-analysis`, so membership is measured rather than inferred
from individual model scores. Individual rankings cannot see redundancy: two
near-identical models both score well yet add nothing together.

Three temporally ordered blocks:
  A  order   greedy adds components by largest reduction in mean log WIS
  B  size    how many to keep is read off this block
  C  report  never consulted until the set is fixed

Choosing both the order and the size on the same block that reports the score
leaks: measured on 2025-26, picking k on the reported block flatters it by
about 0.003 log WIS. Block C exists so the headline number is honest. A cluster
bootstrap over reference dates shows which differences survive sampling noise.

Comparison window: every candidate must be present, which for 2025-26 means
reference dates from 2026-01-24 onwards. That is a post-peak window, so these
results say little about growth-phase skill.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scoring  # noqa: E402  (path set above so the module resolves standalone)

TARGET = 'wk inc flu hosp'
HORIZONS = (0, 1, 2, 3)


def load_tasks(data_dir, quantiles):
  """Join archived component forecasts to the values they were graded against."""
  forecasts = pd.read_parquet(data_dir / 'google_flu' / 'forecasts.parquet')
  forecasts = forecasts[
      (forecasts.target == TARGET)
      & (forecasts.output_type == 'quantile')
      & (forecasts.horizon.astype(int).isin(HORIZONS))
  ].copy()
  forecasts['output_type_id'] = forecasts.output_type_id.astype(float).round(3)

  truth = pd.read_parquet(data_dir / 'flu' / 'truth.parquet')
  # The truth file mixes final oracle values with as-of revision snapshots;
  # only the quantile-scored rows carry the value forecasts are graded against.
  truth = (
      truth[(truth.target == TARGET)
            & (truth.output_type == 'quantile')
            & truth.oracle_value.notna()]
      [['target_end_date', 'location', 'oracle_value']]
      .drop_duplicates()
      .rename(columns={'oracle_value': 'observed'})
  )
  if truth.duplicated(['target_end_date', 'location']).any():
    raise ValueError('truth has multiple observations for one date/location')

  wide = forecasts.pivot_table(
      index=['model_id', 'reference_date', 'horizon', 'location',
             'target_end_date'],
      columns='output_type_id', values='value').reset_index()
  wide = wide.merge(truth, on=['target_end_date', 'location'], how='inner')
  qcols = sorted(c for c in wide.columns if isinstance(c, float))
  if qcols != sorted(quantiles):
    raise ValueError(f'quantile columns {qcols} do not match {sorted(quantiles)}')
  return wide, qcols


def build_arrays(wide, qcols, window_start):
  """Dense (model, task, quantile) array over the common comparison window."""
  wide = wide[wide.reference_date >= pd.Timestamp(window_start)].copy()
  wide['task'] = (wide.reference_date.astype(str) + '|'
                  + wide.horizon.astype(int).astype(str) + '|'
                  + wide.location.astype(str))
  models = sorted(wide.model_id.unique())
  tasks = sorted(wide.task.unique())
  m_idx = {m: i for i, m in enumerate(models)}
  t_idx = {t: i for i, t in enumerate(tasks)}

  grid = np.full((len(models), len(tasks), len(qcols)), np.nan)
  grid[wide.model_id.map(m_idx).to_numpy(),
       wide.task.map(t_idx).to_numpy(), :] = wide[qcols].to_numpy()
  observed = np.full(len(tasks), np.nan)
  observed[wide.task.map(t_idx).to_numpy()] = wide.observed.to_numpy()
  task_ref = np.array([t.split('|')[0] for t in tasks])
  return models, grid, observed, task_ref


def drop_partial_coverage(models, grid):
  """Keep only candidates present on every task in the window.

  A candidate absent from some tasks cannot be compared like-for-like: its
  score would be computed on a different, possibly easier, subset. Dropping is
  reported rather than silent.
  """
  covered = (~np.isnan(grid[:, :, 0])).all(axis=1)
  dropped = [(models[i], float((~np.isnan(grid[i, :, 0])).mean()))
             for i in np.flatnonzero(~covered)]
  for name, fraction in dropped:
    print(f'dropping {name}: covers {fraction:.1%} of window tasks')
  if not covered.any():
    raise ValueError('no candidate covers the whole window')
  return [models[i] for i in np.flatnonzero(covered)], grid[covered]


def ensemble_scores(grid, observed, quantiles, members, mask):
  """Per-task log WIS of the equal-weight quantile mean over `members`."""
  if not list(members):
    raise ValueError('ensemble requires at least one member')
  members = list(members)
  block = grid[members][:, mask, :]
  present = ~np.isnan(block[:, :, 0])
  if not present.any(axis=0).all():
    raise ValueError(
        'at least one task has no contributing member; widen the window or'
        ' drop the offending model.')
  mean_quantiles = np.nanmean(block, axis=0)
  return scoring.log_weighted_interval_score(
      mean_quantiles, observed[mask], quantiles)


def greedy_select(grid, observed, quantiles, select_mask, eval_mask, max_k):
  chosen, remaining, curve = [], list(range(grid.shape[0])), []
  for _ in range(max_k):
    best = min(remaining, key=lambda m: ensemble_scores(
        grid, observed, quantiles, chosen + [m], select_mask).mean())
    chosen.append(best)
    remaining.remove(best)
    curve.append({
        'k': len(chosen),
        'added': best,
        'select_log_wis': float(ensemble_scores(
            grid, observed, quantiles, chosen, select_mask).mean()),
        'eval_log_wis': float(ensemble_scores(
            grid, observed, quantiles, chosen, eval_mask).mean()),
    })
  return chosen, pd.DataFrame(curve)


def bootstrap_difference(grid, observed, quantiles, members_a, members_b,
                         eval_mask, task_ref, n_draws, seed):
  """Cluster bootstrap over reference dates of mean(a) - mean(b) log WIS."""
  rng = np.random.default_rng(seed)
  a = ensemble_scores(grid, observed, quantiles, members_a, eval_mask)
  b = ensemble_scores(grid, observed, quantiles, members_b, eval_mask)
  refs = task_ref[eval_mask]
  groups = {r: np.where(refs == r)[0] for r in np.unique(refs)}
  keys = list(groups)
  draws = np.empty(n_draws)
  for i in range(n_draws):
    pick = np.concatenate(
        [groups[r] for r in rng.choice(keys, len(keys), replace=True)])
    draws[i] = a[pick].mean() - b[pick].mean()
  return float(a.mean() - b.mean()), np.percentile(draws, [2.5, 97.5])


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  root = pathlib.Path(__file__).resolve().parent.parent
  parser.add_argument('--data_dir', type=pathlib.Path,
                      default=root / 'data' / 'epi-hub-analysis')
  parser.add_argument('--out', type=pathlib.Path,
                      default=root / 'data' / 'flu_selection_curve.csv')
  parser.add_argument('--window_start', default='2026-01-24',
                      help='first reference date on which all candidates exist')
  parser.add_argument('--n_order_dates', type=int, default=7,
                      help='block A: reference dates used to order components')
  parser.add_argument('--n_size_dates', type=int, default=6,
                      help='block B: reference dates used to choose k')
  parser.add_argument('--max_k', type=int, default=12)
  parser.add_argument('--n_draws', type=int, default=2000)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--compare', nargs='*', default=[],
                      help='model ids forming a reference set to compare against')
  args = parser.parse_args(argv)

  import constant_defs
  quantiles = constant_defs.QUANTILES

  wide, qcols = load_tasks(args.data_dir, quantiles)
  models, grid, observed, task_ref = build_arrays(wide, qcols, args.window_start)
  models, grid = drop_partial_coverage(models, grid)

  window_refs = sorted(set(task_ref))
  n_report = len(window_refs) - args.n_order_dates - args.n_size_dates
  if n_report < 2:
    raise ValueError(
        f'{len(window_refs)} reference dates cannot supply order'
        f' ({args.n_order_dates}) + size ({args.n_size_dates}) blocks and leave'
        ' at least 2 for reporting.')
  block_a = window_refs[:args.n_order_dates]
  block_b = window_refs[args.n_order_dates:args.n_order_dates + args.n_size_dates]
  block_c = window_refs[args.n_order_dates + args.n_size_dates:]
  select_mask = np.isin(task_ref, block_a)
  eval_mask = np.isin(task_ref, block_b)
  report_mask = np.isin(task_ref, block_c)

  chosen, curve = greedy_select(
      grid, observed, quantiles, select_mask, eval_mask, args.max_k)
  curve['added'] = [models[i] for i in curve['added']]
  args.out.parent.mkdir(parents=True, exist_ok=True)
  curve.to_csv(args.out, index=False)

  print(f'window {window_refs[0]}..{window_refs[-1]}, {len(models)} candidates')
  print(f'  A order  {block_a[0]}..{block_a[-1]}  {int(select_mask.sum())} tasks')
  print(f'  B size   {block_b[0]}..{block_b[-1]}  {int(eval_mask.sum())} tasks')
  print(f'  C report {block_c[0]}..{block_c[-1]}  {int(report_mask.sum())} tasks')
  print(curve.to_string(index=False))

  best_k = int(curve.loc[curve.eval_log_wis.idxmin(), 'k'])
  n_holdout = len(block_b)

  def compare(members):
    return bootstrap_difference(
        grid, observed, quantiles, members, chosen[:best_k], eval_mask,
        task_ref, args.n_draws, args.seed)

  # How far down the curve can we go before the held-out score is distinguishable
  # from the best? Prefer the smallest such k: fewer components, same skill.
  print(f'\ncluster bootstrap vs greedy k={best_k}, '
        f'over {n_holdout} held-out reference dates:')
  indistinguishable = []
  for k in curve['k']:
    if k == best_k:
      continue
    diff, ci = compare(chosen[:k])
    same = ci[0] < 0 < ci[1]
    indistinguishable.append(k) if same else None
    print(f'  k={k:<3d}             diff {diff:+.5f}  '
          f'95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]'
          f'{"" if same else "   CI excludes 0"}')

  if args.compare:
    missing = [m for m in args.compare if m not in models]
    if missing:
      raise ValueError(f'--compare names absent from the window: {missing}')
    diff, ci = compare([models.index(m) for m in args.compare])
    print(f'  supplied set       diff {diff:+.5f}  '
          f'95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]'
          f'{"" if ci[0] < 0 < ci[1] else "   CI excludes 0"}')
  diff, ci = compare(list(range(len(models))))
  print(f'  all {len(models)} candidates  diff {diff:+.5f}  '
        f'95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]'
        f'{"" if ci[0] < 0 < ci[1] else "   CI excludes 0"}')

  parsimonious = min(indistinguishable) if indistinguishable else best_k
  print(f'\nk={best_k} minimises block B; smallest k indistinguishable from it:'
        f' k={parsimonious}')

  # Block C has not been consulted by anything above. This is the honest number.
  chosen_members = chosen[:best_k]
  report = float(ensemble_scores(
      grid, observed, quantiles, chosen_members, report_mask).mean())
  peeked_k = min(range(1, args.max_k + 1),
                 key=lambda k: ensemble_scores(
                     grid, observed, quantiles, chosen[:k], report_mask).mean())
  peeked = float(ensemble_scores(
      grid, observed, quantiles, chosen[:peeked_k], report_mask).mean())
  print(f'\nblock C (untouched), greedy k={best_k}: {report:.5f}')
  print(f'  had we chosen k on block C (k={peeked_k}): {peeked:.5f}'
        f'  -> optimism {report - peeked:+.5f}')
  if args.compare:
    print(f'  supplied set: '
          f'{ensemble_scores(grid, observed, quantiles, [models.index(m) for m in args.compare], report_mask).mean():.5f}')
  print(f'  all {len(models)} candidates: '
        f'{ensemble_scores(grid, observed, quantiles, list(range(len(models))), report_mask).mean():.5f}')
  print('  components: '
        + ', '.join(models[i].replace('Google_SAI-', '') for i in chosen_members))
  print(f'\nwrote {args.out}')


if __name__ == '__main__':
  main()
