"""Vectorised Weighted Interval Score (WIS), matching the CDC hub definition."""

import numpy as np


def weighted_interval_score(predicted, observed, quantile_levels):
  """WIS for many forecasts at once. Lower is better.

  Args:
    predicted: (n, n_quantiles) array of predictive quantiles, columns ordered
      to match `quantile_levels`.
    observed: (n,) array of observed values.
    quantile_levels: sorted sequence of quantile levels.

  Returns:
    (n,) array of WIS values.
  """
  predicted = np.asarray(predicted, dtype=float)
  observed = np.asarray(observed, dtype=float)
  levels = np.asarray(sorted(quantile_levels), dtype=float)

  if predicted.shape[1] != levels.size:
    raise ValueError(
        f'predicted has {predicted.shape[1]} quantile columns but'
        f' {levels.size} levels were supplied.'
    )
  if predicted.shape[0] != observed.shape[0]:
    raise ValueError(
        f'predicted has {predicted.shape[0]} rows but observed has'
        f' {observed.shape[0]}.'
    )

  n_lower = levels.size // 2
  lower_levels = levels[:n_lower]
  # Upper partner of each lower level, matched by rounding as upstream does.
  upper_idx = [
      int(np.argmin(np.abs(levels - round(1 - q, 3)))) for q in lower_levels
  ]

  lower = predicted[:, :n_lower]
  upper = predicted[:, upper_idx]
  alpha = 2.0 * lower_levels
  y = observed[:, None]

  width = upper - lower
  below = np.where(y < lower, (2.0 / alpha) * (lower - y), 0.0)
  above = np.where(y > upper, (2.0 / alpha) * (y - upper), 0.0)
  interval = width + below + above

  median = predicted[:, predicted.shape[1] // 2]
  # w0 = 0.5 on the median term, per Bracher et al. (2021) and Methods eq. (1).
  total = 0.5 * np.abs(observed - median) + ((alpha / 2.0) * interval).sum(axis=1)
  return total / (n_lower + 0.5)


def log_weighted_interval_score(predicted, observed, quantile_levels):
  """WIS on the log(1 + x) scale, the CDC hubs' headline ranking metric."""
  return weighted_interval_score(
      np.log1p(np.asarray(predicted, dtype=float)),
      np.log1p(np.asarray(observed, dtype=float)),
      quantile_levels,
  )
