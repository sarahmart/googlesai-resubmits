"""Runtime configuration shared by every patched ERA model.

Single source of truth for the handful of behaviours the archived model code
hard-coded. `patch_models.py` copies this module alongside the patched models,
so it must depend only on the standard library.
"""

import datetime
import os


def _require(name):
  value = os.environ.get(name)
  if not value:
    raise RuntimeError(
        f'{name} is not set. Patched ERA models require it; set it to an'
        ' absolute path or date before invoking the model.'
    )
  return value


def input_dir():
  """Directory holding dataset.csv, locations.csv and the ILINet extracts."""
  path = _require('EPI_INPUT_DIR')
  if not os.path.isdir(path):
    raise RuntimeError(f'EPI_INPUT_DIR does not exist: {path}')
  return path


def output_dir():
  """Directory for forecast CSVs. Created if absent."""
  path = _require('EPI_OUTPUT_DIR')
  os.makedirs(path, exist_ok=True)
  return path


def reference_date():
  """The forecast reference date (a Saturday), as 'YYYY-MM-DD'.

  Upstream derived this from the wall clock, which silently targets the wrong
  week when a run happens on a Friday or Saturday. Requiring it explicitly
  makes the weekly job reproducible and re-runnable.
  """
  raw = _require('EPI_REFERENCE_DATE')
  parsed = datetime.date.fromisoformat(raw)
  if parsed.weekday() != 5:
    raise RuntimeError(
        f'EPI_REFERENCE_DATE must be a Saturday; {raw} is a'
        f' {parsed.strftime("%A")}.'
    )
  return raw


def run_validation():
  """Whether to run each model's in-built rolling validation block.

  Off by default: it consumes roughly four fifths of a model's runtime and
  produces output that is never submitted.
  """
  return os.environ.get('EPI_RUN_VALIDATION', '0') == '1'


def plotting_enabled():
  return os.environ.get('EPI_PLOTTING', '0') == '1'
