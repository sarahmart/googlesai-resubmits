"""Generate runnable models from the vendored ERA archive.

The archived code at `models/upstream/` is published on GoogleHub as produced
by ERA and does not run as is.

This script makes cosmetic changes to the models so they are runnable, keeping
the diff between `models/upstream/` and `models/patched/` small and auditable. 
Run it whenever the pinned upstream commit changes.
"""

import argparse
import hashlib
import pathlib
import shutil

HUBS = ('flu_hub', 'rsv_hub')
MANIFEST = 'UPSTREAM_SHA_LST.txt'


def verify_vendoring(models_dir):
  """Check the vendored archive still matches the commit it claims to be
  in `UPSTREAM_SHA` (offline).
  """
  manifest_path = models_dir / MANIFEST
  if not manifest_path.is_file():
    raise FileNotFoundError(
        f'{manifest_path} is missing; the vendored archive cannot be verified')

  expected = {}
  for line in manifest_path.read_text().splitlines():
    if line.startswith('#') or not line.strip():
      continue
    digest, relative = line.split('  ', 1)
    expected[relative] = digest

  upstream = models_dir / 'upstream'
  found = {p.relative_to(upstream).as_posix() for p in upstream.rglob('*.py')}
  missing = sorted(set(expected) - found)
  extra = sorted(found - set(expected))
  altered = sorted(
      relative for relative, digest in expected.items()
      if relative in found
      and hashlib.sha256((upstream / relative).read_bytes()).hexdigest() != digest)

  problems = []
  if missing:
    problems.append(f'missing from models/upstream/: {missing}')
  if extra:
    problems.append(f'present but not in the manifest: {extra}')
  if altered:
    problems.append(f'content differs from the pinned commit: {altered}')
  if problems:
    sha = (models_dir / 'UPSTREAM_SHA').read_text().strip()
    raise ValueError(
        f'vendored archive does not match {sha}:\n  '
        + '\n  '.join(problems)
        + '\nRe-vendor from the pinned commit, or update UPSTREAM_SHA and the'
          ' manifest together if the pin is being moved deliberately.')
  return len(expected)


SHARED = ('constant_defs.py', 'epi_utils.py', 'plotting_utils.py')

_RUNTIME_IMPORT = 'import epi_runtime\n'

_MODEL_REWRITES = (
    # Read inputs from a configured directory instead of the repo root.
    (
        "INPUT_DIR = ''",
        _RUNTIME_IMPORT + 'INPUT_DIR = epi_runtime.input_dir()',
        1,
    ),
    # `main()` assigns to `locations`, making the module-level global a local.
    (
        'def main(argv):\n  del argv  # Unused.\n',
        'def main(argv):\n  del argv  # Unused.\n  global locations\n',
        1,
    ),
    # Write forecasts where the pipeline expects them.
    (
        "f'/tmp/{submission_date_str}_{MODEL_NAME}.csv'",
        "f'{epi_runtime.output_dir()}/{submission_date_str}_{MODEL_NAME}.csv'",
        1,
    ),
    # Appears twice: the to_csv call and the log line that echoes it.
    (
        "'/tmp/validation_forecasts.csv'",
        "f'{epi_runtime.output_dir()}/validation_forecasts.csv'",
        2,
    ),
)

# Fixes for bugs in individual archived files, keyed by '<hub>/<filename>'.
# Each entry is (anchor, replacement, expected_occurrences, why).
#
# These are repairs, not adaptations: each file raises before producing a
# forecast. They live here rather than as edits to models/upstream/ so that the
# vendored archive stays byte-identical to the published commit, the reason for
# each change is recorded next to it, and the anchor assertion fires if upstream
# ever repairs the file itself.
_PER_MODEL_FIXES = {
    'flu_hub/Google_SAI-Adapted_7.py': (
        (
            'HORIZONS = [0, 1, 2, 3]',
            'HORIZONS = [0, 1, 2, 3]\n'
            '# PATCH: upstream references a lower-case `horizons` at module\n'
            '# scope but never binds it, so this model raises NameError before\n'
            '# forecasting. Sibling Google_SAI-Adapted_24 carries exactly this\n'
            '# alias; Adapted_7 is missing it.\n'
            'horizons = HORIZONS',
            1,
            'NameError: name `horizons` is not defined',
        ),
    ),
    'rsv_hub/constant_defs.py': (
        (
            'HORIZONS = [-1, 0, 1, 2, 3]',
            '# PATCH: the models were run with horizons 0-3, confirmed by the\n'
            '# Google_SAI authors, and every submitted Google_SAI-RSVEns file\n'
            '# carries 0-3 only. Keeping -1 also breaks both RSV models: its\n'
            '# target week is the last observed week, which duplicates that row\n'
            '# in the forecast scaffold.\n'
            'HORIZONS = [0, 1, 2, 3]',
            1,
            'declared -1 horizon was never submitted and duplicates the scaffold',
        ),
    ),
    'rsv_hub/Google_SAI-Novel_1.py': (
        (
            "format_for_cdc(test_forecasts, 'wk inc flu hosp')",
            "format_for_cdc(test_forecasts, 'wk inc rsv hosp')",
            1,
            'RSV model labelled its output with the influenza target',
        ),
        (
            "  current_fold_train_end_date = test_x_copy[\n"
            "      'reference_date'\n"
            "  ].min() - pd.Timedelta(weeks=1)",
            '  # PATCH: reference_date arrives as a datetime.date while this\n'
            "  # model keeps target_end_date as datetime64 (it never calls\n"
            '  # .dt.date, unlike the flu models), so the comparison against\n'
            '  # this value raises TypeError. Novel_2 coerces the same value.\n'
            '  current_fold_train_end_date = pd.to_datetime(\n'
            "      test_x_copy['reference_date'].min()\n"
            '  ) - pd.Timedelta(weeks=1)',
            1,
            'TypeError: Invalid comparison between dtype=datetime64[ns] and date',
        ),
    ),
    'flu_hub/Google_SAI-Hybrid_1.py': (
        (
            '        prev_week_epiweek = (\n'
            '            pd.to_datetime(prev_week_date).isocalendar().week.astype(int)\n'
            '        )',
            '        # PATCH: `prev_week_date` is a scalar here, so\n'
            '        # Timestamp.isocalendar().week is a plain int and has no\n'
            '        # .astype. The call only works on a Series, where\n'
            '        # .isocalendar() returns a DataFrame.\n'
            '        prev_week_epiweek = int(\n'
            '            pd.to_datetime(prev_week_date).isocalendar().week\n'
            '        )',
            1,
            "AttributeError: 'int' object has no attribute 'astype'",
        ),
    ),
}


_VALIDATION_START = '  # --- Execute Validation Run ---'
_VALIDATION_END = '  submission_date_str = get_next_saturday_date_str()'


def _replace_exact(text, anchor, replacement, context, expected=1):
  count = text.count(anchor)
  if count != expected:
    raise ValueError(
        f'{context}: expected {expected} occurrence(s) of {anchor!r}, found'
        f' {count}. Upstream structure has changed; review before patching.'
    )
  return text.replace(anchor, replacement)


def _gate_validation_block(text, context):
  """Indent the in-built validation block under a runtime flag."""
  lines = text.split('\n')
  starts = [i for i, line in enumerate(lines) if line == _VALIDATION_START]
  ends = [i for i, line in enumerate(lines) if line == _VALIDATION_END]
  if len(starts) != 1 or len(ends) != 1:
    raise ValueError(
        f'{context}: expected one validation start and one end marker, found'
        f' {len(starts)} and {len(ends)}.'
    )
  start, end = starts[0], ends[0]
  if end <= start:
    raise ValueError(f'{context}: validation end marker precedes its start.')
  block = ['  ' + line if line.strip() else line for line in lines[start:end]]
  gated = ['  if epi_runtime.run_validation():'] + block
  return '\n'.join(lines[:start] + gated + lines[end:])


def _validate_fix_table():
  """Check the shape of `_PER_MODEL_FIXES` before anything relies on it."""
  for context, fixes in _PER_MODEL_FIXES.items():
    for i, fix in enumerate(fixes):
      if not isinstance(fix, tuple) or len(fix) != 4:
        raise ValueError(
            f'_PER_MODEL_FIXES[{context!r}][{i}] has {len(fix)} field(s),'
            ' expected 4: (anchor, replacement, expected_count, why).')
      anchor, replacement, expected, why = fix
      if not isinstance(anchor, str) or not isinstance(replacement, str):
        raise ValueError(
            f'_PER_MODEL_FIXES[{context!r}][{i}]: anchor and replacement must'
            ' both be strings.')
      if not isinstance(expected, int) or expected < 1:
        raise ValueError(
            f'_PER_MODEL_FIXES[{context!r}][{i}]: expected_count must be a'
            f' positive integer, got {expected!r}.')
      if not isinstance(why, str) or not why:
        raise ValueError(
            f'_PER_MODEL_FIXES[{context!r}][{i}]: `why` must be a non-empty'
            ' string naming the error the fix repairs.')


def _apply_per_file_fixes(text, context):
  for anchor, replacement, expected, why in _PER_MODEL_FIXES.get(context, ()):
    text = _replace_exact(
        text, anchor, replacement, f'{context} [{why}]', expected)
  return text


def _patch_model(text, context):
  for anchor, replacement, expected in _MODEL_REWRITES:
    text = _replace_exact(text, anchor, replacement, context, expected)
  text = _apply_per_file_fixes(text, context)
  return _gate_validation_block(text, context)


def _patch_epi_utils(text, context):
  """Make the reference date explicit instead of wall-clock derived."""
  anchor = (
      '  today = Date.today()\n'
      '  days_to_add = (5 - today.weekday()) % 7\n'
      '  next_saturday = today + Timedelta(days=days_to_add)\n'
      "  return next_saturday.strftime('%Y-%m-%d')\n"
  )
  text = _replace_exact(text, anchor, '  return epi_runtime.reference_date()\n',
                        context)
  return _replace_exact(text, 'import constant_defs\n',
                        _RUNTIME_IMPORT + 'import constant_defs\n', context)


def _patch_plotting_utils(text, context):
  """Force a headless backend and make plotting opt-in."""
  text = _replace_exact(
      text,
      'from matplotlib import dates as mdates\n',
      "import matplotlib\nmatplotlib.use('Agg')\n"
      + _RUNTIME_IMPORT
      + 'from matplotlib import dates as mdates\n',
      context,
  )
  signature = 'def plot_season_forecasts('
  _replace_exact(text, signature, signature, context)  # assert uniqueness
  body = text.index('\n', text.index('):', text.index(signature))) + 1
  guard = '  if not epi_runtime.plotting_enabled():\n    return\n'
  return text[:body] + guard + text[body:]


_SHARED_PATCHERS = {
    'epi_utils.py': _patch_epi_utils,
    'plotting_utils.py': _patch_plotting_utils,
    'constant_defs.py': None,
}


def patch(repo_root):
  upstream = repo_root / 'models' / 'upstream'
  patched = repo_root / 'models' / 'patched'
  runtime = repo_root / 'pipeline' / 'epi_runtime.py'
  if not runtime.is_file():
    raise FileNotFoundError(f'missing {runtime}')

  _validate_fix_table()
  verified = verify_vendoring(repo_root / 'models')
  print(f'verified {verified} vendored files against '
        f'{(repo_root / "models" / "UPSTREAM_SHA").read_text().strip()[:12]}')

  written = 0
  seen = set()
  for hub in HUBS:
    src, dst = upstream / hub / 'model_py', patched / hub
    if not src.is_dir():
      raise FileNotFoundError(f'missing vendored sources: {src}')
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime, dst / 'epi_runtime.py')

    for path in sorted(src.glob('*.py')):
      text = path.read_text()
      context = f'{hub}/{path.name}'
      if path.name in _SHARED_PATCHERS:
        patcher = _SHARED_PATCHERS[path.name]
        text = text if patcher is None else patcher(text, context)
        text = _apply_per_file_fixes(text, context)
      else:
        text = _patch_model(text, context)
      (dst / path.name).write_text(text)
      seen.add(context)
      written += 1

  # A per-model fix keyed to a file that is never patched would apply nothing
  # and report success, so treat an unused key as an error rather than let a
  # renamed or removed model silently lose its repair.
  unused = sorted(set(_PER_MODEL_FIXES) - seen)
  if unused:
    raise ValueError(
        f'per-model fixes target files that were not patched: {unused}. '
        'Check the key spelling against "<hub>/<filename>", or drop the entry '
        'if upstream removed the model.')
  return written


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--repo_root', type=pathlib.Path,
      default=pathlib.Path(__file__).resolve().parent.parent)
  args = parser.parse_args()
  count = patch(args.repo_root)
  print(f'patched {count} files into {args.repo_root / "models" / "patched"}')


if __name__ == '__main__':
  main()
