"""Reproduce the ERA model archive from a pinned upstream commit on GoogleHub:
`https://github.com/google-research/google-research/tree/master/epi_forecasts`.

This script creates / refreshes `models/upstream/`: it fetches every file from one
immutable commit and writes `UPSTREAM_SHA` and `UPSTREAM_MANIFEST.txt` from the fetch.

  # verify the current vendoring against its pinned commit, rewrite the manifest
  python pipeline/vendor_models.py

  # move to a newer upstream commit (full 40-character sha, never a branch)
  python pipeline/vendor_models.py --sha <commit>

"""

import argparse
import hashlib
import json
import pathlib
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
OWNER = 'google-research/google-research'
SUBDIRS = ('flu_hub/model_py', 'rsv_hub/model_py')
MANIFEST = 'UPSTREAM_SHA_LST.txt'


def _get(url):
  with urllib.request.urlopen(url, timeout=120) as fh:
    return fh.read()


def list_files(sha):
  """Every .py under the vendored subdirectories at this commit."""
  tree = json.loads(
      _get(f'https://api.github.com/repos/{OWNER}/git/trees/{sha}'
           '?recursive=1'))
  if tree.get('truncated'):
    raise RuntimeError('GitHub truncated the tree listing; cannot enumerate')
  prefixes = tuple(f'epi_forecasts/{d}/' for d in SUBDIRS)
  paths = [
      e['path'] for e in tree['tree']
      if e['type'] == 'blob'
      and e['path'].startswith(prefixes)
      and e['path'].endswith('.py')
  ]
  if not paths:
    raise RuntimeError(f'no model files found at {sha}; is the commit correct?')
  return sorted(paths)


def vendor(sha, write_files):
  models = REPO / 'models'
  upstream = models / 'upstream'
  rows, changed = [], []

  for path in list_files(sha):
    relative = path[len('epi_forecasts/'):]
    blob = _get(f'https://raw.githubusercontent.com/{OWNER}/{sha}/{path}')
    digest = hashlib.sha256(blob).hexdigest()
    local = upstream / relative

    if write_files:
      local.parent.mkdir(parents=True, exist_ok=True)
      if not local.is_file() or local.read_bytes() != blob:
        changed.append(relative)
      local.write_bytes(blob)
    else:
      if not local.is_file():
        raise FileNotFoundError(
            f'{relative} is missing locally; re-vendor with --sha to fetch it')
      if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
        raise ValueError(
            f'{relative} differs from {sha}. Refusing. Re-vendor with --sha to overwrite it.')
    rows.append(f'{digest}  {relative}')

  (models / 'UPSTREAM_SHA').write_text(sha + '\n')
  (models / MANIFEST).write_text(
      f'# sha256 of each vendored file at {OWNER}@{sha}\n'
      '# Verified against the pinned commit when generated.\n'
      + '\n'.join(rows) + '\n')
  return rows, changed


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--sha',
      help='upstream commit to vendor (full 40-char sha). Omit to verify the '
           'existing vendoring against its current pin and rewrite the manifest.')
  args = parser.parse_args()

  write_files = args.sha is not None
  sha = args.sha or (REPO / 'models' / 'UPSTREAM_SHA').read_text().strip()
  if len(sha) != 40 or not all(c in '0123456789abcdef' for c in sha):
    raise SystemExit(
        f'{sha!r} is not a full commit sha. Branch names and short shas are '
        'rejected: the archive must be pinned to something immutable.')

  rows, changed = vendor(sha, write_files)
  print(f'{len(rows)} files vendored at {sha[:12]}')
  if write_files:
    print(f'{len(changed)} changed'
          + (': ' + ', '.join(changed[:5]) + ('...' if len(changed) > 5 else '')
             if changed else ''))
  else:
    print('local copy matches the pinned commit; manifest rewritten')
  print('Now re-run pipeline/patch_models.py.')


if __name__ == '__main__':
  main()
