"""The vendored model archive must match the commit it claims to be.

models/UPSTREAM_SHA is a claim about provenance. These tests make it checkable,
so a stray edit to models/upstream/ fails here instead of silently changing what
the models compute.
"""

import hashlib
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'pipeline'))

import patch_models  # noqa: E402


def _manifest():
  rows = (REPO / 'models' / patch_models.MANIFEST).read_text().splitlines()
  return dict(
      reversed(line.split('  ', 1))
      for line in rows if line.strip() and not line.startswith('#'))


def test_vendored_files_match_manifest():
  """Every vendored file hashes to what the manifest records."""
  count = patch_models.verify_vendoring(REPO / 'models')
  assert count == len(_manifest())


def test_manifest_covers_every_vendored_file():
  """No vendored file escapes the check, and none is recorded but absent."""
  upstream = REPO / 'models' / 'upstream'
  on_disk = {p.relative_to(upstream).as_posix() for p in upstream.rglob('*.py')}
  assert on_disk == set(_manifest())


def test_manifest_names_the_pinned_commit():
  """The manifest header and UPSTREAM_SHA cannot drift apart."""
  sha = (REPO / 'models' / 'UPSTREAM_SHA').read_text().strip()
  header = (REPO / 'models' / patch_models.MANIFEST).read_text()
  assert sha in header


def test_tampering_is_detected(tmp_path):
  """A single altered byte fails verification."""
  staged = tmp_path / 'models'
  staged.mkdir()
  for name in ('UPSTREAM_SHA', patch_models.MANIFEST):
    (staged / name).write_text((REPO / 'models' / name).read_text())
  target = None
  for src in (REPO / 'models' / 'upstream').rglob('*.py'):
    rel = src.relative_to(REPO / 'models' / 'upstream')
    dst = staged / 'upstream' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    if target is None:
      target = dst

  assert patch_models.verify_vendoring(staged)  # clean copy passes
  target.write_bytes(target.read_bytes() + b'\n# tampered\n')
  with pytest.raises(ValueError, match='differs from the pinned commit'):
    patch_models.verify_vendoring(staged)
