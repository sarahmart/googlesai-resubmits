"""The per-model fix table is hand-written, so its shape is checked."""

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'pipeline'))

import patch_models  # noqa: E402


def test_fix_table_is_well_formed():
  patch_models._validate_fix_table()


def test_every_fix_targets_a_vendored_file():
  """A fix keyed to a file that does not exist would apply nothing."""
  upstream = REPO / 'models' / 'upstream'
  for context in patch_models._PER_MODEL_FIXES:
    hub, name = context.split('/')
    assert (upstream / hub / 'model_py' / name).is_file(), context


def test_every_anchor_appears_in_its_file_exactly_as_expected():
  """The anchor must match the vendored text, at the declared count."""
  upstream = REPO / 'models' / 'upstream'
  for context, fixes in patch_models._PER_MODEL_FIXES.items():
    hub, name = context.split('/')
    text = (upstream / hub / 'model_py' / name).read_text()
    for anchor, _, expected, why in fixes:
      assert text.count(anchor) == expected, f'{context}: {why}'


@pytest.mark.parametrize('bad', [
    ('anchor', 1, 'why'),                      # replacement missing
    ('anchor', 'replacement', 0, 'why'),       # non-positive count
    ('anchor', 'replacement', 1, ''),          # no reason given
])
def test_malformed_entries_are_rejected(bad, monkeypatch):
  monkeypatch.setitem(patch_models._PER_MODEL_FIXES, 'flu_hub/x.py', (bad,))
  with pytest.raises(ValueError, match='_PER_MODEL_FIXES'):
    patch_models._validate_fix_table()
