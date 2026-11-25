"""Run one week end to end, from the CDC releases to a file ready to submit.

This is the only script run on a normal week, which is why it sits at the
top level rather than in `pipeline/` with the modules it drives.

  python run_week.py --pathogen flu --reference_date 2026-11-28

Six steps, in order: build the NHSN/NSSP inputs, build the ILINet extracts, run
each configured component model, combine them into the ensemble, validate the
result against the hub's requirements, and draw the check plot.

NB This will NOT commit or push to Git.
"""

import argparse
import concurrent.futures
import pathlib
import subprocess
import sys
import time

import yaml

HUB_PATHS = {
    'flu': ('cdcepi/FluSight-forecast-hub', 'Google_SAI-FluEns'),
    'rsv': ('CDCgov/rsv-forecast-hub', 'Google_SAI-RSVEns'),
}


def play_completion_sound():
  """Play a short notification for end of run."""
  if sys.platform == 'darwin':
    try:
      played = subprocess.run(
          ['/usr/bin/afplay', '/System/Library/Sounds/Glass.aiff'],
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
      if played.returncode == 0:
        return
    except OSError:
      pass
  print('\a', end='', flush=True)


def components(repo_root, pathogen):
  """The component list for this pathogen, from its config."""
  ensemble = yaml.safe_load(
      (repo_root / 'config' / f'{pathogen}.yaml').read_text())['ensemble']
  by_phase = ensemble.get('components_by_phase')
  if by_phase is None:
    return list(ensemble['components']), 'all phases'
  phase = ensemble['phase']
  if phase not in by_phase:
    raise ValueError(f'phase {phase!r} has no component list')
  return list(by_phase[phase]), phase


def step(name, argv, repo_root, quiet=False):
  """Run one pipeline command, echoing it so the run is reproducible by hand."""
  print(f'\n--- {name} ---', flush=True)
  print('   ' + ' '.join([pathlib.Path(sys.executable).name] + argv[1:]), flush=True)
  started = time.monotonic()
  result = subprocess.run(argv, cwd=repo_root, capture_output=quiet, text=True)
  if result.returncode != 0:
    if quiet and result.stdout:
      print(result.stdout[-2000:], flush=True)
    if quiet and result.stderr:
      print(result.stderr[-2000:], file=sys.stderr, flush=True)
    raise SystemExit(f'{name} failed (exit {result.returncode}); stopping.')
  return time.monotonic() - started


def main():
  root = pathlib.Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--pathogen', choices=sorted(HUB_PATHS), required=True)
  parser.add_argument('--reference_date', required=True,
                      help='the upcoming Saturday, YYYY-MM-DD')
  parser.add_argument('--repo_root', type=pathlib.Path, default=root)
  parser.add_argument('--jobs', type=int, default=6,
                      help='component models to run concurrently')
  parser.add_argument('--skip_inputs', action='store_true',
                      help='reuse the inputs already on disk')
  args = parser.parse_args()

  repo, py = args.repo_root, sys.executable
  ref, pathogen = args.reference_date, args.pathogen
  inputs = repo / 'data' / 'inputs' / pathogen
  outputs = repo / 'data' / 'outputs' / pathogen
  ensemble_dir = repo / 'data' / 'ensemble' / pathogen
  models, phase = components(repo, pathogen)
  hub_repo, model_id = HUB_PATHS[pathogen]

  print(f'{model_id}  reference date {ref}  [{phase}]  {len(models)} components', flush=True)

  if args.skip_inputs:
    print('\n--- inputs: reusing what is on disk (--skip_inputs) ---', flush=True)
  else:
    step('build inputs', [py, 'pipeline/build_inputs.py', '--pathogen', pathogen,
                          '--out_dir', str(inputs)], repo)
    step('build ILINet', [py, 'pipeline/build_ilinet.py', '--out_dir', str(inputs),
                          '--locations', str(inputs / 'locations.csv')], repo)

  print(f'\n--- run {len(models)} components ({args.jobs} at a time) ---', flush=True)
  outputs.mkdir(parents=True, exist_ok=True)
  failures, timings = [], {}

  def run_one(model):
    argv = [py, 'pipeline/run_model.py', '--hub', pathogen, '--model', model,
            '--reference_date', ref, '--input_dir', str(inputs),
            '--output_dir', str(outputs)]
    started = time.monotonic()
    done = subprocess.run(argv, cwd=repo, capture_output=True, text=True)
    return model, done, time.monotonic() - started

  with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
    for model, done, elapsed in pool.map(run_one, models):
      timings[model] = elapsed
      if done.returncode == 0:
        print(f'   OK   {model:<26} {elapsed:6.0f}s', flush=True)
      else:
        failures.append(model)
        tail = (done.stderr or done.stdout or '').strip().splitlines()[-1:]
        print(f'   FAIL {model:<26} {elapsed:6.0f}s  {tail[0] if tail else ""}'[:140], flush=True)

  if failures:
    raise SystemExit(
        f'\n{len(failures)} component(s) failed: {", ".join(failures)}.\n'
        'Stopping before the ensemble: the configured set is what was chosen,\n'
        'and silently ensembling the survivors would change the model.')

  step('ensemble', [py, 'pipeline/create_ensemble.py', '--pathogen', pathogen,
                    '--reference_date', ref, '--submission_dir', str(outputs),
                    '--out_dir', str(ensemble_dir)], repo)

  submission = ensemble_dir / f'{ref}-{model_id}.csv'
  step('validate', [py, 'pipeline/validate_submission.py', '--pathogen', pathogen,
                    '--reference_date', ref, '--submission', str(submission)], repo)

  figure = repo / 'data' / 'figures' / f'{pathogen}_{ref}.png'
  step('check plot', [py, 'pipeline/plot_forecasts.py', '--pathogen', pathogen,
                      '--reference_date', ref, '--input_dir', str(inputs),
                      '--submission', str(submission), '--out', str(figure)], repo)

  print(f"""
=== ready to submit ===
  file   {submission}
  figure {figure}
  to     {hub_repo}  ->  model-output/{model_id}/{ref}-{model_id}.csv

Review the figure before pushing: the fan should continue the observed trend,
not collapse. """)
  play_completion_sound()


if __name__ == '__main__':
  main()
