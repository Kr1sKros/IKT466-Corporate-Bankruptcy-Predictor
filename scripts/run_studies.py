"""Run all eight validation-only studies and freeze them before any test evaluation (#56).

Every model x condition study runs as its own process (LR, XGBoost, MLP, GRU
runners with ``--conditions``) into ``OUTPUT/<model>__<condition>``.
``--parallel N`` runs up to N studies at the same time; every study stays
single-threaded, CPU-only and seeded, so results do not depend on N. Use one
free CPU core per parallel study. ``frozen_studies.json`` is written only when
all eight studies are complete; it is the input for the held-out test
evaluation (#60/#62).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {'logistic_regression': 'train_logistic.py', 'xgboost': 'train_xgboost.py',
           'mlp': 'train_mlp.py', 'gru': 'train_gru.py'}
CONDITIONS = ('latest_year', 'five_year')
# Slowest first, so a long GRU study never starts last.
JOBS = [(m, c) for m in ('gru', 'mlp', 'xgboost', 'logistic_regression') for c in ('five_year', 'latest_year')]


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def single_thread_env():
    """Each study uses one CPU thread for every numeric library and sees no GPU.

    Containers often show the host's cores (e.g. 96) while allowing only a few;
    libraries would then start dozens of threads per study and slow everything.
    The studies are CPU-only by design, so GPUs are hidden; this also avoids
    failures from a GPU driver that is older than the installed PyTorch build.
    """
    import os
    return {**os.environ, **{name: '1' for name in THREAD_VARIABLES}, 'CUDA_VISIBLE_DEVICES': ''}


def job_name(model, condition):
    return f'{model}__{condition}'


def run_job(job, csv, config, output, preprocessor, started_at=None):
    model, condition = job
    if started_at is not None:
        started_at[job] = perf_counter()
    command = [sys.executable, str(ROOT / 'scripts' / RUNNERS[model]), '--csv', str(csv),
               '--config', str(config), '--output', str(output / job_name(*job)),
               '--conditions', condition]
    if preprocessor is not None:
        command += ['--preprocessor', str(preprocessor)]
    started = perf_counter()
    with open(output / f'{job_name(*job)}.log', 'w', encoding='utf-8') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=single_thread_env()).returncode
    return job, code, perf_counter() - started


def _clock(seconds):
    if seconds is None:
        return '-'
    seconds = int(seconds)
    return f'{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def progress_table(output, budget, started_at, finished, now):
    """One line per study: trials done, failures, best validation AP, elapsed, ETA."""
    lines = [f'{"study":32s} {"trials":>9s} {"failed":>6s} {"best AP":>8s} {"elapsed":>8s} {"ETA":>8s}  status']
    for job in JOBS:
        name = job_name(*job)
        path = output / name / job[1] / 'trials.json'
        try:
            records = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        except (OSError, ValueError):  # being rewritten right now
            records = []
        done = len(records)
        failed = sum(r['state'] == 'FAIL' for r in records)
        values = [r['value'] for r in records if r['state'] == 'COMPLETE']
        best = f'{max(values):.4f}' if values else '-'
        start = started_at.get(job)
        elapsed = (finished[job][1] if job in finished else now - start) if start is not None else None
        eta = elapsed / done * (budget - done) if done and job not in finished else None
        if job in finished:
            status = 'done' if finished[job][0] == 0 else f'FAILED (exit {finished[job][0]}, see {name}.log)'
        else:
            status = 'running' if start is not None else 'waiting'
        lines.append(f'{name:32s} {done:>4d}/{budget:<4d} {failed:>6d} {best:>8s} '
                     f'{_clock(elapsed):>8s} {_clock(eta):>8s}  {status}')
    return '\n'.join(lines)


def collect(output):
    studies, missing = {}, []
    for model in RUNNERS:
        for condition in CONDITIONS:
            run_directory = output / job_name(model, condition)
            path = run_directory / 'frozen_selections.json'
            selections = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            if set(selections) != {condition}:
                missing.append(f'{model}/{condition}')
                continue
            summary = json.loads((run_directory / condition / 'study_summary.json').read_text(encoding='utf-8'))
            selection = selections[condition]
            studies.setdefault(model, {})[condition] = {
                'study_name': selection['study_name'], 'trial_number': selection['trial_number'],
                'validation_average_precision': selection['validation_average_precision'],
                'threshold': selection['threshold'], 'threshold_source': selection['threshold_source'],
                'attempted': summary['attempted'], 'completed': summary['completed'],
                'failed': summary['failed'], 'study_seconds': summary['elapsed_seconds'],
                'directory': f'{job_name(model, condition)}/{condition}'}
    return studies, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--config', default=ROOT / 'configs/config.yaml', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preprocessor', type=Path)
    parser.add_argument('--parallel', type=int, default=1, help='Studies at once (1-8), one CPU core each')
    parser.add_argument('--progress-every', type=float, default=30., help='Seconds between progress tables')
    args = parser.parse_args(argv)
    if not 1 <= args.parallel <= len(JOBS):
        raise ValueError('--parallel must be between 1 and 8')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing studies are never overwritten')
    output.mkdir(parents=True)
    preprocessor = args.preprocessor.resolve() if args.preprocessor else None
    budget = yaml.safe_load(args.config.read_text(encoding='utf-8'))['study_plan']['tuning']['attempted_trials_per_study']
    print(f'Running {len(JOBS)} studies x {budget} trials, {args.parallel} at a time. Output: {output}', flush=True)
    started = perf_counter()
    started_at, finished, results = {}, {}, []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_job, job, args.csv.resolve(), args.config.resolve(), output,
                               preprocessor, started_at): job for job in JOBS}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=args.progress_every)
            for future in done:
                job, code, seconds = future.result()
                finished[job] = (code, seconds)
                results.append((job, code, seconds))
                print(f'[{datetime.now():%H:%M:%S}] {job_name(*job)} '
                      f'{"finished" if code == 0 else f"FAILED (exit {code})"} after {_clock(seconds)}',
                      flush=True)
            print(f'\n[{datetime.now():%H:%M:%S}] {len(finished)}/{len(JOBS)} studies finished, '
                  f'total {_clock(perf_counter() - started)}', flush=True)
            print(progress_table(output, budget, started_at, finished, perf_counter()), flush=True)
    results.sort(key=lambda r: JOBS.index(r[0]))
    runs = {job_name(*job): {'exit_code': code, 'seconds': seconds, 'log': f'{job_name(*job)}.log'}
            for job, code, seconds in results}
    studies, missing = collect(output)
    report = {'runs': runs, 'studies': studies, 'parallel': args.parallel,
              'elapsed_seconds': perf_counter() - started, 'missing': missing,
              'test_evaluation': 'not_run_here'}
    if missing or any(r['exit_code'] for r in runs.values()):
        report['status'] = 'incomplete'
        (output / 'run_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        raise SystemExit(f'Not all eight studies completed ({", ".join(missing) or "runner error"}); '
                         f'see {output / "run_report.json"} and the logs. No frozen_studies.json written.')
    report['status'] = 'all_eight_frozen'
    (output / 'frozen_studies.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for model, conditions in studies.items():
        print(model, {c: round(s['validation_average_precision'], 4) for c, s in conditions.items()})


if __name__ == '__main__':
    main()
