"""Run all eight validation-only studies and freeze them before any test evaluation (#56).

Each model runner (LR, XGBoost, MLP, GRU) runs its two condition studies into
``OUTPUT/<model>``. ``--parallel N`` runs up to N model runners at the same time
as separate processes; every study stays single-threaded and seeded, so results
do not depend on N. ``frozen_studies.json`` is written only when all eight
studies are complete; it is the input for the held-out test evaluation (#60/#62).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {'logistic_regression': 'train_logistic.py', 'xgboost': 'train_xgboost.py',
           'mlp': 'train_mlp.py', 'gru': 'train_gru.py'}
CONDITIONS = ('latest_year', 'five_year')


def run_model(model, csv, config, output, preprocessor):
    directory = output / model
    command = [sys.executable, str(ROOT / 'scripts' / RUNNERS[model]), '--csv', str(csv),
               '--config', str(config), '--output', str(directory)]
    if preprocessor is not None:
        command += ['--preprocessor', str(preprocessor)]
    started = perf_counter()
    with open(output / f'{model}.log', 'w', encoding='utf-8') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    return model, code, perf_counter() - started


def collect(output, elapsed):
    studies, missing = {}, []
    for model in RUNNERS:
        path = output / model / 'frozen_selections.json'
        selections = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        for condition in CONDITIONS:
            if condition not in selections:
                missing.append(f'{model}/{condition}')
                continue
            summary = json.loads((output / model / condition / 'study_summary.json').read_text(encoding='utf-8'))
            selection = selections[condition]
            studies.setdefault(model, {})[condition] = {
                'study_name': selection['study_name'], 'trial_number': selection['trial_number'],
                'validation_average_precision': selection['validation_average_precision'],
                'threshold': selection['threshold'], 'threshold_source': selection['threshold_source'],
                'attempted': summary['attempted'], 'completed': summary['completed'],
                'failed': summary['failed'], 'study_seconds': summary['elapsed_seconds'],
                'directory': f'{model}/{condition}'}
    return studies, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--config', default=ROOT / 'configs/config.yaml', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preprocessor', type=Path)
    parser.add_argument('--parallel', type=int, default=1, help='Model runners at once (1-4)')
    args = parser.parse_args(argv)
    if not 1 <= args.parallel <= len(RUNNERS):
        raise ValueError('--parallel must be between 1 and 4')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing studies are never overwritten')
    output.mkdir(parents=True)
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(lambda m: run_model(m, args.csv.resolve(), args.config.resolve(), output,
                                                    args.preprocessor and args.preprocessor.resolve()),
                                RUNNERS))
    runs = {m: {'exit_code': code, 'seconds': seconds, 'log': f'{m}.log'} for m, code, seconds in results}
    studies, missing = collect(output, perf_counter() - started)
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
