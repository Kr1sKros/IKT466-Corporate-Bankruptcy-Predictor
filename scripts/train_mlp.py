"""Tune latest-year/five-year MLP on validation only; defer final test evaluation (#56)."""
import argparse
from pathlib import Path

from models.neural_study import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--config', default=Path(__file__).resolve().parents[1] / 'configs/config.yaml', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preprocessor', type=Path,
                        help='Shared #46/#87 artifact (default: results/models/preprocessor.json in the config repository)')
    args = parser.parse_args()
    result = run('mlp', args.csv, args.config, args.output, args.preprocessor)
    print({name: selection['validation_average_precision'] for name, selection in result['selections'].items()})


if __name__ == '__main__':
    main()
