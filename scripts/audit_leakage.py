"""Run the leakage audit (#52) on the audited data, persisted split and saved preprocessor.

Prerequisites: ``scripts/prepare_data.py`` and ``scripts/prepare_preprocessing.py``.
Writes ``results/metrics/leakage_audit.json``; exits 1 if any check fails.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_loader import download_dataset, load_data  # noqa: E402
from leakage import LeakageError, audit  # noqa: E402
from preprocessing import FittedPreprocessor, file_sha256, load_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, help="Local CSV (default: KaggleHub download)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--preprocessor", type=Path, default=REPO_ROOT / "results" / "models" / "preprocessor.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "metrics" / "leakage_audit.json")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    csv = args.csv or download_dataset()
    if file_sha256(csv) != config["dataset"]["sha256"]:
        print("Dataset differs from the audited CSV", file=sys.stderr)
        return 1
    if not args.preprocessor.exists():
        print(f"{args.preprocessor} is missing. Run scripts/prepare_preprocessing.py first.", file=sys.stderr)
        return 1
    split = config["split"]
    companies = load_splits(REPO_ROOT / split["directory"], expected_sha256=split["sha256"])
    preprocessor = FittedPreprocessor.load(args.preprocessor)

    try:
        report = audit(load_data(csv), companies, preprocessor, config)
    except LeakageError as error:
        print(f"Leakage check failed: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("All leakage checks passed.")
    print(json.dumps({"indirect_signals": report["indirect_signals"],
                      "record_availability": report["record_availability"]}, indent=2))
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
