"""Command-line entry point for the Phase 1 dataset audit."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .data_audit import audit_dataset, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a credit-risk CSV dataset.")
    parser.add_argument("dataset", type=Path, help="Path to the CSV dataset")
    parser.add_argument("--target", default=None, help="Target column (auto-detected by default)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = audit_dataset(load_dataset(args.dataset), args.target)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
