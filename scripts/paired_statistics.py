# Copyright 2026 Yi-Syuan Lin and Kuan-Yu Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from isff.statistics import paired_error_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the paired absolute-error statistics used in the manuscript."
        )
    )
    parser.add_argument("single_csv", type=Path)
    parser.add_argument("isff_csv", type=Path)
    parser.add_argument("--id-column", default="trace_id")
    parser.add_argument("--error-column", default="abs_error")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_error_csv(
    path: Path,
    *,
    id_column: str,
    error_column: str,
) -> dict[str, float]:
    """Read one unique absolute-error value per reference-station case."""
    errors: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for required in (id_column, error_column):
            if required not in fieldnames:
                raise ValueError(
                    f"{path} does not contain column {required!r}"
                )

        for row_number, row in enumerate(reader, start=2):
            case_id = row[id_column]
            if case_id in errors:
                raise ValueError(
                    f"duplicate {id_column}={case_id!r} in {path}"
                )
            try:
                errors[case_id] = float(row[error_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid {error_column!r} at {path}:{row_number}"
                ) from exc
    return errors


def main() -> None:
    args = parse_args()
    single = read_error_csv(
        args.single_csv,
        id_column=args.id_column,
        error_column=args.error_column,
    )
    isff = read_error_csv(
        args.isff_csv,
        id_column=args.id_column,
        error_column=args.error_column,
    )

    matched_ids = [
        case_id
        for case_id in single
        if case_id in isff
    ]
    if not matched_ids:
        raise ValueError("the input files have no matched case IDs")

    single_errors = np.asarray(
        [single[case_id] for case_id in matched_ids],
        dtype=float,
    )
    isff_errors = np.asarray(
        [isff[case_id] for case_id in matched_ids],
        dtype=float,
    )
    statistics = paired_error_statistics(
        single_errors,
        isff_errors,
        bootstrap_resamples=args.bootstrap,
        random_seed=args.seed,
    )

    print(f"Matched N: {statistics.matched_n}")
    print(
        "Mean paired reduction: "
        f"{statistics.mean_reduction:.8f}"
    )
    print(
        "95% bootstrap CI: "
        f"[{statistics.ci_lower:.8f}, "
        f"{statistics.ci_upper:.8f}]"
    )
    print(
        "One-sided Wilcoxon p: "
        f"{statistics.wilcoxon_p:.12g}"
    )


if __name__ == "__main__":
    main()
