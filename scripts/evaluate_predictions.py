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
from pathlib import Path

import numpy as np

from isff.evaluation import evaluate_pwave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate P-wave probability traces using the manuscript protocol."
        )
    )
    parser.add_argument(
        "npz",
        type=Path,
        help="NPZ file containing 'predictions' and 'labels' arrays",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=["max", "single", "continue", "avg"],
        default="max",
    )
    parser.add_argument(
        "--trigger-length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=100.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.npz) as data:
        if "predictions" not in data or "labels" not in data:
            raise KeyError(
                "NPZ must contain 'predictions' and 'labels' arrays"
            )
        result = evaluate_pwave(
            data["predictions"],
            data["labels"],
            threshold=args.threshold,
            mode=args.mode,
            trigger_length=args.trigger_length,
            tolerance_samples=args.tolerance_samples,
            sampling_rate_hz=args.sampling_rate,
        )

    print(
        f"TP={result.tp} FP={result.fp} "
        f"TN={result.tn} FN={result.fn}"
    )
    print(f"Precision={result.precision:.6f}")
    print(f"Recall={result.recall:.6f}")
    print(f"F1={result.f1:.6f}")
    print(f"MAE={result.mae_seconds:.6f} s")
    print(
        "AbsErrorSD="
        f"{result.absolute_error_sd_seconds:.6f} s"
    )


if __name__ == "__main__":
    main()
