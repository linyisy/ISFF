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

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class PairedStatistics:
    matched_n: int
    mean_reduction: float
    ci_lower: float
    ci_upper: float
    wilcoxon_p: float


def paired_error_statistics(
    single_abs_error_seconds: np.ndarray,
    isff_abs_error_seconds: np.ndarray,
    *,
    bootstrap_resamples: int = 10_000,
    random_seed: int = 42,
) -> PairedStatistics:
    """Compute the manuscript's paired timing-error statistics.

    The Wilcoxon alternative is directional: single-station absolute errors
    are larger than the corresponding ISFF absolute errors. The bootstrap
    confidence interval characterizes the mean paired reduction.
    """
    single = np.asarray(single_abs_error_seconds, dtype=float)
    isff = np.asarray(isff_abs_error_seconds, dtype=float)

    if single.ndim != 1 or isff.ndim != 1:
        raise ValueError("paired error arrays must be one-dimensional")
    if single.shape != isff.shape:
        raise ValueError("paired error arrays must have identical shape")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")

    valid = np.isfinite(single) & np.isfinite(isff)
    single = single[valid]
    isff = isff[valid]
    if single.size == 0:
        raise ValueError("no valid paired errors")

    reduction = single - isff
    wilcoxon_p = float(
        wilcoxon(
            single,
            isff,
            alternative="greater",
        ).pvalue
    )

    rng = np.random.default_rng(random_seed)
    num_pairs = reduction.size
    bootstrap_means = np.empty(
        bootstrap_resamples,
        dtype=float,
    )

    chunk_size = 256
    for start in range(
        0,
        bootstrap_resamples,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            bootstrap_resamples,
        )
        sample_indices = rng.integers(
            0,
            num_pairs,
            size=(end - start, num_pairs),
        )
        bootstrap_means[start:end] = reduction[
            sample_indices
        ].mean(axis=1)

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )
    return PairedStatistics(
        matched_n=int(num_pairs),
        mean_reduction=float(reduction.mean()),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        wilcoxon_p=wilcoxon_p,
    )
