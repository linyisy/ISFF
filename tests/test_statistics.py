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

import numpy as np

from isff.statistics import paired_error_statistics


def test_paired_statistics_direction():
    single = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    isff = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = paired_error_statistics(single, isff, bootstrap_resamples=200, random_seed=1)
    assert result.matched_n == 5
    assert abs(result.mean_reduction - 0.1) < 1e-12
    assert result.ci_lower > 0
