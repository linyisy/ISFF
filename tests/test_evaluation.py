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

from isff.evaluation import evaluate_pwave, trigger_index


def test_trigger_modes_use_reported_run_end_semantics():
    p = np.array([0.0, 0.6, 0.7, 0.8, 0.0])
    assert trigger_index(p, 0.5, "single") == 1
    assert trigger_index(p, 0.5, "max") == 3
    assert trigger_index(p, 0.5, "continue", 3) == 3
    assert trigger_index(p, 0.5, "avg", 3) == 3


def test_timing_error_includes_outside_tolerance():
    pred = np.zeros((1, 200))
    gt = np.zeros((1, 200))
    pred[0, 150] = 1
    gt[0, 20] = 1
    res = evaluate_pwave(pred, gt, threshold=0.5, mode="max", tolerance_samples=50)
    assert res.fp == 1
    assert res.absolute_errors == [130]
    assert res.mae_seconds == 1.3
    assert res.absolute_error_sd_seconds == 0.0


def test_select_reference_station_cases():
    from isff.evaluation import select_reference_station_cases

    values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    selected = select_reference_station_cases(values, [2, 0])
    assert np.array_equal(selected[0], values[0, 2])
    assert np.array_equal(selected[1], values[1, 0])
