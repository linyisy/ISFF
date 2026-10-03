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

from isff.data.windowing import construct_common_window, normalize_waveforms


def test_fixed_reference_placement_preserves_moveout():
    x = np.zeros((2, 3, 7000), dtype=np.float32)
    meta = [
        {"trace_p_arrival_sample": 3500, "trace_s_arrival_sample": 3900},
        {"trace_p_arrival_sample": 3600, "trace_s_arrival_sample": 4000},
    ]
    out, shifted, start = construct_common_window(x, meta, 0, output_length=3000, placement=1500)
    assert out.shape == (2, 3, 3000)
    assert shifted[0]["trace_p_arrival_sample"] == 1500
    assert shifted[1]["trace_p_arrival_sample"] == 1600
    assert shifted[1]["trace_p_arrival_sample"] - shifted[0]["trace_p_arrival_sample"] == 100


def test_normalize_zero_trace_is_finite():
    x = np.zeros((2, 3, 10), dtype=np.float32)
    y = normalize_waveforms(x)
    assert np.isfinite(y).all()


def test_reference_station_indices_only_valid_p_labels():
    from isff.data.windowing import reference_station_indices
    meta = [
        {"trace_p_arrival_sample": 100},
        {"trace_p_arrival_sample": float("nan")},
        {"trace_p_arrival_sample": 250},
        {},
    ]
    assert reference_station_indices(meta) == [0, 2]
