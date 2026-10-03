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

import torch

from isff.factory import build_isff_picker, build_single_station_picker


def test_vit_factory_builds_single_and_multi_models():
    x = torch.randn(1, 3, 3, 3000)
    coords = torch.tensor([[[121.0, 24.0], [121.1, 24.1], [121.2, 24.2]]])
    multi = build_isff_picker("vit").eval()
    single = build_single_station_picker("vit").eval()
    with torch.no_grad():
        multi_pred, dist = multi(x, coords)
        single_pred = single(x)
    assert multi_pred[0].shape == (1, 3, 1, 3000)
    assert single_pred[0].shape == (1, 3, 1, 3000)
    assert dist.shape == (1, 3)
