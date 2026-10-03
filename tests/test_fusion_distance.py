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

from isff.distance import DistancePredictor, HaversineDistanceTargets, masked_distance_l1_loss
from isff.fusion import ISFF


def test_all_reported_interfaces():
    for d, length in [(32, 12), (16, 24), (40, 75)]:
        x = torch.randn(2, 4, d, length)
        c = torch.tensor([[[121.0, 24.0]] * 4] * 2)
        update, dist = ISFF(d, length)(x, c)
        assert update.shape == x.shape
        assert dist.shape == (2, 4, d * length)


def test_distance_shape_and_masked_loss():
    feat = torch.randn(2, 4, 384)
    pred = DistancePredictor(384)(feat)
    assert pred.shape == (2, 6)
    assert torch.isfinite(pred).all()
    coords = torch.tensor([
        [[121.0,24.0],[121.1,24.1],[121.2,24.2],[121.3,24.3]],
        [[121.0,24.0],[0.0,0.0],[121.2,24.2],[121.3,24.3]],
    ])
    target = HaversineDistanceTargets()(coords)
    assert torch.isnan(target[1]).any()
    assert torch.isfinite(masked_distance_l1_loss(pred, target))

