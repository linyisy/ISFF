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

from isff.backbones.vit import ViTMLP, ViTSeismicAdapter


def test_vit_interface_and_output_shapes():
    model = ViTSeismicAdapter().eval()
    x = torch.randn(2, 3, 3000)
    with torch.no_grad():
        rep, state = model.encode(x)
        assert rep.shape == (2, 40, 75)
        p, s, d = model.decode(rep, state)
        assert p.shape == s.shape == d.shape == (2, 1, 3000)



class _IdentityAttention(torch.nn.Module):
    def forward(self, query, key, value):
        return query


class _CountingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return inputs


def test_vit_feedforward_path_matches_experiment_implementation():
    block = ViTMLP(4, 40, 200, 100).eval()
    block.multiheadattention = _IdentityAttention()
    first = _CountingLayer()
    second = _CountingLayer()
    block.fullyconnected1 = first
    block.fullyconnected2 = second
    with torch.no_grad():
        output = block(torch.randn(2, 75, 40))
    assert output.shape == (2, 75, 40)
    assert first.calls == 2
    assert second.calls == 0
