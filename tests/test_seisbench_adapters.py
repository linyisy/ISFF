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

import pytest
import torch

pytest.importorskip("obspy")
pytest.importorskip("seisbench")

from isff.backbones.seisbench_adapters import PhaseNetAdapter, EQTransformerAdapter


def test_phasenet_interface_shape():
    m = PhaseNetAdapter().eval()
    with torch.no_grad():
        rep, state = m.encode(torch.randn(2, 3, 3001))
        assert rep.shape == (2, 32, 12)
        out = m.decode(rep, state)
        assert out.shape == (2, 3, 3001)


def test_eqtransformer_interface_shape():
    m = EQTransformerAdapter().eval()
    with torch.no_grad():
        rep, state = m.encode(torch.randn(1, 3, 3000))
        assert rep.shape == (1, 16, 24)
        out = m.decode(rep, state)
        assert len(out) == 3
        assert all(x.shape == (1, 3000) for x in out)
