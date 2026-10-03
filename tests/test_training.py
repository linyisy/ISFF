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

from isff.training import eqtransformer_optimizer_at_milestone


def test_eqt_milestone_optimizer_contains_all_supplied_parameters():
    a = torch.nn.Linear(2, 2)
    b = torch.nn.Linear(2, 2)
    c = torch.nn.Linear(2, 1)
    params = list(a.parameters()) + list(b.parameters()) + list(c.parameters())
    opt = eqtransformer_optimizer_at_milestone(params, 21)
    assert opt is not None
    assert opt.param_groups[0]["lr"] == 1e-4
    expected = sum(p.numel() for p in params)
    actual = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert actual == expected
