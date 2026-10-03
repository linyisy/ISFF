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

from isff.losses import (
    add_distance_objective,
    eqtransformer_multitask_bce,
    phasenet_vector_cross_entropy,
    vit_multitask_bce,
)


def test_phasenet_loss_is_finite():
    logits = torch.randn(2, 3, 20)
    prediction = torch.softmax(logits, dim=1)
    target = torch.softmax(torch.randn(2, 3, 20), dim=1)
    assert torch.isfinite(phasenet_vector_cross_entropy(prediction, target))


def test_eqtransformer_loss_accepts_station_axis():
    prediction = [
        torch.full((2, 3, 20), 0.5),
        torch.full((2, 3, 20), 0.5),
        torch.full((2, 3, 20), 0.5),
    ]
    picking_target = torch.zeros(2, 3, 3, 20)
    detection_target = torch.zeros(2, 3, 1, 20)
    picking_target[..., 0, 5] = 1.0
    picking_target[..., 1, 10] = 1.0
    detection_target[..., 0, 4:12] = 1.0
    loss = eqtransformer_multitask_bce(
        prediction,
        picking_target,
        detection_target,
    )
    assert torch.isfinite(loss)


def test_vit_loss_uses_explicit_task_weights():
    prediction = tuple(
        torch.full((2, 3, 1, 20), 0.5)
        for _ in range(3)
    )
    picking_target = torch.zeros(2, 3, 3, 20)
    detection_target = torch.zeros(2, 3, 1, 20)
    loss = vit_multitask_bce(
        prediction,
        picking_target,
        detection_target,
        task_weights=(0.75, 0.20, 0.05),
    )
    assert torch.isfinite(loss)


def test_distance_objective_returns_total_and_auxiliary_loss():
    seismic_loss = torch.tensor(2.0)
    predicted = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[100.0, 100.0]])
    total, auxiliary = add_distance_objective(
        seismic_loss,
        predicted,
        target,
    )
    assert torch.isfinite(total)
    assert auxiliary.item() == 100.0
    assert total.item() > seismic_loss.item()
