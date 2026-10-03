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

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .distance import masked_distance_l1_loss

DISTANCE_LOSS_WEIGHT = 2e-7


def phasenet_vector_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Point-wise vector cross entropy for PhaseNet N/P/S outputs."""
    if prediction.shape != target.shape:
        raise ValueError(
            "PhaseNet prediction and target shapes must match"
        )
    if prediction.shape[-2] != 3:
        raise ValueError(
            "PhaseNet prediction must have three N/P/S channels"
        )

    cross_entropy = target * torch.log(prediction + 1e-5)
    return -cross_entropy.mean(dim=-1).sum(dim=-1).mean()


def eqtransformer_multitask_bce(
    prediction: Sequence[torch.Tensor],
    picking_target: torch.Tensor,
    detection_target: torch.Tensor,
    *,
    task_weights: tuple[float, float, float] = (
        0.4,
        0.55,
        0.05,
    ),
    positive_weight: float = 0.89,
    negative_weight: float = 0.11,
    reduction: str = "mean",
) -> torch.Tensor:
    """EQTransformer P/S/detection binary cross-entropy objective."""
    if len(prediction) != 3:
        raise ValueError(
            "EQTransformer prediction must contain detection, P, and S"
        )
    if picking_target.shape[-2] < 2:
        raise ValueError(
            "picking_target must contain P and S channels"
        )
    if detection_target.shape[-2] != 1:
        raise ValueError(
            "detection_target must contain one detection channel"
        )
    if len(task_weights) != 3:
        raise ValueError("task_weights must contain three values")

    detection_prediction, p_prediction, s_prediction = prediction
    p_target = picking_target[..., 0, :]
    s_target = picking_target[..., 1, :]
    detection_target = detection_target[..., 0, :]

    if p_prediction.shape != p_target.shape:
        raise ValueError(
            "P prediction and target shapes must match"
        )
    if s_prediction.shape != s_target.shape:
        raise ValueError(
            "S prediction and target shapes must match"
        )
    if detection_prediction.shape != detection_target.shape:
        raise ValueError(
            "detection prediction and target shapes must match"
        )

    def weighted_bce(
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.full_like(
            target,
            negative_weight,
        )
        weights = torch.where(
            target != 0,
            torch.as_tensor(
                positive_weight,
                device=target.device,
                dtype=target.dtype,
            ),
            weights,
        )
        return F.binary_cross_entropy(
            predicted,
            target,
            weight=weights,
            reduction=reduction,
        )

    return (
        task_weights[0] * weighted_bce(p_prediction, p_target)
        + task_weights[1] * weighted_bce(s_prediction, s_target)
        + task_weights[2]
        * weighted_bce(
            detection_prediction,
            detection_target,
        )
    )


def vit_multitask_bce(
    prediction: Sequence[torch.Tensor],
    picking_target: torch.Tensor,
    detection_target: torch.Tensor,
    *,
    task_weights: tuple[float, float, float],
) -> torch.Tensor:
    """ViT P/S/detection BCE with validation-selected task weights."""
    if len(prediction) != 3:
        raise ValueError(
            "ViT prediction must contain P, S, and detection outputs"
        )
    if len(task_weights) != 3:
        raise ValueError("task_weights must contain three values")

    p_prediction, s_prediction, detection_prediction = prediction
    p_prediction = p_prediction.squeeze(-2)
    s_prediction = s_prediction.squeeze(-2)
    detection_prediction = detection_prediction.squeeze(-2)

    p_target = picking_target[..., 0, :]
    s_target = picking_target[..., 1, :]
    detection_target = detection_target[..., 0, :]

    if p_prediction.shape != p_target.shape:
        raise ValueError(
            "P prediction and target shapes must match"
        )
    if s_prediction.shape != s_target.shape:
        raise ValueError(
            "S prediction and target shapes must match"
        )
    if detection_prediction.shape != detection_target.shape:
        raise ValueError(
            "detection prediction and target shapes must match"
        )

    return (
        task_weights[0]
        * F.binary_cross_entropy(
            p_prediction,
            p_target,
        )
        + task_weights[1]
        * F.binary_cross_entropy(
            s_prediction,
            s_target,
        )
        + task_weights[2]
        * F.binary_cross_entropy(
            detection_prediction,
            detection_target,
        )
    )


def add_distance_objective(
    seismic_loss: torch.Tensor,
    predicted_distance: torch.Tensor,
    target_distance: torch.Tensor,
    *,
    distance_weight: float = DISTANCE_LOSS_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the auxiliary station-distance loss to a seismic loss."""
    if distance_weight < 0:
        raise ValueError("distance_weight must be non-negative")

    distance_loss = masked_distance_l1_loss(
        predicted_distance,
        target_distance,
    )
    total_loss = seismic_loss + distance_weight * distance_loss
    return total_loss, distance_loss
