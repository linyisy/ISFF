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

"""Minimal synthetic example showing how an arbitrary picker can use ISFF.

This example is intentionally small and self-contained. It demonstrates the
BackboneAdapter interface, station-wise baseline inference, ISFF-augmented
inference, and the auxiliary Haversine distance objective. It does *not*
reproduce the manuscript experiments or reported performance numbers.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from isff.backbones.base import BackboneAdapter
from isff.distance import HaversineDistanceTargets, masked_distance_l1_loss
from isff.model import ISFFAugmentedPicker, SingleStationPicker


class ExampleBackbone(BackboneAdapter):
    """Small encoder/decoder picker used only to illustrate ISFF integration."""

    feature_dim = 8
    temporal_length = 16
    input_length = 256

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, self.feature_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.temporal_length),
        )
        self.output_head = nn.Conv1d(self.feature_dim, 3, kernel_size=1)

    def encode(self, waveforms: torch.Tensor):
        representation = self.encoder(waveforms)
        state = {"input_length": waveforms.shape[-1]}
        return representation, state

    def decode(self, representation: torch.Tensor, state):
        upsampled = F.interpolate(
            representation,
            size=state["input_length"],
            mode="linear",
            align_corners=False,
        )
        return torch.softmax(self.output_head(upsampled), dim=1)


def main() -> None:
    torch.manual_seed(7)

    batch_size = 1
    num_stations = 4

    # Synthetic three-component station waveforms: (B, N, C, T).
    waveforms = torch.randn(
        batch_size,
        num_stations,
        3,
        ExampleBackbone.input_length,
    )

    # Coordinates are ordered as [longitude, latitude] in degrees.
    lonlat = torch.tensor(
        [
            [
                [121.00, 24.00],
                [121.08, 24.04],
                [121.17, 24.10],
                [121.26, 24.15],
            ]
        ],
        dtype=torch.float32,
    )

    # Use identical initial backbone weights for the baseline and ISFF example.
    base_backbone = ExampleBackbone()
    single_model = SingleStationPicker(deepcopy(base_backbone)).eval()
    isff_model = ISFFAugmentedPicker(
        deepcopy(base_backbone),
        num_heads=4,
        attention_dropout=0.1,
    ).eval()

    with torch.no_grad():
        single_probabilities = single_model(waveforms)
        isff_probabilities, predicted_distances = isff_model(
            waveforms,
            lonlat,
        )
        target_distances = HaversineDistanceTargets()(lonlat)
        distance_loss = masked_distance_l1_loss(
            predicted_distances,
            target_distances,
        )

    expected_pairs = num_stations * (num_stations - 1) // 2

    assert single_probabilities.shape == (
        batch_size,
        num_stations,
        3,
        ExampleBackbone.input_length,
    )
    assert isff_probabilities.shape == single_probabilities.shape
    assert predicted_distances.shape == (batch_size, expected_pairs)
    assert target_distances.shape == predicted_distances.shape
    assert torch.isfinite(distance_loss)

    print("ISFF synthetic quickstart completed successfully.")
    print(f"Single-station probabilities: {tuple(single_probabilities.shape)}")
    print(f"ISFF probabilities:           {tuple(isff_probabilities.shape)}")
    print(f"Station-pair distances:       {tuple(predicted_distances.shape)}")
    print(f"Auxiliary distance L1 loss:   {distance_loss.item():.4f} km")


if __name__ == "__main__":
    main()
