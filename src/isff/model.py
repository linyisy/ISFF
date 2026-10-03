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

from typing import Any

import torch
import torch.nn as nn

from .backbones.base import BackboneAdapter
from .distance import DistancePredictor
from .fusion import ISFF


class SingleStationPicker(nn.Module):
    """Station-wise baseline using the same backbone adapter as ISFF."""

    def __init__(self, backbone: BackboneAdapter):
        super().__init__()
        self.backbone = backbone

    def forward(self, waveforms: torch.Tensor):
        if waveforms.ndim == 3:
            self._validate_input_length(waveforms.shape[-1])
            representation, state = self.backbone.encode(waveforms)
            return self.backbone.decode(representation, state)

        if waveforms.ndim != 4:
            raise ValueError(
                "waveforms must have shape (B, C, T) or (B, N, C, T)"
            )

        batch_size, num_stations, channels, input_length = waveforms.shape
        self._validate_input_length(input_length)
        stationwise = waveforms.reshape(
            batch_size * num_stations,
            channels,
            input_length,
        )
        representation, state = self.backbone.encode(stationwise)
        predictions = self.backbone.decode(representation, state)
        return _restore_station_axis(
            predictions,
            batch_size,
            num_stations,
        )

    def _validate_input_length(self, input_length: int) -> None:
        if input_length != self.backbone.input_length:
            raise ValueError(
                f"{type(self.backbone).__name__} expects input length "
                f"{self.backbone.input_length}, got {input_length}"
            )


class ISFFAugmentedPicker(nn.Module):
    """Multi-station ISFF wrapper around a compatible picker backbone."""

    def __init__(
        self,
        backbone: BackboneAdapter,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.isff = ISFF(
            feature_dim=backbone.feature_dim,
            temporal_length=backbone.temporal_length,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.distance_predictor = DistancePredictor(
            self.isff.distance_feature_dim
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        lonlat: torch.Tensor,
    ):
        if waveforms.ndim != 4:
            raise ValueError("waveforms must have shape (B, N, C, T)")

        batch_size, num_stations, channels, input_length = waveforms.shape
        if input_length != self.backbone.input_length:
            raise ValueError(
                f"{type(self.backbone).__name__} expects input length "
                f"{self.backbone.input_length}, got {input_length}"
            )
        if lonlat.shape != (batch_size, num_stations, 2):
            raise ValueError("lonlat must have shape (B, N, 2)")

        stationwise = waveforms.reshape(
            batch_size * num_stations,
            channels,
            input_length,
        )
        encoded_flat, state = self.backbone.encode(stationwise)

        expected_shape = (
            self.backbone.feature_dim,
            self.backbone.temporal_length,
        )
        if encoded_flat.shape[1:] != expected_shape:
            raise RuntimeError(
                "backbone adapter returned an unexpected interface shape: "
                f"expected (*, {expected_shape[0]}, {expected_shape[1]}), "
                f"got {tuple(encoded_flat.shape)}"
            )

        encoded = encoded_flat.reshape(
            batch_size,
            num_stations,
            *expected_shape,
        )
        update, distance_features = self.isff(encoded, lonlat)

        fused_flat = (encoded + update).reshape(
            batch_size * num_stations,
            *expected_shape,
        )
        predictions_flat = self.backbone.decode(fused_flat, state)
        predictions = _restore_station_axis(
            predictions_flat,
            batch_size,
            num_stations,
        )
        predicted_distance = self.distance_predictor(distance_features)
        return predictions, predicted_distance


def _restore_station_axis(
    output: Any,
    batch_size: int,
    num_stations: int,
):
    if torch.is_tensor(output):
        return output.reshape(
            batch_size,
            num_stations,
            *output.shape[1:],
        )

    if isinstance(output, tuple):
        return tuple(
            _restore_station_axis(item, batch_size, num_stations)
            for item in output
        )

    if isinstance(output, list):
        return [
            _restore_station_axis(item, batch_size, num_stations)
            for item in output
        ]

    raise TypeError(
        f"unsupported backbone output type: {type(output)!r}"
    )
