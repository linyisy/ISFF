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

import torch
import torch.nn as nn


class StationCoordinateEmbedding(nn.Module):
    """Sinusoidal station-coordinate embedding used by ISFF.

    Coordinates are ordered as ``[longitude, latitude]`` and used in their
    original degree units. For feature dimension ``d`` and temporal length
    ``L``, the output shape is ``(B, N, d, L)``.
    """

    def __init__(self, feature_dim: int, temporal_length: int):
        super().__init__()
        if feature_dim <= 0 or feature_dim % 4 != 0:
            raise ValueError(
                "feature_dim must be positive and divisible by 4"
            )
        if temporal_length <= 0:
            raise ValueError("temporal_length must be positive")

        self.feature_dim = int(feature_dim)
        self.temporal_length = int(temporal_length)

        frequency_index = torch.arange(
            self.feature_dim // 4,
            dtype=torch.float32,
        )
        omega = 10000.0 ** (
            -4.0 * frequency_index / self.feature_dim
        )
        self.register_buffer("omega", omega)
        self.register_buffer(
            "temporal_index",
            torch.arange(self.temporal_length, dtype=torch.float32),
        )

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        if lonlat.ndim != 3 or lonlat.shape[-1] != 2:
            raise ValueError("lonlat must have shape (B, N, 2)")

        lonlat = lonlat.to(dtype=self.omega.dtype)
        longitude = lonlat[..., 0].unsqueeze(-1).unsqueeze(-1)
        latitude = lonlat[..., 1].unsqueeze(-1).unsqueeze(-1)
        omega = self.omega.view(1, 1, -1, 1)
        temporal_index = self.temporal_index.view(1, 1, 1, -1)

        longitude_phase = longitude * omega * temporal_index
        latitude_phase = latitude * omega * temporal_index

        return torch.cat(
            [
                torch.sin(longitude_phase),
                torch.cos(longitude_phase),
                torch.sin(latitude_phase),
                torch.cos(latitude_phase),
            ],
            dim=2,
        )


class CrossStationAttention(nn.Module):
    """Apply self-attention across stations at each latent time index."""

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (B, N, d, L)")

        batch_size, num_stations, feature_dim, temporal_length = (
            features.shape
        )
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected feature dimension {self.feature_dim}, "
                f"got {feature_dim}"
            )
        if num_stations <= 0:
            raise ValueError("num_stations must be positive")

        station_sequence = (
            features.permute(0, 3, 1, 2)
            .contiguous()
            .view(
                batch_size * temporal_length,
                num_stations,
                feature_dim,
            )
        )
        attended, _ = self.attention(
            station_sequence,
            station_sequence,
            station_sequence,
            need_weights=False,
        )
        return (
            attended.view(
                batch_size,
                temporal_length,
                num_stations,
                feature_dim,
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )


class ISFF(nn.Module):
    """Backbone-compatible inter-station feature fusion module."""

    def __init__(
        self,
        feature_dim: int,
        temporal_length: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.temporal_length = int(temporal_length)
        flat_dim = self.feature_dim * self.temporal_length

        self.coordinate_embedding = StationCoordinateEmbedding(
            feature_dim=self.feature_dim,
            temporal_length=self.temporal_length,
        )
        self.cross_station_attention = CrossStationAttention(
            feature_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.picking_projection = nn.Linear(flat_dim, flat_dim)
        self.distance_projection = nn.Linear(flat_dim, flat_dim)

    @property
    def distance_feature_dim(self) -> int:
        return self.feature_dim * self.temporal_length

    def forward(
        self,
        features: torch.Tensor,
        lonlat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("features must have shape (B, N, d, L)")

        batch_size, num_stations, feature_dim, temporal_length = (
            features.shape
        )
        if (
            feature_dim,
            temporal_length,
        ) != (
            self.feature_dim,
            self.temporal_length,
        ):
            raise ValueError(
                "unexpected ISFF interface shape: "
                f"expected (*, *, {self.feature_dim}, "
                f"{self.temporal_length}), got {tuple(features.shape)}"
            )
        if lonlat.shape != (batch_size, num_stations, 2):
            raise ValueError(
                "lonlat must have shape (B, N, 2) matching features"
            )

        coordinate_embedding = self.coordinate_embedding(lonlat).to(
            dtype=features.dtype
        )
        location_aware = features + coordinate_embedding
        fused = self.cross_station_attention(location_aware)
        flattened = fused.reshape(
            batch_size,
            num_stations,
            feature_dim * temporal_length,
        )

        picking_update = self.picking_projection(flattened).reshape(
            batch_size,
            num_stations,
            feature_dim,
            temporal_length,
        )
        distance_features = self.distance_projection(flattened)
        return picking_update, distance_features
