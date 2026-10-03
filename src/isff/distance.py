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
import torch.nn.functional as F

_PAIR_CHUNK_SIZE = 2048


class DistancePredictor(nn.Module):
    """MLP for auxiliary pairwise station-distance prediction."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, int] = (128, 64),
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if len(hidden_dims) != 2 or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain two positive integers")

        hidden1, hidden2 = hidden_dims
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim * 2, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, station_features: torch.Tensor) -> torch.Tensor:
        if station_features.ndim != 3:
            raise ValueError(
                "station_features must have shape (B, N, feature_dim)"
            )
        if station_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, "
                f"got {station_features.shape[-1]}"
            )

        batch_size, num_stations, _ = station_features.shape
        pair_indices = torch.triu_indices(
            num_stations,
            num_stations,
            offset=1,
            device=station_features.device,
        )
        num_pairs = pair_indices.shape[1]
        if num_pairs == 0:
            return station_features.new_empty((batch_size, 0))

        outputs: list[torch.Tensor] = []
        for start in range(0, num_pairs, _PAIR_CHUNK_SIZE):
            end = min(start + _PAIR_CHUNK_SIZE, num_pairs)
            indices = pair_indices[:, start:end]
            first = station_features[:, indices[0], :]
            second = station_features[:, indices[1], :]
            pair_features = torch.cat([first, second], dim=-1)
            outputs.append(self.network(pair_features.float()).squeeze(-1))

        return torch.cat(outputs, dim=1)


class HaversineDistanceTargets(nn.Module):
    """Compute unique pairwise great-circle distances in kilometers.

    Input coordinates are ordered as ``[longitude, latitude]`` in degrees.
    Coordinate pairs that are missing or represented by zeros are excluded
    from the auxiliary loss by returning NaN targets.
    """

    def __init__(self, radius_km: float = 6371.0):
        super().__init__()
        if radius_km <= 0:
            raise ValueError("radius_km must be positive")
        self.radius_km = float(radius_km)

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        if lonlat.ndim != 3 or lonlat.shape[-1] != 2:
            raise ValueError("lonlat must have shape (B, N, 2)")

        lonlat = lonlat.to(dtype=torch.float32)
        longitude = lonlat[..., 0]
        latitude = lonlat[..., 1]
        valid_station = (
            torch.isfinite(longitude)
            & torch.isfinite(latitude)
            & (longitude != 0.0)
            & (latitude != 0.0)
        )

        longitude = torch.deg2rad(longitude)
        latitude = torch.deg2rad(latitude)

        num_stations = lonlat.shape[1]
        pair_indices = torch.triu_indices(
            num_stations,
            num_stations,
            offset=1,
            device=lonlat.device,
        )
        first, second = pair_indices[0], pair_indices[1]

        delta_latitude = latitude[:, second] - latitude[:, first]
        delta_longitude = longitude[:, second] - longitude[:, first]

        haversine = (
            torch.sin(delta_latitude / 2.0).pow(2)
            + torch.cos(latitude[:, first])
            * torch.cos(latitude[:, second])
            * torch.sin(delta_longitude / 2.0).pow(2)
        )
        haversine = torch.clamp(haversine, min=0.0, max=1.0)
        distance = (
            2.0
            * self.radius_km
            * torch.arcsin(torch.sqrt(haversine))
        )

        valid_pair = valid_station[:, first] & valid_station[:, second]
        return distance.masked_fill(~valid_pair, float("nan"))


def masked_distance_l1_loss(
    predicted_distance: torch.Tensor,
    target_distance: torch.Tensor,
) -> torch.Tensor:
    """Return mean absolute distance error over valid unique station pairs."""
    if predicted_distance.shape != target_distance.shape:
        raise ValueError(
            "predicted_distance and target_distance shapes must match"
        )

    valid = torch.isfinite(target_distance)
    if not valid.any():
        return predicted_distance.sum() * 0.0

    return F.l1_loss(
        predicted_distance[valid],
        target_distance[valid],
        reduction="mean",
    )
