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

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

WaveformSample = tuple[np.ndarray, Mapping[str, Any]]


def station_id(metadata: Mapping[str, Any]) -> str:
    """Return a canonical ``network.station.location`` identifier."""
    network = metadata.get("station_network_code")
    station = metadata.get("station_code")
    location = metadata.get("station_location_code")

    if network is None or station is None:
        return ""

    network = str(network)
    station = str(station)
    location = "" if location is None else str(location)
    if not network or not station:
        return ""

    return f"{network}.{station}.{location}"


def event_id_from_metadata(
    metadata: Mapping[str, Any],
    *,
    event_id_key: str,
    prefix_length: int | None = None,
) -> str:
    """Extract the event identifier used for event/station noise lookup."""
    value = metadata.get(event_id_key)
    if value is None:
        return ""
    value = str(value)
    return (
        value[:prefix_length]
        if prefix_length is not None
        else value
    )


def build_fixed_station_tensor(
    event_samples: Sequence[WaveformSample],
    station_order: Sequence[str],
    noise_samples: Sequence[WaveformSample],
    station_coordinates: Mapping[str, tuple[float, float]],
    *,
    event_id: str | None = None,
    noise_event_id_key: str = "trace_name_original",
    noise_event_id_prefix_length: int | None = 10,
    category_key: str = "trace_category",
    noise_category_value: str = "noise",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build a fixed station tensor using hierarchical noise imputation.

    Missing event observations are filled in this order: same-event/same-
    station noise, same-station noise from another event, then noise from
    another station while retaining the target slot's station geometry.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not station_order:
        raise ValueError("station_order must not be empty")
    if len(set(station_order)) != len(station_order):
        raise ValueError("station_order must contain unique station IDs")
    if not noise_samples:
        raise ValueError("noise_samples must not be empty")

    event_by_station: dict[str, WaveformSample] = {}
    for waveform, metadata in event_samples:
        sid = station_id(metadata)
        if not sid:
            continue
        if sid in event_by_station:
            raise ValueError(
                f"duplicate event waveform for station {sid}"
            )
        event_by_station[sid] = (
            np.asarray(waveform),
            metadata,
        )

    noise_by_station: dict[str, list[WaveformSample]] = defaultdict(list)
    noise_by_event_station: dict[
        tuple[str, str],
        list[WaveformSample],
    ] = defaultdict(list)

    normalized_noise_samples: list[WaveformSample] = []
    for waveform, metadata in noise_samples:
        waveform = np.asarray(waveform)
        normalized_noise_samples.append((waveform, metadata))
        sid = station_id(metadata)
        if sid:
            noise_by_station[sid].append(
                (waveform, metadata)
            )

        noise_event_id = event_id_from_metadata(
            metadata,
            event_id_key=noise_event_id_key,
            prefix_length=noise_event_id_prefix_length,
        )
        if noise_event_id and sid:
            noise_by_event_station[
                (noise_event_id, sid)
            ].append((waveform, metadata))

    selected_waveforms: list[np.ndarray] = []
    selected_metadata: list[dict[str, Any]] = []

    for sid in station_order:
        if sid in event_by_station:
            waveform, metadata = event_by_station[sid]
            selected_waveforms.append(waveform)
            selected_metadata.append(
                deepcopy(dict(metadata))
            )
            continue

        same_event_noise = noise_by_event_station.get(
            (event_id or "", sid),
            [],
        )
        same_station_noise = noise_by_station.get(
            sid,
            [],
        )

        if same_event_noise:
            donor_pool = same_event_noise
        elif same_station_noise:
            donor_pool = same_station_noise
        else:
            donor_pool = normalized_noise_samples

        donor_index = int(
            rng.integers(len(donor_pool))
        )
        waveform, donor_metadata = donor_pool[donor_index]
        metadata = deepcopy(dict(donor_metadata))
        metadata[category_key] = noise_category_value

        if sid not in station_coordinates:
            raise KeyError(
                f"missing coordinates for station slot {sid}"
            )
        longitude, latitude = station_coordinates[sid]
        metadata["station_longitude_deg"] = float(longitude)
        metadata["station_latitude_deg"] = float(latitude)

        parts = sid.split(".")
        if len(parts) != 3:
            raise ValueError(
                "station IDs must use network.station.location format"
            )
        network, station, location = parts
        metadata["station_network_code"] = network
        metadata["station_code"] = station
        metadata["station_location_code"] = location

        selected_waveforms.append(waveform)
        selected_metadata.append(metadata)

    first_shape = selected_waveforms[0].shape
    if len(first_shape) != 2:
        raise ValueError(
            "waveforms must have shape (channels, time)"
        )
    channels = first_shape[0]
    for waveform in selected_waveforms:
        if waveform.ndim != 2:
            raise ValueError(
                "waveforms must have shape (channels, time)"
            )
        if waveform.shape[0] != channels:
            raise ValueError(
                "all waveforms must have the same number of channels"
            )

    max_length = max(
        waveform.shape[-1]
        for waveform in selected_waveforms
    )
    tensor = np.zeros(
        (
            len(station_order),
            channels,
            max_length,
        ),
        dtype=np.result_type(
            *[
                waveform.dtype
                for waveform in selected_waveforms
            ]
        ),
    )
    for index, waveform in enumerate(selected_waveforms):
        tensor[
            index,
            :,
            : waveform.shape[-1],
        ] = waveform

    return tensor, selected_metadata
