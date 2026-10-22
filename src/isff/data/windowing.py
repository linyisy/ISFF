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

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np


def _slice_with_padding(
    waveform: np.ndarray,
    start: int,
    length: int,
) -> np.ndarray:
    """Slice the last axis, padding with zeros outside available samples."""
    result = np.zeros(
        (*waveform.shape[:-1], length),
        dtype=waveform.dtype,
    )
    source_start = max(0, start)
    source_end = min(
        waveform.shape[-1],
        start + length,
    )
    if source_end <= source_start:
        return result

    destination_start = source_start - start
    destination_end = destination_start + (
        source_end - source_start
    )
    result[..., destination_start:destination_end] = waveform[
        ..., source_start:source_end
    ]
    return result


def construct_common_window(
    waveforms: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    reference_index: int,
    *,
    output_length: int = 3000,
    placement: int | None = None,
    rng: np.random.Generator | None = None,
    p_key: str = "trace_p_arrival_sample",
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Construct the common event window used by all station slots.

    The reference-station P arrival determines one common temporal window.
    Exactly the same shift and crop are then applied to every station slot;
    non-reference phase annotations are not used for alignment.
    """
    waveforms = np.asarray(waveforms)
    if waveforms.ndim != 3:
        raise ValueError(
            "waveforms must have shape (num_stations, channels, time)"
        )
    if len(metadata) != waveforms.shape[0]:
        raise ValueError(
            "metadata must contain one entry per station waveform"
        )
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    if not 0 <= reference_index < len(metadata):
        raise IndexError("reference_index out of range")

    if rng is None:
        rng = np.random.default_rng()

    reference_p = metadata[reference_index].get(p_key)
    if reference_p is None:
        raise ValueError(
            "reference station must have a valid P-wave annotation"
        )
    try:
        reference_p_value = float(reference_p)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference station must have a valid P-wave annotation"
        ) from exc
    if not np.isfinite(reference_p_value):
        raise ValueError(
            "reference station must have a valid P-wave annotation"
        )
    reference_p_sample = int(reference_p_value)

    staging_length = 6000
    staging_reference_sample = 3000
    staging_start = reference_p_sample - staging_reference_sample

    if placement is None:
        max_crop_start = staging_length - output_length
        if max_crop_start < 0:
            raise ValueError(
                "output_length cannot exceed the staging window length"
            )
        crop_start = int(
            rng.integers(
                0,
                max_crop_start + 1,
            )
        )
    else:
        placement = int(placement)
        if not 0 <= placement < output_length:
            raise ValueError(
                "placement must lie inside the output window"
            )
        crop_start = staging_reference_sample - placement
        if (
            crop_start < 0
            or crop_start + output_length > staging_length
        ):
            raise ValueError(
                "placement is incompatible with the staging window"
            )

    total_start = staging_start + crop_start
    windowed = np.stack(
        [
            _slice_with_padding(
                waveform,
                total_start,
                output_length,
            )
            for waveform in waveforms
        ]
    )

    shifted_metadata = [
        deepcopy(dict(item))
        for item in metadata
    ]
    for item in shifted_metadata:
        for key, value in list(item.items()):
            if not key.endswith("_sample") or value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                item[key] = numeric_value - total_start

    return windowed, shifted_metadata, total_start


def normalize_waveforms(waveforms: np.ndarray) -> np.ndarray:
    """Demean and standard-deviation normalize each component over time."""
    waveforms = np.asarray(
        waveforms,
        dtype=np.float32,
    )
    mean = waveforms.mean(
        axis=-1,
        keepdims=True,
    )
    std = waveforms.std(
        axis=-1,
        keepdims=True,
    )
    std = np.where(std == 0, 1.0, std)
    return (waveforms - mean) / std


def reference_station_indices(
    metadata: Sequence[Mapping[str, Any]],
    *,
    p_key: str = "trace_p_arrival_sample",
) -> list[int]:
    """Return stations with valid P labels for reference-case evaluation."""
    indices: list[int] = []
    for index, item in enumerate(metadata):
        value = item.get(p_key)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            indices.append(index)
    return indices
