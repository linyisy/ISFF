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

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

TriggerMode = Literal["max", "single", "continue", "avg"]


@dataclass(frozen=True)
class EvaluationResult:
    """P-wave detection and picking metrics for one evaluation set."""

    tp: int
    fp: int
    tn: int
    fn: int
    signed_errors: list[int]
    absolute_errors: list[int]
    pick_records: list[dict[str, Any]]
    sampling_rate_hz: float

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    @property
    def mae_samples(self) -> float:
        return (
            float(np.mean(self.absolute_errors))
            if self.absolute_errors
            else float("nan")
        )

    @property
    def absolute_error_sd_samples(self) -> float:
        return (
            float(np.std(self.absolute_errors))
            if self.absolute_errors
            else float("nan")
        )

    @property
    def mae_seconds(self) -> float:
        return self.mae_samples / self.sampling_rate_hz

    @property
    def absolute_error_sd_seconds(self) -> float:
        return self.absolute_error_sd_samples / self.sampling_rate_hz


def trigger_index(
    probability: np.ndarray,
    threshold: float,
    mode: TriggerMode,
    trigger_length: int | None = None,
) -> int | None:
    """Return the trigger index used by the manuscript evaluation protocol."""
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1:
        raise ValueError("probability must be a one-dimensional time series")
    if p.size == 0:
        return None
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if mode == "max":
        index = int(np.argmax(p))
        return index if p[index] >= threshold else None

    if mode == "single":
        indices = np.flatnonzero(p >= threshold)
        return int(indices[0]) if indices.size else None

    if trigger_length is None or trigger_length <= 0:
        raise ValueError(f"mode {mode!r} requires trigger_length > 0")

    if mode == "avg":
        if p.size < trigger_length:
            return None
        kernel = np.ones(trigger_length, dtype=float) / trigger_length
        rolling_mean = np.convolve(p, kernel, mode="valid")
        indices = np.flatnonzero(rolling_mean >= threshold)
        return int(indices[0] + trigger_length - 1) if indices.size else None

    if mode == "continue":
        run_length = 0
        for index, is_above in enumerate(p >= threshold):
            run_length = run_length + 1 if is_above else 0
            if run_length == trigger_length:
                return index
        return None

    raise ValueError(f"unknown mode {mode!r}")


def select_reference_station_cases(
    values: np.ndarray,
    reference_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Select the designated reference-station output from each evaluation case."""
    values = np.asarray(values)
    reference_indices = np.asarray(reference_indices, dtype=int)

    if values.ndim < 2:
        raise ValueError("values must have shape (num_cases, num_stations, ...)")
    if reference_indices.shape != (values.shape[0],):
        raise ValueError("reference_indices must contain one index per case")
    if np.any(reference_indices < 0) or np.any(
        reference_indices >= values.shape[1]
    ):
        raise IndexError("reference station index out of range")

    return values[np.arange(values.shape[0]), reference_indices]


def evaluate_pwave(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    mode: TriggerMode = "max",
    trigger_length: int | None = None,
    tolerance_samples: int = 50,
    sampling_rate_hz: float = 100.0,
    metadata: Sequence[dict[str, Any]] | None = None,
) -> EvaluationResult:
    """Evaluate P-wave triggering and picking.

    Timing errors are recorded whenever both the prediction and ground-truth
    label trigger, including cases outside the tolerance used for precision,
    recall, and F1.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.shape != labels.shape or predictions.ndim != 2:
        raise ValueError(
            "predictions and labels must both have shape (num_cases, time)"
        )
    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if metadata is not None and len(metadata) != predictions.shape[0]:
        raise ValueError("metadata must contain one entry per evaluation case")

    tp = fp = tn = fn = 0
    signed_errors: list[int] = []
    absolute_errors: list[int] = []
    records: list[dict[str, Any]] = []

    for case_index, (prediction, label) in enumerate(
        zip(predictions, labels)
    ):
        true_index = int(np.argmax(label)) if not np.all(label == 0) else None
        if true_index is not None and label[true_index] < 0.3:
            true_index = None

        predicted_index = trigger_index(
            prediction,
            threshold,
            mode,
            trigger_length,
        )

        if (
            true_index is not None
            and predicted_index is not None
            and abs(predicted_index - true_index) <= tolerance_samples
        ):
            tp += 1
            outcome = "tp"
        elif predicted_index is not None:
            fp += 1
            outcome = "fp"
        elif true_index is not None:
            fn += 1
            outcome = "fn"
        else:
            tn += 1
            outcome = "tn"

        if true_index is not None and predicted_index is not None:
            error = int(predicted_index - true_index)
            signed_errors.append(error)
            absolute_errors.append(abs(error))
        else:
            error = None

        meta = metadata[case_index] if metadata is not None else {}
        records.append(
            {
                "trace_id": meta.get("trace_id", case_index),
                "event_id": meta.get("event_id"),
                "station_id": meta.get("station_id"),
                "true_p_sample": true_index,
                "pred_p_sample": predicted_index,
                "error": error,
                "abs_error": abs(error) if error is not None else None,
                "outcome": outcome,
            }
        )

    return EvaluationResult(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        signed_errors=signed_errors,
        absolute_errors=absolute_errors,
        pick_records=records,
        sampling_rate_hz=float(sampling_rate_hz),
    )
