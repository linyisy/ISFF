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

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class EarlyStopping:
    """Validation-loss early stopping with manuscript patience setting."""

    patience: int = 7
    best: float = float("inf")
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    def update(self, validation_loss: float) -> bool:
        if validation_loss < self.best:
            self.best = float(validation_loss)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def eqtransformer_lr_for_epoch(
    epoch: int,
    initial_lr: float = 1e-3,
) -> float | None:
    """Return the EQTransformer learning rate at a scheduled milestone."""
    if initial_lr <= 0:
        raise ValueError("initial_lr must be positive")

    factors = {
        21: 1e-1,
        41: 1e-2,
        61: 1e-3,
        91: 0.5e-3,
    }
    factor = factors.get(int(epoch))
    return (
        None
        if factor is None
        else float(initial_lr * factor)
    )


def eqtransformer_optimizer_at_milestone(
    parameters: Iterable[torch.nn.Parameter],
    epoch: int,
    *,
    initial_lr: float = 1e-3,
) -> torch.optim.Optimizer | None:
    """Recreate Adam at an EQTransformer learning-rate milestone.

    The parameter iterable must include the backbone, ISFF module, and
    auxiliary distance predictor, matching the reported training setup.
    """
    learning_rate = eqtransformer_lr_for_epoch(
        epoch,
        initial_lr=initial_lr,
    )
    if learning_rate is None:
        return None

    parameter_list = list(parameters)
    if not parameter_list:
        raise ValueError("parameters must not be empty")
    return torch.optim.Adam(
        parameter_list,
        lr=learning_rate,
    )
