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

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BackboneAdapter(nn.Module, ABC):
    """Minimal interface required by the generic ISFF wrapper."""

    feature_dim: int
    temporal_length: int
    input_length: int

    @abstractmethod
    def encode(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Encode station-wise waveforms of shape (B, C, T)."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, representation: torch.Tensor, state: Any):
        """Decode a fused representation back through the original prediction path."""
        raise NotImplementedError
