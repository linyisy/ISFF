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

from typing import Literal

from .backbones.base import BackboneAdapter
from .model import ISFFAugmentedPicker, SingleStationPicker

BackboneName = Literal["phasenet", "eqtransformer", "vit"]


def build_backbone(name: BackboneName) -> BackboneAdapter:
    key = name.lower()
    if key == "vit":
        from .backbones.vit import ViTSeismicAdapter
        return ViTSeismicAdapter()
    if key in {"phasenet", "eqtransformer"}:
        from .backbones.seisbench_adapters import PhaseNetAdapter, EQTransformerAdapter
        return PhaseNetAdapter() if key == "phasenet" else EQTransformerAdapter()
    raise ValueError(f"unsupported backbone: {name!r}")


def build_isff_picker(
    name: BackboneName,
    *,
    num_heads: int = 4,
    attention_dropout: float = 0.1,
) -> ISFFAugmentedPicker:
    return ISFFAugmentedPicker(
        build_backbone(name),
        num_heads=num_heads,
        attention_dropout=attention_dropout,
    )


def build_single_station_picker(name: BackboneName) -> SingleStationPicker:
    return SingleStationPicker(build_backbone(name))
