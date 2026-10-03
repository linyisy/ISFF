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

from .base import BackboneAdapter


def _seisbench_models():
    try:
        import seisbench.models as sbm
    except ImportError as exc:  # pragma: no cover - external dependency
        raise ImportError(
            "PhaseNet and EQTransformer adapters require SeisBench 0.2.3."
        ) from exc
    return sbm


class PhaseNetAdapter(BackboneAdapter):
    """Expose the deepest PhaseNet encoder representation (32 x 12)."""

    feature_dim = 32
    temporal_length = 12
    input_length = 3001

    def __init__(self):
        super().__init__()
        sbm = _seisbench_models()
        self.model = sbm.PhaseNet(
            in_channels=3,
            classes=3,
            phases="NPS",
        )

    def encode(
        self,
        waveforms: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        model = self.model
        x_in = model.activation(
            model.in_bn(model.inc(waveforms))
        )
        x1 = model.activation(
            model.bnd1(model.conv1(x_in))
        )
        x2 = model.activation(
            model.bnd2(model.conv2(x1))
        )
        x3 = model.activation(
            model.bnd3(model.conv3(x2))
        )
        x4 = model.activation(
            model.bnd4(model.conv4(x3))
        )

        expected = (
            self.feature_dim,
            self.temporal_length,
        )
        if x4.shape[1:] != expected:
            raise RuntimeError(
                "unexpected PhaseNet interface shape: "
                f"{tuple(x4.shape)}"
            )
        return x4, (x_in, x1, x2, x3)

    def decode(
        self,
        representation: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        model = self.model
        x_in, x1, x2, x3 = state
        x = torch.cat(
            [
                model.activation(
                    model.bnu1(model.up1(representation))
                ),
                x3,
            ],
            dim=1,
        )
        x = torch.cat(
            [
                model.activation(model.bnu2(model.up2(x))),
                x2,
            ],
            dim=1,
        )
        x = torch.cat(
            [
                model.activation(model.bnu3(model.up3(x))),
                x1,
            ],
            dim=1,
        )
        x = torch.cat(
            [
                model.activation(model.bnu4(model.up4(x))),
                x_in,
            ],
            dim=1,
        )
        return model.softmax(model.out(x))


class EQTransformerAdapter(BackboneAdapter):
    """Expose the final shared EQTransformer representation (16 x 24)."""

    feature_dim = 16
    temporal_length = 24
    input_length = 3000

    def __init__(self):
        super().__init__()
        sbm = _seisbench_models()
        self.model = sbm.EQTransformer(
            in_samples=self.input_length
        )

    def encode(
        self,
        waveforms: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        model = self.model
        x = model.encoder(waveforms)
        x = model.res_cnn_stack(x)
        x = model.bi_lstm_stack(x)
        x, _ = model.transformer_d0(x)
        x, _ = model.transformer_d(x)

        expected = (
            self.feature_dim,
            self.temporal_length,
        )
        if x.shape[1:] != expected:
            raise RuntimeError(
                "unexpected EQTransformer interface shape: "
                f"{tuple(x.shape)}"
            )
        return x, None

    def decode(
        self,
        representation: torch.Tensor,
        state: Any,
    ):
        del state
        model = self.model

        detection = model.decoder_d(representation)
        detection = torch.sigmoid(
            model.conv_d(detection)
        ).squeeze(1)
        outputs = [detection]

        for lstm, attention, decoder, conv in zip(
            model.pick_lstms,
            model.pick_attentions,
            model.pick_decoders,
            model.pick_convs,
            strict=True,
        ):
            pick = representation.permute(2, 0, 1)
            pick = lstm(pick)[0]
            pick = model.dropout(pick)
            pick = pick.permute(1, 2, 0)
            pick, _ = attention(pick)
            pick = decoder(pick)
            outputs.append(
                torch.sigmoid(conv(pick)).squeeze(1)
            )

        # SeisBench EQTransformer output order: detection, P, S.
        return outputs
