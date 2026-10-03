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
import torch.nn.functional as F

from .base import BackboneAdapter


class ConvBlock(nn.Module):
    """Convolution block matching the ViT-based seismic backbone."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.batchnorm(self.conv1d(inputs)))


class ResidualConvBlock(nn.Module):
    """Three-convolution residual block used by the ViT backbone."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels, in_channels, kernel_size=11)
        self.conv_block2 = ConvBlock(in_channels, in_channels, kernel_size=11)
        self.conv_block3 = ConvBlock(in_channels, out_channels, kernel_size=11)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_block1(inputs)
        out2 = self.conv_block2(out1)
        out2 = out2 + inputs
        return self.dropout(self.conv_block3(out2))


class OutputConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Conv1d(1, 1, kernel_size=15, padding="same", stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.conv_block1(inputs))


class PatchEmbedding(nn.Module):
    """Patch extraction equivalent to the explicit loop in the experimental implementation."""

    def __init__(self, patch_size: int = 40):
        super().__init__()
        self.patch_size = int(patch_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, wavelength = inputs.shape
        if wavelength % self.patch_size != 0:
            raise ValueError("ViT input length must be divisible by patch_size")
        patches = inputs.unfold(-1, self.patch_size, self.patch_size)
        return (
            patches.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, wavelength // self.patch_size, channels * self.patch_size)
        )


class Embedding(nn.Module):
    def __init__(self, patch_size: int, emb_size: int, in_channel: int):
        super().__init__()
        self.embedding = nn.Linear(patch_size * in_channel, emb_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(inputs))


class ViTMultiHeadAttention(nn.Module):
    """Custom attention used by the ViT-based backbone configuration.

    This is the backbone's internal attention, distinct from ISFF attention.
    Tensor layout and scaling follow the implementation used in the experiments.
    """

    def __init__(self, num_heads: int, in_channel: int):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = in_channel // num_heads
        self.query = nn.Linear(in_channel, in_channel, bias=False)
        self.key = nn.Linear(in_channel, in_channel, bias=False)
        # Keep the parameter name used by the ViT implementation.
        self.vector = nn.Linear(in_channel, in_channel, bias=False)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.query(q).view(batch, len_q, self.num_heads, self.head_dim)
        k = self.key(k).view(batch, len_k, self.num_heads, self.head_dim)
        v = self.vector(v).view(batch, len_v, self.num_heads, self.head_dim)
        k = k.transpose(2, 3)
        output = torch.matmul(q, k) / (len_k ** 0.5)
        output = self.dropout(F.softmax(output, dim=-1))
        output = torch.matmul(output, v)
        output = self.dropout(output)
        return output.view(batch, len_q, -1)


class ViTMLP(nn.Module):
    """Feed-forward block used by the ViT-based backbone configuration."""

    def __init__(self, num_heads: int, in_channel: int, neurons1: int, neurons2: int):
        super().__init__()
        self.multiheadattention = ViTMultiHeadAttention(num_heads, in_channel)
        self.dropout = nn.Dropout(p=0.1)
        self.fullyconnected1 = nn.Sequential(
            nn.Linear(in_channel, neurons1),
            nn.ReLU(),
            nn.Linear(neurons1, in_channel),
        )
        self.fullyconnected2 = nn.Sequential(
            nn.Linear(in_channel, neurons2),
            nn.ReLU(),
            nn.Linear(neurons2, in_channel),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.multiheadattention(inputs, inputs, inputs)
        output = self.fullyconnected1(inputs)
        output = self.dropout(output)
        output = self.fullyconnected1(output)
        return self.dropout(output)


class ViTTransformerEncoder(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        self.norm_layer = nn.LayerNorm(in_channel, eps=1e-5)
        self.multiheadattention = ViTMultiHeadAttention(4, 40)
        self.mlp = ViTMLP(4, 40, 200, 100)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.permute(0, 2, 1)
        output = self.norm_layer(inputs)
        output = self.multiheadattention(output, output, output)
        output1 = output + inputs
        output = self.norm_layer(output1)
        output = self.mlp(output)
        return output + output1


class ViTSeismicAdapter(BackboneAdapter):
    """Encode/decode adapter faithful to the ViT-based seismic model.

    The encoder returns a backbone-compatible ISFF interface of ``(B, 40, 75)``.
    The decoder flattens that representation directly to 3,000 samples before
    applying the P, S, and detection output heads.
    """

    feature_dim = 40
    temporal_length = 75
    input_length = 3000

    def __init__(self):
        super().__init__()
        self.convblock_b1 = ResidualConvBlock(3, 10)
        self.convblock_b2 = ResidualConvBlock(10, 20)
        self.convblock_b3 = ResidualConvBlock(20, 40)
        self.patching = PatchEmbedding(40)
        self.embedding = Embedding(40, 40, 40)
        self.pos_embedding = nn.Parameter(torch.randn(1, 75, 40))
        self.convblock_b4_1 = ResidualConvBlock(40, 40)
        self.convblock_b4_2 = ResidualConvBlock(40, 40)
        self.convblock_b4_3 = ResidualConvBlock(40, 40)
        self.convblock_b4_4 = ResidualConvBlock(40, 40)
        self.transformer_encoder1 = ViTTransformerEncoder(40)
        self.transformer_encoder2 = ViTTransformerEncoder(40)
        self.transformer_encoder3 = ViTTransformerEncoder(40)
        self.transformer_encoder4 = ViTTransformerEncoder(40)
        self.p_convblock_b5 = OutputConvBlock()
        self.s_convblock_b5 = OutputConvBlock()
        self.detection_convblock_b5 = OutputConvBlock()

    def encode(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, Any]:
        output = self.convblock_b1(waveforms)
        output = self.convblock_b2(output)
        output = self.convblock_b3(output)
        output = self.patching(output)
        output = self.embedding(output)
        output = output + self.pos_embedding

        output = output.permute(0, 2, 1)
        output = self.convblock_b4_1(output)
        output = self.transformer_encoder1(output)
        output = output.permute(0, 2, 1)
        output = self.convblock_b4_2(output)
        output = self.transformer_encoder2(output)
        output = output.permute(0, 2, 1)
        output = self.convblock_b4_3(output)
        output = self.transformer_encoder3(output)
        output = output.permute(0, 2, 1)
        output = self.convblock_b4_4(output)
        output = self.transformer_encoder4(output)  # B,75,40

        representation = output.permute(0, 2, 1).contiguous()  # B,40,75
        if representation.shape[1:] != (self.feature_dim, self.temporal_length):
            raise RuntimeError(f"unexpected ViT interface shape {tuple(representation.shape)}")
        return representation, None

    def decode(self, representation: torch.Tensor, state: Any):
        if representation.shape[1:] != (self.feature_dim, self.temporal_length):
            raise ValueError(
                f"expected representation (*,{self.feature_dim},{self.temporal_length})"
            )
        output = representation.flatten(start_dim=1, end_dim=2).unsqueeze(1)
        return (
            self.p_convblock_b5(output),
            self.s_convblock_b5(output),
            self.detection_convblock_b5(output),
        )

