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

from pathlib import Path

from isff.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_configs_have_confirmed_interfaces():
    for name in ("cwasn.yaml", "instance.yaml"):
        cfg = load_yaml(ROOT / "configs" / name)
        assert cfg["interfaces"]["PhaseNet"] == {
            "feature_dim": 32,
            "temporal_length": 12,
            "input_length": 3001,
        }
        assert cfg["interfaces"]["EQTransformer"] == {
            "feature_dim": 16,
            "temporal_length": 24,
            "input_length": 3000,
        }
        assert cfg["interfaces"]["ViT"] == {
            "feature_dim": 40,
            "temporal_length": 75,
            "input_length": 3000,
        }


def test_controlled_protocol_has_all_five_positions():
    expected = [750, 1500, 2000, 2750, 2900]
    for name in ("controlled_cwasn.yaml", "controlled_instance.yaml"):
        cfg = load_yaml(ROOT / "configs" / name)
        assert cfg["positions"] == expected
