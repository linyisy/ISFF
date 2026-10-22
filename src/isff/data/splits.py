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

CWASN_SPLIT = {
    "train": set(range(2012, 2019)),
    "validation": {2019},
    "test": {2020, 2021},
}

INSTANCE_SPLIT = {
    "train": set(range(2005, 2013)) | {2016, 2017},
    "validation": {2014, 2019},
    "test": {2013, 2015, 2018, 2020},
}


def split_for_year(year: int, split_spec: dict[str, set[int]]) -> str:
    matches = [name for name, years in split_spec.items() if int(year) in years]
    if len(matches) != 1:
        raise ValueError(f"year {year} maps to {matches}; split must be unique")
    return matches[0]
