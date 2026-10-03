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

import numpy as np

from isff.data.station_tensor import build_fixed_station_tensor


def _meta(sid, category, lon, lat, event="E1"):
    net, sta, loc = sid.split(".")
    return {
        "station_network_code": net,
        "station_code": sta,
        "station_location_code": loc,
        "trace_category": category,
        "station_longitude_deg": lon,
        "station_latitude_deg": lat,
        "trace_name_original": event,
    }


def test_random_noise_retains_target_slot_coordinates():
    station_order = ["NW.A.00", "NW.B.00"]
    event = [(np.ones((3, 10), dtype=np.float32), _meta("NW.A.00", "earthquake", 121, 24))]
    noise = [(np.zeros((3, 10), dtype=np.float32), _meta("NW.A.00", "noise", 121, 24, "N1"))]
    coords = {"NW.A.00": (121.0, 24.0), "NW.B.00": (122.0, 25.0)}
    x, meta = build_fixed_station_tensor(
        event, station_order, noise, coords, event_id="E1", rng=np.random.default_rng(1)
    )
    assert x.shape == (2, 3, 10)
    assert meta[1]["station_longitude_deg"] == 122.0
    assert meta[1]["station_latitude_deg"] == 25.0
    assert meta[1]["station_code"] == "B"


def test_instance_style_metadata_keys_are_supported():
    station_order = ["IV.A.00", "IV.B.00"]
    event = [(
        np.ones((3, 10), dtype=np.float32),
        {
            "station_network_code": "IV",
            "station_code": "A",
            "station_location_code": "00",
            "source_type": "earthquake",
            "source_id": "E1",
        },
    )]
    noise = [(
        np.zeros((3, 10), dtype=np.float32),
        {
            "station_network_code": "IV",
            "station_code": "B",
            "station_location_code": "00",
            "source_type": "noise",
            "source_id": "E1",
        },
    )]
    coords = {"IV.A.00": (12.0, 42.0), "IV.B.00": (13.0, 43.0)}
    _, meta = build_fixed_station_tensor(
        event,
        station_order,
        noise,
        coords,
        event_id="E1",
        noise_event_id_key="source_id",
        noise_event_id_prefix_length=None,
        category_key="source_type",
        noise_category_value="noise",
        rng=np.random.default_rng(0),
    )
    assert meta[1]["source_type"] == "noise"
    assert meta[1]["station_code"] == "B"
    assert meta[1]["station_longitude_deg"] == 13.0
    assert meta[1]["station_latitude_deg"] == 43.0
