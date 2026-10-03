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

import os
import subprocess
import sys
from pathlib import Path


def test_quickstart_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source_path, environment.get("PYTHONPATH")])
    )

    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "quickstart.py")],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    assert "ISFF synthetic quickstart completed successfully." in completed.stdout
    assert "Single-station probabilities:" in completed.stdout
    assert "Station-pair distances:" in completed.stdout
