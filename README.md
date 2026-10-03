# ISFF: Inter-Station Feature Fusion for Seismic P-Wave Picking

This repository contains the reference implementation accompanying **“A Backbone-Compatible Inter-Station Feature Fusion Framework for Seismic P-Wave Picking.”**

ISFF augments an existing single-station seismic picker at a high-level intermediate representation. Station-coordinate embeddings are added to the backbone features, multi-head self-attention exchanges information across stations at each latent temporal position, and the resulting update is returned to the original downstream prediction path through residual fusion. A pairwise station-distance objective provides weak auxiliary spatial supervision during training.

The manuscript evaluates exactly three backbones:

| Backbone | ISFF interface `(d, L)` | Input length | ISFF heads | Attention dropout |
|---|---:|---:|---:|---:|
| PhaseNet | `(32, 12)` | 3001 | 4 | 0.1 |
| EQTransformer | `(16, 24)` | 3000 | 4 | 0.1 |
| ViT | `(40, 75)` | 3000 | 4 | 0.1 |

The auxiliary distance predictor uses ReLU hidden nonlinearities and a linear scalar output.

## Repository layout

```text
configs/                    Dataset, training, and validation-selected evaluation settings
results/                    Results reported in the manuscript
scripts/                    P-wave evaluation and paired statistical analysis
examples/                   Executable synthetic ISFF quickstart
docs/                       User guide for attaching ISFF to another backbone
src/isff/
  backbones/                PhaseNet, EQTransformer, and ViT interfaces
  data/                     Fixed-station tensor and common-window construction
  distance.py               Pairwise distance prediction and Haversine targets
  evaluation.py             P-wave detection and picking metrics
  fusion.py                 Coordinate embedding and cross-station attention
  losses.py                 Seismic objectives and auxiliary distance objective
  model.py                  Single-station and ISFF-augmented wrappers
  statistics.py             Paired Wilcoxon and bootstrap analysis
tests/                      Tests for the reference implementation
```

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

PhaseNet and EQTransformer use SeisBench 0.2.3 as an external dependency. SeisBench is not redistributed in this repository and remains subject to its own GPL-3.0 license.

```bash
pip install seisbench==0.2.3 obspy
```


## Synthetic quickstart

A self-contained synthetic example demonstrates the minimal `BackboneAdapter` interface and the complete ISFF forward path without requiring SeisBench or either manuscript dataset:

```bash
python examples/quickstart.py
```

The example constructs a minimal synthetic encoder/decoder backbone, generates synthetic multi-station waveforms and station coordinates, runs both the station-wise baseline and ISFF-augmented model, and evaluates the auxiliary Haversine distance branch. It is an integration example only and does not reproduce the manuscript results. See `docs/USER_GUIDE.md` for the adapter contract and steps for connecting another picker.

## Model construction

The ViT-based configuration can be instantiated without the optional SeisBench dependency:

```python
import torch
from isff.factory import build_isff_picker

model = build_isff_picker("vit").eval()
waveforms = torch.randn(1, 5, 3, 3000)  # B, N, C, T
lonlat = torch.tensor(
    [[[121.0, 24.0], [121.1, 24.1], [121.2, 24.2], [121.3, 24.3], [121.4, 24.4]]]
)

with torch.no_grad():
    predictions, predicted_distances = model(waveforms, lonlat)
```

After installing SeisBench, use `"phasenet"` or `"eqtransformer"` to construct the other two manuscript backbones.

## Dataset settings

The repository does not redistribute CWASN/CWA Benchmark or INSTANCE waveform data.

### CWASN

- HL strong-motion records and Level-4 earthquake waveforms
- 2012–2018 training, 2019 validation, 2020–2021 testing
- 5,669 earthquake events and 138,476 earthquake waveforms
- 330,005 filtered noise traces
- fixed 235-station representation

### INSTANCE

- EH records
- P-wave label confidence at least 50
- one detected event and one P-wave annotation per waveform
- at least five stations with P-wave observations per event
- 2005–2012 and 2016–2017 training, 2014 and 2019 validation, 2013/2015/2018/2020 testing
- 9,855 earthquake events and 105,075 earthquake waveforms
- fixed 180-station representation

Missing event observations use the same hierarchy in both datasets: same-event/same-station noise when available, same-station noise from another event otherwise, and finally noise from another station while retaining the geographic coordinates of the target station slot.

## Reference-station and waveform-window protocol

During training, one station with an available labeled P-wave arrival is selected as the reference station. Its P-wave arrival determines the common temporal window, and the same shift and crop are applied to every station slot. Non-reference phase annotations are not used for station-specific alignment.

During evaluation, stations with available labeled P-wave arrivals are evaluated in turn as the reference station. Each case retains the complete fixed station tensor as contextual input, while metrics are computed only for the designated reference station. The corresponding single-station baseline evaluates the same reference-station cases independently.

The controlled-placement protocol uses reference-station P-wave positions 750, 1500, 2000, 2750, and 2900 samples. Validation-selected settings are stored in `configs/controlled_cwasn.yaml` and `configs/controlled_instance.yaml`.

## Training settings

`configs/common.yaml` records the common manuscript settings:

- Adam optimizer with initial learning rate `1e-3`
- early stopping after seven validation epochs without improvement
- four ISFF attention heads with dropout `0.1`
- auxiliary station-distance loss weight `2e-7`
- EQTransformer learning-rate milestones at epochs 21, 41, 61, and 91

Dataset- and configuration-specific validation-selected task weights and decision settings are stored in `configs/cwasn.yaml` and `configs/instance.yaml`.

## Evaluation

P-wave evaluation follows the manuscript protocol:

- precision, recall, and F1 use a ±0.5 s tolerance at 100 Hz
- MAE and the standard deviation of the absolute picking error are computed whenever both the prediction and label trigger, including timing errors larger than ±0.5 s
- validation-selected probability and trigger settings are fixed before test evaluation

Example:

```bash
python scripts/evaluate_predictions.py predictions.npz --threshold 0.02 --mode max
```

The NPZ file must contain two arrays named `predictions` and `labels`, each with shape `(num_cases, time)`.

## Statistical analysis

The CWASN random-placement analysis uses a one-sided paired Wilcoxon signed-rank test with the alternative that the single-station absolute picking error is larger than the corresponding ISFF error. A 95% confidence interval for the mean paired reduction is obtained from 10,000 paired bootstrap resamples.

```bash
python scripts/paired_statistics.py single.csv isff.csv
```

Each CSV must contain a unique case identifier (`trace_id` by default) and an absolute-error column (`abs_error` by default). The script pairs only IDs present in both files.

The random-placement results, controlled-placement results, and paired CWASN statistics reported in the manuscript are stored under `results/`.

## Tests

```bash
PYTHONPATH=src pytest -q
```

PhaseNet/EQTransformer adapter tests are skipped when the optional SeisBench/ObsPy environment is not installed.

## Citation

Please cite the accompanying manuscript:

> Yi-Syuan Lin and Kuan-Yu Chen, *A Backbone-Compatible Inter-Station Feature Fusion Framework for Seismic P-Wave Picking*.

## License

Original source code in this repository is licensed under the **Apache License 2.0**. See `LICENSE` and `NOTICE`. Third-party software is not redistributed and remains governed by its own license; see `THIRD_PARTY_NOTICES.md`.
