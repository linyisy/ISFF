# ISFF user guide

This guide explains the minimum interface needed to attach ISFF to another seismic picker. The executable synthetic example is `examples/quickstart.py`.

## 1. Implement the backbone adapter

A compatible backbone subclasses `isff.backbones.base.BackboneAdapter` and provides three interface constants:

- `feature_dim`: channel dimension `d` at the ISFF insertion point
- `temporal_length`: latent temporal length `L`
- `input_length`: waveform length expected by the backbone

The adapter implements two methods:

```python
class MyBackbone(BackboneAdapter):
    feature_dim = ...
    temporal_length = ...
    input_length = ...

    def encode(self, waveforms):
        # waveforms: (B, C, T)
        # representation must be (B, d, L)
        return representation, state

    def decode(self, representation, state):
        # representation: (B, d, L), after optional ISFF residual fusion
        return backbone_predictions
```

`state` may contain skip connections or any other intermediate information required by the original downstream prediction path.

## 2. Wrap the backbone with ISFF

```python
from isff.model import ISFFAugmentedPicker

model = ISFFAugmentedPicker(
    backbone,
    num_heads=4,
    attention_dropout=0.1,
)
```

Multi-station inputs have shape `(B, N, C, T)`. Coordinates have shape `(B, N, 2)` and are ordered as `[longitude, latitude]` in degrees.

```python
predictions, predicted_distances = model(waveforms, lonlat)
```

The wrapper applies the shared backbone station-wise, performs cross-station fusion on the selected `(d, L)` representation, adds the projected ISFF update through the residual path, and then calls the original decoder.

## 3. Auxiliary distance target

The manuscript uses unique station pairs and Haversine distances in kilometers:

```python
from isff.distance import HaversineDistanceTargets, masked_distance_l1_loss

target_distances = HaversineDistanceTargets()(lonlat)
distance_loss = masked_distance_l1_loss(
    predicted_distances,
    target_distances,
)
```

The manuscript training objective uses distance weight `2e-7`.

## 4. Run the synthetic example

After installing the package:

```bash
python examples/quickstart.py
```

The example uses random synthetic waveforms and four example station coordinates. It verifies tensor shapes and the distance branch, but it is not intended to reproduce the manuscript results.

## 5. Manuscript backbones

The repository already provides adapters/factories for the three backbones evaluated in the manuscript:

- PhaseNet: `(d, L) = (32, 12)`, input length 3001
- EQTransformer: `(d, L) = (16, 24)`, input length 3000
- ViT: `(d, L) = (40, 75)`, input length 3000

PhaseNet and EQTransformer require the optional SeisBench/ObsPy environment described in the main README. The synthetic quickstart has no such optional dependency.
