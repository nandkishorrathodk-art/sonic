# Packing and shipping an AVCS model bundle

A trained checkpoint (`.pt`) on its own is not shippable evidence. Nothing
records what data it was measured on, whether it still loads, or whether the
bytes an operator received are the bytes that were evaluated. `avcs.model_bundle`
closes that gap.

## What a bundle contains

```
<cortex>-bundle/
  <cortex>.pt          # the checkpoint, strict-load verified at pack time
  <cortex>.eval.json   # held-out evaluation with named dataset_provenance
  manifest.json        # schema_version, provenance, evaluation, per-file SHA-256
```

`manifest.json` records a SHA-256 and size for every bundled file, the resolved
evaluation sidecar, the runtime (Python/torch/avcs versions), and
`strict_load_verified: true`.

## Pack

```bash
python -m avcs.model_bundle pack \
  --checkpoint sonic_data/avcs/cortex_real.pt \
  --output-dir sonic_data/model_bundles \
  --archive
```

The packer refuses to bundle a checkpoint that:

- does not exist,
- does not strict-load into `ArtificialVisualCortex`,
- has no `.eval.json` sidecar, or
- has a sidecar with no named `dataset_provenance`.

A model whose numbers are not tied to named data is not evidence, so it is not
shippable.

## Verify

```bash
python -m avcs.model_bundle verify --bundle sonic_data/model_bundles/cortex_real-bundle
```

Exit code is `0` when every checksum matches and non-zero with the specific
problem (checksum mismatch, missing file, absent provenance) otherwise. Run this
on the receiving side before trusting the model.

## Load

```python
from avcs.model_bundle import load_bundle

model, manifest = load_bundle("sonic_data/model_bundles/cortex_real-bundle")
print(manifest["dataset_provenance"])
```

`load_bundle` verifies the bundle first and raises `ValueError` rather than
loading weights from a bundle whose bytes do not match its manifest.

## Tests

`tests/test_model_bundle.py` covers the honesty gate: packing produces checksums
and provenance; packing is refused without a sidecar, without provenance, and
for an unloadable checkpoint; verification detects a tampered checkpoint and a
missing file; loading is refused for a tampered bundle.
