# SaToMe

Official implementation of **SaToMe: Salience-Driven Token Merging
for Omni-Capable Diffusion MLLMs** (EMNLP Findings).

SaToMe is a training-free visual-token merging method for diffusion multimodal
language models. It shares a local salience primitive across two task-specific
paths:

- **T2I:** trajectory-windowed, cell-local merging and soft restoration.
- **I2T:** salience-driven additive merging after an early transformer layer.

## Release Status

This is an initial, usable release focused on the LLaDA-o T2I path. The public
core has been extracted from the experiment implementation, stripped of
ablation branches, and checked against the reference experiment invariants.

| Path | Backbone | Status |
|---|---|---|
| T2I | LLaDA-o | **MVP available; GPU smoke pending** |
| I2T | LLaDA-o | planned |
| I2T | LaViDa | planned |

The MVP provides the default T2I inference path and a matched-seed smoke demo.
Full benchmark launchers and the two I2T integrations will follow.

## Quick Start: LLaDA-o T2I

Start from an environment that can run the official LLaDA-o demo, then install
SaToMe and apply the small integration patch:

```bash
git clone --recursive https://github.com/sixun-liu/SaToMe.git
cd SaToMe
python -m pip install -e .
./scripts/apply_llada_o_patch.sh
```

Run a matched-seed baseline/SaToMe smoke comparison:

```bash
python scripts/llada_o_t2i_demo.py \
  --model-path /path/to/GSAI-ML-LLaDA-o \
  --mode both \
  --output-dir outputs/t2i_smoke
```

The command writes `baseline.png`, `satome.png`, and `metrics.json`. At the
default 1024x1024 setting, SaToMe retains 2,764/4,096 visual tokens during 29
active model steps and protects 358/1,024 cells.

See [the integration guide](integrations/llada_o/README.md) for implementation
boundaries and [the extraction manifest](docs/EXTRACTION_MANIFEST.md) for the
released-code scope.

## Repository Layout

```text
src/satome/          Backbone-independent method code
configs/             Paper configuration contracts
integrations/        Thin hooks for the two upstream backbones
scripts/             Inference and evaluation entry points
tests/               CPU unit tests for method invariants
docs/                Release and reproducibility notes
```

## Development Install

```bash
python -m pip install -e '.[dev]'
pytest
```

Model checkpoints and benchmark datasets are not distributed in this
repository.

## MVP Configuration

The released profile uses a 2x2 grid, a 35% protected-cell ratio, prune-only
sequence packing, current-input reconstruction owners, copied owner velocity
at dropped positions, and soft restoration with `alpha=0.95`. Merging is active
on steps `[16, 45)`. See [configs/t2i/llada_o.yaml](configs/t2i/llada_o.yaml)
for the two-stage protection schedule.

## Upstream Backbones

The release is developed against pinned versions of
[LaViDa](https://github.com/jacklishufan/LaViDa) and
[LLaDA-o](https://github.com/ML-GSAI/LLaDA-o). See [UPSTREAMS.md](UPSTREAMS.md)
for exact revisions and licensing details.

## License

SaToMe is released under the Apache License 2.0. The upstream backbones are
also Apache-2.0 licensed; their notices and modification boundaries are kept
explicit in this repository.

## Citation

The proceedings BibTeX entry will be added when the final EMNLP Findings
metadata is available.
