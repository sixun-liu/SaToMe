# SaToMe

Official implementation scaffold for **SaToMe: Salience-Driven Token Merging
for Omni-Capable Diffusion MLLMs** (EMNLP Findings).

SaToMe is a training-free visual-token merging method for diffusion multimodal
language models. It shares a local salience primitive across two task-specific
paths:

- **T2I:** trajectory-windowed, cell-local merging and soft restoration.
- **I2T:** salience-driven additive merging after an early transformer layer.

## Release Status

This repository is being rebuilt from the exact experiment implementations.
The backbone-independent salience and T2I cell-planning primitives are present
and tested. Backbone integrations are tracked below and will only be marked
runnable after parity checks against the accepted-paper results.

| Path | Backbone | Status |
|---|---|---|
| T2I | LLaDA-o | extraction in progress |
| I2T | LLaDA-o | extraction in progress |
| I2T | LaViDa | extraction in progress |

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

## Upstream Backbones

The release is developed against pinned versions of
[LaViDa](https://github.com/jacklishufan/LaViDa) and
[LLaDA-o](https://github.com/ML-GSAI/LLaDA-o). See [UPSTREAMS.md](UPSTREAMS.md)
for exact revisions and licensing details.

## License

SaToMe is released under the Apache License 2.0. The upstream backbones are
also Apache-2.0 licensed; their notices and modification boundaries are kept
explicit in this repository.
