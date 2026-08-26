# Extraction Manifest

This document tracks the one-time migration from the research worktrees into
the public implementation. Historical experiment variants are intentionally
excluded.

## T2I / LLaDA-o

Research source:

- branch: `exp/lladao-t2i-route-a-poc`
- primary backbone files:
  - `references/LLaDA-o/modeling/lladao/lladao.py`
  - `references/LLaDA-o/modeling/lladao/llada_navit.py`
- final mechanism family: `v16` / D63
- accepted-paper defaults: 50 steps, window `[s16, s45]`, 2x2 cells,
  protection ratio 0.35, soft restoration alpha 0.95

Extract only:

1. temporal protection-signal capture;
2. first-window-step protected-cell commit;
3. six-way reconstruction-minimizing 4-to-2 split;
4. sequence packing and unpacking;
5. source-position soft restoration;
6. active-window control.

Do not extract the `v1`-`v15`/`v17`, `C*`, `D3`, oracle, random-partition,
velocity-correction, or debug branches.

Open parity check: the historical D63 benchmark scripts use the internal
`lagdx` signal name, while the accepted-paper method is described as temporal
hidden delta (`delta h`). Resolve this against saved run metadata before
declaring the T2I integration benchmark-compatible.

The historical salience helper returns `-cos(h, local_mean)` while the paper
and public core return `1-cos(h, local_mean)`. These scores differ by the
constant `+1`, so all rankings and TopK masks are exactly unchanged.

## I2T / LLaDA-o

Research source:

- `references/LLaDA-o/modeling/lladao/route_a.py`
- integration hooks in `lladao.py`, `demo_pipeline.py`, and `inferencer.py`
- evaluation runner: `scripts/exp/llada_o/run_lladao_i2t_suite.py`

Extract the single released salience/additive-merge path only. Remove no-cache
diagnostics, layer sweeps, and experimental layout modes after reproducing the
paper operating points.

## I2T / LaViDa

Research source:

- branch: `exp/mmada-migration` at `3cbeb1f`
- primary files:
  - `llava/model/language_model/llada/generate.py`
  - `llava/model/language_model/llada/modeling_llada.py`
  - `llava/model/llava_arch.py`
  - thin argument plumbing in the LaViDa model wrappers

Extract the layer-1 salience/additive-merge path and the visual-layout metadata
needed for it. Exclude MMaDA experiments, progressive schedules, cache probes,
and D3ToM comparison code.

Open parity check: reconcile the paper's token-level selection notation with
the spatial-bin implementation used by the benchmark launchers, then expose
one unambiguous public configuration.
