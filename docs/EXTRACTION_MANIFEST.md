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
- benchmark defaults: 50 steps, window `[s16, s45]`, 2x2 cells,
  protection ratio 0.35, prune-only packing, and soft restoration alpha 0.95
- exact 1K COCO schedule: fixed VAE-space temporal-delta (`lagdx`)
  isolation mask during `[s16, s25)`, followed by per-step current-state
  isolation masks during `[s25, s45)`; reconstruction owners always come from
  the current LLM input (`owner=xt`)

Extract only:

1. temporal VAE-delta protection-signal capture;
2. early-stage protected-cell commit;
3. six-way reconstruction-minimizing 4-to-2 split;
4. sequence packing and unpacking;
5. source-position soft restoration;
6. active-window control.

Do not extract the `v1`-`v15`/`v17`, `C*`, `D3`, oracle, random-partition,
velocity-correction, or debug branches.

Provenance note: saved 1K COCO metadata confirms that the reported run used
`lagdx`, not hidden-state `delta h`. It also confirms that the committed mask
was used for steps 16-24 and current-state isolation was recomputed for steps
25-44. The public `benchmark_compatible_d63` profile preserves those mechanics
instead of relabeling them.

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
