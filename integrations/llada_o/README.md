# LLaDA-o Integration

Target upstream revision: `00b40b4123c730eaf765bba439f63dd1d453d152`.

## T2I MVP

The T2I integration is a 47-line upstream hook. Planning, packing, unmerging,
and runtime state live in `src/satome/integrations/llada_o_t2i.py`.

```bash
python -m pip install -e .
./scripts/apply_llada_o_patch.sh
python scripts/llada_o_t2i_demo.py \
  --model-path /path/to/GSAI-ML-LLaDA-o \
  --mode both
```

The `both` mode loads the model once and emits matched-seed `baseline.png`,
`satome.png`, and `metrics.json`. The benchmark-compatible profile should
report 29 active steps, 358 protected cells, and 2,764/4,096 retained visual
tokens for a 1024x1024 image. LLaDA-o constructs 50 timestep points but omits
the terminal point from model evaluation, so the half-open `[16, 45)` window
contains 29 forward passes.

The extraction passes CPU invariants and exact D63 planning-equation parity.
An end-to-end checkpoint smoke is still pending on a CUDA-enabled host.

I2T extraction is not included in the MVP yet.
