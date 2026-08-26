#!/usr/bin/env python3
"""Generate a baseline/SaToMe T2I smoke comparison with LLaDA-o."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from satome import T2IConfig
from satome.integrations import enable_llada_o_t2i


REPO_ROOT = Path(__file__).resolve().parents[1]
LLADA_O_ROOT = REPO_ROOT / "third_party" / "llada_o"
sys.path.insert(0, str(LLADA_O_ROOT))

from demo_pipeline import LLaDAMultimodalDemo  # noqa: E402


DEFAULT_PROMPT = "A giant panda wearing a space suit, surfing on a galaxy wave."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--mode", choices=("baseline", "satome", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/t2i_smoke"))
    parser.add_argument("--max-mem-per-gpu", default="80GiB")
    parser.add_argument("--offload-dir", default="/tmp/lladao_offload")
    return parser.parse_args()


def generate(demo: LLaDAMultimodalDemo, args: argparse.Namespace, method: str) -> dict:
    if method == "satome":
        runtime = enable_llada_o_t2i(demo.model, T2IConfig(total_steps=args.steps))
    else:
        demo.model._satome_t2i_runtime = None
        runtime = None

    start = time.perf_counter()
    result = demo.text_to_image(
        args.prompt,
        seed=args.seed,
        cfg_text_scale=args.cfg,
        cfg_img_scale=1.0,
        num_timesteps=args.steps,
    )
    elapsed = time.perf_counter() - start
    output_path = args.output_dir / f"{method}.png"
    result["image"].save(output_path)

    record = {
        "method": method,
        "prompt": args.prompt,
        "seed": args.seed,
        "cfg": args.cfg,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "output": str(output_path),
    }
    if runtime is not None:
        record.update(
            active_merge_steps=len(runtime.history),
            retained_visual_tokens=runtime.history[0].kept_tokens if runtime.history else None,
            total_visual_tokens=runtime.history[0].total_tokens if runtime.history else None,
            protected_cells=runtime.history[0].protected_cells if runtime.history else None,
        )
    print(json.dumps(record, indent=2), flush=True)
    return record


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    demo = LLaDAMultimodalDemo.from_pretrained(
        model_path=args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_dir=args.offload_dir,
    )
    methods = ("baseline", "satome") if args.mode == "both" else (args.mode,)
    records = [generate(demo, args, method) for method in methods]
    (args.output_dir / "metrics.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
