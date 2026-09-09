#!/usr/bin/env uv run
"""Convert a Hugging Face model to ONNX for use with ortinfer.

Wraps optimum's exporters (dynamic batch/sequence axes, external data for
large models) and optionally applies dynamic INT8 quantization.

Examples:
    uv run convert_to_onnx.py --repo sentence-transformers/paraphrase-MiniLM-L3-v2 \
        --task feature-extraction --out models/minilm-l3
    uv run convert_to_onnx.py --repo cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --task text-classification --out models/ms-marco --int8
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# ortinfer `kind` -> optimum task
TASK_BY_KIND = {
    "embedding": "feature-extraction",
    "rerank": "text-classification",
    "zeroshot": "text-classification",
    "pii": "token-classification",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="Hugging Face model id")
    p.add_argument("--task", help="optimum task (feature-extraction, text-classification, token-classification, ...)")
    p.add_argument("--kind", choices=sorted(TASK_BY_KIND), help="ortinfer model kind; maps to a task automatically")
    p.add_argument("--out", required=True, help="output directory (usable directly as ortinfer `path`)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (default 17)")
    p.add_argument("--fp16", action="store_true", help="export in half precision (CUDA host only)")
    p.add_argument("--int8", action="store_true", help="post-export dynamic INT8 quantization (CPU-optimized)")
    p.add_argument("--overwrite", action="store_true", help="remove the output dir first")
    p.add_argument("--subfolder", default="", help="HF repo subfolder")
    return p.parse_args()


def convert(args: argparse.Namespace) -> Path:
    from optimum.exporters.onnx import main_export

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            sys.exit(f"error: {out} exists (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    task = args.task or TASK_BY_KIND.get(args.kind or "")
    if not task:
        sys.exit("error: provide --task or --kind")

    print(f"exporting {args.repo} (task={task}, opset={args.opset}, fp16={args.fp16}) -> {out}")
    main_export(
        model_name_or_path=args.repo,
        output=str(out),
        task=task,
        opset=args.opset,
        fp16=args.fp16,
        no_patch_model=args.fp16,  # patching can't serialize fp16 weights
        subfolder=args.subfolder,
    )

    # optimum writes into an `onnx/` subdir for some tasks; flatten for ortinfer simplicity
    nested = out / "onnx"
    graph = out / "model.onnx"
    if not graph.exists() and (nested / "model.onnx").exists():
        for f in nested.iterdir():
            shutil.move(str(f), str(out / f.name))
        nested.rmdir()

    if args.int8:
        quantize(out)

    return out


def quantize(out: Path) -> None:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    src = out / "model.onnx"
    if not src.exists():
        sys.exit("error: --int8 requires onnx/model.onnx")
    print("applying dynamic INT8 quantization ...")
    qout = out / "quantized"
    ORTQuantizer.from_pretrained(out).export(
        quantization_config=AutoQuantizationConfig.create_default_quantization_config("int8_dynamic"),
        save_dir=str(qout),
    )
    for f in qout.iterdir():
        shutil.move(str(f), str(out / f.name))
    shutil.rmtree(qout)


def print_yaml_snippet(args: argparse.Namespace, out: Path) -> None:
    kind = args.kind or {"feature-extraction": "embedding", "token-classification": "pii"}.get(
        args.task or "", "rerank"
    )
    name = Path(args.repo).name.lower().replace("_", "-")
    cfg = out / "config.json"
    if cfg.exists():
        labels = json.loads(cfg.read_text()).get("id2label", {})
        if labels and kind == "rerank":
            values = " ".join(labels.values()).lower()
            if "entail" in values or "contradict" in values:
                kind = "zeroshot"
    print("\n# add to your ortinfer config.yaml:")
    print("models:")
    print(f"  - name: {name}")
    print(f"    kind: {kind}")
    print(f"    path: {out.resolve()}")


def main() -> None:
    args = parse_args()
    out = convert(args)
    print_yaml_snippet(args, out)
    print(f"\ndone: {out}")


if __name__ == "__main__":
    main()
