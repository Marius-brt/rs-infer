#!/usr/bin/env uv run
"""Export LiquidAI LFM2.5-Encoder PII detectors (or any LFM2 token-classifier) to ONNX.

optimum's `main_export` cannot be used for these repos:
  * the architecture is `trust_remote_code` (Lfm2BidirP2ForTokenClassification),
  * the remote code PATCHES the backbone to be bidirectional (symmetric gated
    short conv + non-causal attention). Exporting the native transformers LFM2
    model instead would silently produce a CAUSAL, wrongly-weighted graph.

We therefore load the remote class itself and trace it with torch's dynamo
exporter (symbolic shapes are required: the short-conv forward contains a
shape-dependent truncation branch that must resolve symbolically for dynamic
sequence lengths).

The output directory is immediately usable as an ortinfer `path:` model with
`kind: pii` (model.onnx + config.json + tokenizer.json, id2label BIOES).

Example:
    uv run export_lfm2_pii.py --repo LiquidAI/LFM2.5-Encoder-350M-PII-Detector \
        --out ../models/lfm-pii --verify
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


AUX_NOTE = "outputs [batch, seq, num_labels] float32 logits"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="LiquidAI/LFM2.5-Encoder-350M-PII-Detector")
    p.add_argument("--out", required=True, help="output directory for the ortinfer `path:` model")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--seq", type=int, default=64, help="example sequence length for tracing")
    p.add_argument("--no-verify", dest="verify", action="store_false", help="skip the ORT-vs-torch consistency check")
    p.set_defaults(verify=True)
    return p.parse_args()


def load(repo: str):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerFast

    print(f"loading {repo} (trust_remote_code=True) ...")
    try:
        tok = AutoTokenizer.from_pretrained(repo)
    except (ValueError, KeyError):
        # Some repos set tokenizer_class: TokenizersBackend (unsupported by this
        # transformers version); tokenizer.json is self-contained -> load it raw.
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(repo, allow_patterns=["tokenizer.json", "tokenizer_config.json"]))
        tok = PreTrainedTokenizerFast(tokenizer_file=str(src / "tokenizer.json"))
        tcfg = json.loads((src / "tokenizer_config.json").read_text())
        for key in ("bos", "eos", "pad", "mask"):
            t = tcfg.get(f"{key}_token")
            if t:
                setattr(tok, f"{key}_token", t)
        extra = tcfg.get("extra_special_tokens") or []
        if isinstance(extra, dict):
            extra = list(extra.values())
        tok.add_special_tokens({"additional_special_tokens": list(extra)}) if extra else None
        print("loaded tokenizer.json directly (tokenizer_class fallback)")
    model = AutoModelForTokenClassification.from_pretrained(repo, dtype=torch.float32, trust_remote_code=True)
    if model.__class__.__name__.lower().startswith("lfm2"):
        print(f"remote architecture loaded: {model.__class__.__name__}")
    model.eval()
    return tok, model


def export(args: argparse.Namespace) -> Path:
    import torch

    tok, model = load(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # NOTE: example batch size MUST be >= 2: dynamo specializes dims equal to 1
    # into constants, which breaks Expand broadcast nodes at inference time.
    sample = tok(
        [
            "Email Dr. Laura Schmidt at laura@charite.de or call +49 30 4505 1234.",
            "My IBAN is DE89370400440532013000.",
        ],
        return_tensors="pt",
        truncation=True,
        max_length=args.seq,
        padding="longest",
    )
    ids = sample["input_ids"]
    mask = sample["attention_mask"]

    class OnnxGraph(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    graph = OnnxGraph(model)
    target = out / "model.onnx"
    print(f"exporting graph (dynamo, opset {args.opset}) ... {AUX_NOTE}")
    batch = torch.export.Dim("batch", min=1, max=256)
    seq = torch.export.Dim("seq", min=1, max=4096)
    torch.onnx.export(
        graph,
        (ids, mask),
        str(target),
        dynamo=True,
        opset_version=args.opset,
        external_data=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes=[{0: batch, 1: seq}, {0: batch, 1: seq}],
    )
    if not target.exists():
        raise SystemExit("export produced no model.onnx (check the error above)")

    # Auxiliary files ortinfer needs locally: tokenizer + config (id2label).
    from huggingface_hub import snapshot_download

    patterns = ["tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]
    src = Path(snapshot_download(args.repo, allow_patterns=patterns))
    for name in patterns:
        f = src / name
        if f.is_file():
            shutil.copy2(f, out / name)
    cfg = json.loads((out / "config.json").read_text())
    n = len(cfg.get("id2label", {}))
    print(f"copied aux files to {out}; {n} BIOES labels")
    return out


def bioes_spans(text: str, offsets, preds, id2label: dict[int, str]) -> list[str]:
    spans, cur = [], None
    def flush():
        if cur and cur[1] > cur[0]:
            spans.append(text[cur[0]:cur[1]] + f" [{cur[2]}]")
    for (s, e), p in zip(offsets, preds):
        label = id2label.get(int(p), "O")
        if e <= s:
            flush(); cur = None; continue
        prefix, _, typ = label.partition("-")
        if prefix in ("B", "S") or (prefix == "I" and (cur is None or cur[2] != typ)) or prefix == "":
            flush()
            if prefix != "O":
                cur = [s, e, typ]
        if cur:
            cur[1] = max(cur[1], e)
        if prefix in ("E", "S"):
            flush(); cur = None
    flush()
    return spans


def verify(args: argparse.Namespace, out: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch

    tok, model = load(args.repo)
    texts = [
        "Email Dr. Laura Schmidt at laura@charite.de.",
        "My IBAN is DE89370400440532013000 and SSN 123-45-6789.",
    ]
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    for t in texts:
        enc = tok(t, return_tensors="np", truncation=True, max_length=256)
        feed = {k: enc[k] for k in ("input_ids", "attention_mask")}
        onnx_logits = sess.run(None, feed)[0]

        tents = {k: torch.tensor(v) for k, v in feed.items()}
        with torch.no_grad():
            torch_logits = model(**tents).logits.numpy()

        diff = np.abs(onnx_logits - torch_logits).max()
        agree = (onnx_logits.argmax(-1) == torch_logits.argmax(-1)).mean()
        preds = onnx_logits[0].argmax(-1).tolist()
        off = tok(t, return_offsets_mapping=True, truncation=True, max_length=256)["offset_mapping"]
        cfg = json.loads((out / "config.json").read_text())
        id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
        spans = bioes_spans(t, off, preds, id2label)
        print(f"\ntext: {t}\n  max|Δ|={diff:.2e} argmax-agreement={agree:.4f}\n  entities: {spans}")
        if agree < 0.98 or diff > 5e-2:
            raise SystemExit("FAILED: ONNX output diverges from torch — do not use this export")
    print("\nverification passed (labels match torch reference)")


def main() -> None:
    args = parse_args()
    out = export(args)
    if args.verify:
        verify(args, out)
    print(f"""
done. Add to config.yaml:

models:
  - name: lfm-pii
    kind: pii
    path: {out.resolve()}
    max_len: 256
    eps: [cpu]   # CoreML support for this arch varies; benchmark before enabling
""")


if __name__ == "__main__":
    main()
