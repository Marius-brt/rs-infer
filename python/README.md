# ortinfer python utilities

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install pinned deps (torch, transformers, optimum, ...)
```

## convert_to_onnx.py — HF model -> ONNX

Exports any supported HF checkpoint to the layout ortinfer expects
(`model.onnx` + `tokenizer.json` + `config.json`), with dynamic batch/sequence
axes and automatic external-data handling. Optional dynamic INT8 quantization.

```bash
uv run convert_to_onnx.py --repo sentence-transformers/paraphrase-MiniLM-L3-v2 \
  --task feature-extraction --out ../models/minilm-l3

uv run convert_to_onnx.py --repo cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --kind rerank --out ../models/ms-marco --int8
```

`--kind` maps ortinfer kinds to optimum tasks
(`embedding`→feature-extraction, `rerank`/`zeroshot`→text-classification,
`pii`→token-classification) and the script prints a ready-to-paste YAML
snippet for `config.yaml`.

## benchmark.py — HTTP load test

```bash
# start the server, then:
uv run benchmark.py --endpoint embeddings --batch 16 --concurrency 8 --total 300
uv run benchmark.py --endpoint rerank --model ms-marco --concurrency 4 --total 100
uv run benchmark.py --endpoint pii-detect --batch 4 --len 120 --json /tmp/pii.json
```

Reports p50/p90/p99 latency, req/s, docs/s and token throughput
(`usage.total_tokens`), against every endpoint type.
