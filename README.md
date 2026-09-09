# ort-infer

A Rust inference server on top of [ONNX Runtime](https://onnxruntime.ai) via the
[ort](https://ort.pyke.io) crate. One binary serves embeddings, rerankers, PII
detection, and zero-shot classification. Models are declared in a YAML config and
fetched from the Hugging Face Hub at startup (or read from a local directory) — no
manual model prep.

## Endpoints

| Model type | Endpoints |
|---|---|
| Embedding | `POST /v1/embeddings` · `POST /embed` |
| Reranker / cross-encoder | `POST /v1/rerank` · `POST /rerank` · `POST /v1/score` |
| PII / NER | `POST /pii/detect` · `POST /pii/redact` |
| Zero-shot classifier | `POST /classify/zero-shot` |
| True/false (NLI entailment) | `POST /classify/true-false` |
| Ops | `GET /health` · `GET /v1/models` · `GET /metrics` |

## Quick start

Build for your platform, then run:

```bash
make <profile>                                   # see table below
cp configs/config.example.yaml config.yaml       # pick your models
./target/release/ortinfer-server --config config.yaml
./scripts/smoke.sh                               # requires jq
```

| Profile | `make` command | Notes |
|---------|----------------|-------|
| CPU (works everywhere) | `make cpu` | no extra features |
| macOS + CoreML/ANE | `make mac-coreml` | |
| Linux + CUDA | `make gpu-cuda` | needs CUDA ≥ 13.2 & cuDNN 9 on `PATH` |
| Linux + TensorRT | `make gpu-trt` | datacenter GPUs |
| Linux + TensorRT-RTX | `make gpu-rtx` | consumer GeForce/RTX |

First boot downloads the configured models into the HF cache (`HF_HOME` respected).

## Documentation

Full documentation lives at
[**https://<your-user>.github.io/ort-infer/**](https://<your-user>.github.io/ort-infer/)
(published via GitHub Pages from the [`docs-site/`](docs-site/) Fumadocs app).

- [Overview](docs-site/content/docs/index.mdx)
- [Usage](docs-site/content/docs/usage.mdx) — endpoints, request examples, adding a model
- [Configuration](docs-site/content/docs/configuration.mdx) — tables of every config and CLI option, logging levels
- API reference (under `/docs/api` on the site) — generated from [`docs-site/openapi.yaml`](docs-site/openapi.yaml) via [Fumadocs OpenAPI](https://www.fumadocs.dev/docs/integrations/openapi); edit the YAML to update it

To work on the docs locally:

```bash
cd docs-site && npm install && npm run dev   # http://localhost:3000
```

## Configuration

Models live in a YAML config (default `config.yaml`). Each entry has a `name`,
a `kind` (`embedding` | `rerank` | `zeroshot` | `pii`), and a source — `hf`
(repo id) or `path` (local dir). `eps: [...]` sets the execution-provider
priority; CPU is always the fallback. See the Configuration page on the docs
site for all options.

```yaml
models:
  - name: e5-small
    kind: embedding
    hf: Xenova/multilingual-e5-small
    max_len: 512
    replicas: 2
    eps: [coreml, cpu]
```

**CoreML caveat:** for small encoder models CoreML is often *slower* than CPU
(ORT splits the graph into many CPU↔CoreML partitions, ~3x slower for
e5-small). Default to `eps: [cpu]` and benchmark per model.

## Development

```bash
make test        # cargo test --workspace (tests live in crates/core/src/tests/)
make lint        # clippy
RUST_LOG=debug   # verbose EP/session/file logs; INFO shows startup + request summaries
```

Architecture: `crates/core` (engine: `ep`, `hub`, `tokenize`, `pool`, `registry`,
`pipeline/*`) and `crates/server` (the HTTP binary). Python helpers for model
export and load-testing live in [`python/`](python/README.md).

## Known limitations

- Encoder-only zero-shot only (no seq2seq BART-MNLI exports).
- No dynamic continuous batching across requests; array inputs batch within one request.
- No auth/TLS — put it behind a reverse proxy.