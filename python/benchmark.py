#!/usr/bin/env uv run
"""Load-test an ortinfer server: latency percentiles, RPS, token throughput.

Examples:
    uv run benchmark.py --url http://127.0.0.1:8080 --endpoint embeddings \
        --batch 16 --concurrency 8 --total 200
    uv run benchmark.py --endpoint rerank --model ms-marco --concurrency 4 --total 100
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time

import httpx

WORDS = (
    "the quick brown fox jumps over lazy dogs while machine learning models predict "
    "distributed systems scale horizontally through careful engineering and testing"
).split()


def make_text(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


def build_request(args: argparse.Namespace, rng: random.Random) -> tuple[str, dict]:
    """Returns (path, json payload) for one request."""
    texts = [make_text(rng, args.len) for _ in range(args.batch)]
    e = args.endpoint
    if e == "embeddings":
        return "/v1/embeddings", {"model": args.model, "input": texts}
    if e == "embed-tei":
        return "/embed", {"inputs": texts}
    if e == "rerank":
        return "/v1/rerank", {
            "model": args.model,
            "query": make_text(rng, 12),
            "documents": texts,
            "top_n": args.batch,
        }
    if e == "score":
        return "/v1/score", {"model": args.model, "text_1": texts[0], "text_2": texts}
    if e == "pii-detect":
        body = texts[0] if args.batch == 1 else texts
        key = "text" if args.batch == 1 else "texts"
        return "/pii/detect", {"model": args.model, key: body}
    if e == "pii-redact":
        body = texts[0] if args.batch == 1 else texts
        key = "text" if args.batch == 1 else "texts"
        return "/pii/redact", {"model": args.model, key: body, "mode": "mask"}
    if e == "classify":
        return "/classify/zero-shot", {
            "model": args.model,
            "input": texts,
            "candidate_labels": ["politics", "sports", "science", "cooking", "weather"],
        }
    if e == "true-false":
        return "/classify/true-false", {
            "model": args.model,
            "input": texts,
            "question": "Is this text about science?",
        }
    sys.exit(f"unknown endpoint {e}")


def tokens_used(payload: dict | list) -> int:
    if isinstance(payload, dict):
        u = payload.get("usage") or {}
        return int(u.get("total_tokens", 0))
    return 0


async def worker(client: httpx.AsyncClient, path: str, body: dict, lat: list, errors: list) -> int:
    t0 = time.perf_counter()
    try:
        r = await client.post(path, json=body)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            errors.append(f"{r.status_code}: {r.text[:120]}")
            return 0
        lat.append(dt)
        return tokens_used(r.json())
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))
        return 0


async def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency * 2)
    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout, limits=limits) as client:
        try:
            models = (await client.get("/v1/models")).json()["data"]
            kinds = {m["id"]: m["kind"] for m in models}
            if args.model and args.model not in kinds:
                sys.exit(f"model '{args.model}' not on server; available: {sorted(kinds)}")
            if not args.model and models:
                args.model = models[0]["id"]
            print(f"server models: {kinds}\nbenchmarking {args.endpoint} with model={args.model}")
        except (httpx.HTTPError, KeyError) as exc:
            sys.exit(f"cannot reach server at {args.url}: {exc}")

        # warmup
        for _ in range(args.warmup):
            path, body = build_request(args, rng)
            await worker(client, path, body, [], [])

        lat: list[float] = []
        errors: list[str] = []
        total_tokens = 0
        queue: asyncio.Queue = asyncio.Queue()
        for _ in range(args.total):
            path, body = build_request(args, rng)
            queue.put_nowait((path, body))

        async def loop() -> None:
            nonlocal total_tokens
            while True:
                try:
                    path, body = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                total_tokens += await worker(client, path, body, lat, errors)

        sem = asyncio.Semaphore(args.concurrency)

        async def guarded() -> None:
            async with sem:
                await loop()

        t0 = time.perf_counter()
        await asyncio.gather(*(guarded() for _ in range(args.concurrency)))
        wall = time.perf_counter() - t0

    if not lat:
        print("\n".join(errors[:5]))
        sys.exit("all requests failed")
    lat_s = sorted(lat)
    pct = lambda p: lat_s[min(len(lat_s) - 1, int(p / 100 * len(lat_s)))] * 1000  # noqa: E731
    reqs_ok = len(lat)
    docs = reqs_ok * args.batch
    print(f"""
  endpoint      {args.endpoint}  (batch {args.batch}, {args.len} words/doc)
  requests      {reqs_ok} ok / {args.total} sent ({len(errors)} errors)
  wall time     {wall:.2f}s
  latency p50   {pct(50):8.1f} ms
  latency p90   {pct(90):8.1f} ms
  latency p99   {pct(99):8.1f} ms
  throughput    {reqs_ok / wall:8.1f} req/s   {docs / wall:9.1f} docs/s   {total_tokens / wall:9.0f} tok/s""")
    if args.json:
        Path_json = args.json
        with open(Path_json, "w") as fh:
            json.dump(
                {"endpoint": args.endpoint, "batch": args.batch, "concurrency": args.concurrency,
                 "requests_ok": reqs_ok, "errors": len(errors), "wall_s": wall,
                 "p50_ms": pct(50), "p90_ms": pct(90), "p99_ms": pct(99),
                 "docs_per_s": docs / wall, "tok_per_s": total_tokens / wall},
                fh, indent=2)
        print(f"  wrote {Path_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--endpoint", default="embeddings",
                   choices=["embeddings", "embed-tei", "rerank", "score", "pii-detect", "pii-redact", "classify", "true-false"])
    p.add_argument("--model", default=None, help="model name (default: first on server)")
    p.add_argument("--batch", type=int, default=8, help="inputs (docs/texts) per request")
    p.add_argument("--len", type=int, default=40, help="words per generated text")
    p.add_argument("--concurrency", type=int, default=4, help="parallel in-flight requests")
    p.add_argument("--total", type=int, default=100, help="total requests")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", help="write a machine-readable summary to this path")
    return p.parse_args()


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
