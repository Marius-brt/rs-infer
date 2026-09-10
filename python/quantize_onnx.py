#!/usr/bin/env uv run
"""Quantize an exported rsinfer ONNX model (CPU INT8) without re-exporting it.

Operates on a model directory as produced by convert_to_onnx.py / optimum
(model.onnx + optional model.onnx.data + tokenizer.json) and writes additional
graph files next to the original so rsinfer can pick them via `file:`:

    model_int8.onnx          dynamic INT8 (MatMul weights int8, runtime scale)
    model_int8_static.onnx   static QDQ INT8 (calibrated activation scales)

Examples:
    uv run quantize_onnx.py --in models/qwen3-embedding-0.6b --mode dynamic
    uv run quantize_onnx.py --in models/qwen3-embedding-0.6b --mode static
    uv run quantize_onnx.py --in models/qwen3-embedding-0.6b --mode static \\
        --calib-file sentences.txt --calib-samples 512
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np

# Generic web-ish sentences for calibration (mixed length/domain); override with --calib-file.
CALIB_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Machine learning models require large amounts of training data to generalize well.",
    "Caffeine affects people differently depending on body weight and tolerance.",
    "The stock market rallied after the central bank signaled a pause in rate hikes.",
    "Photographers often wait hours for the perfect lighting during golden hour.",
    "Rust's ownership system prevents data races at compile time without a garbage collector.",
    "The recipe calls for two cups of flour, a pinch of salt, and three eggs.",
    "Climate change is altering migration patterns of birds across the northern hemisphere.",
    "She booked a flight to Tokyo and planned to visit the Meiji shrine first.",
    "Regular exercise improves sleep quality and reduces symptoms of anxiety.",
    "The new library building features a green roof and solar panels on every wing.",
    "Quantum computers exploit superposition and entanglement to speed up certain algorithms.",
    "His phone kept buzzing with notifications during the entire cinema screening.",
    "The ancient trade route connected cities across the desert for centuries.",
    "A cup of coffee in the morning helps many developers focus on hard problems.",
    "The government announced new regulations for electric vehicle charging stations.",
    "Marine biologists tracked the whales using acoustic sensors deployed along the coast.",
    "Refactoring legacy code is easier when you start by writing characterization tests.",
    "The novel explores themes of memory, loss, and identity in postwar Europe.",
    "High bandwidth memory reduces bottlenecks in large language model inference.",
    "Autumn foliage peaks earlier than usual this year because of the dry summer.",
    "The marathon route winds past the harbor, through the old town, and back.",
    "Astronomers detected a faint radio burst from a distant dwarf galaxy.",
    "Baking bread at home requires patience, a scale, and a hot oven with steam.",
    "Version control systems like git make collaboration on large codebases tractable.",
    "The museum acquired a rare manuscript from the early fifteenth century.",
    "Urban beekeepers place hives on rooftops to support pollinator populations.",
    "Distributed databases use consensus protocols to tolerate machine failures.",
    "The symphony performed an unfamiliar composer's piece to great applause.",
    "Sleep researchers recommend consistent bedtimes over weekend recovery sleep.",
    "The startup raised a seed round to expand its recommendation engine.",
    "Geologists use seismic waves to image structures deep inside the earth.",
    "Proper handwashing remains the cheapest way to prevent infectious disease.",
    "The compiler inlined the hot loop and improved throughput by fifteen percent.",
    "Sourdough starters bubble fastest between twenty four and twenty six degrees.",
    "The documentary followed migratory herds across the serengeti for two years.",
    "Embedding models map sentences to vectors so similar texts land nearby.",
    "The city council debated the budget for public transit extensions last night.",
    "Hydrogen fuel cells emit only water vapor but remain expensive to produce.",
    "A good unit test fails loudly when behavior changes, not when details move.",
    "The lighthouse keeper logged every ship that passed during the long winter.",
    "Text embeddings can be compared with cosine similarity or dot product.",
    "Volcanic soil supports unusually dense agriculture around the island.",
    "The orchestra tuned to a slightly lower pitch than the modern standard.",
    "Compiler warnings about unused variables often reveal dead code paths.",
    "Rain gardens filter stormwater runoff before it reaches the sewer system.",
    "The archive preserves oral histories from the region's fishing communities.",
    "Sparse attention reduces compute for very long documents considerably.",
    "The bakery sold out of rye loaves before noon on market day.",
    "Sensor fusion combines camera, lidar, and radar readings in self-driving cars.",
    "The librarian organized a reading club for children aged eight to ten.",
    "Batching small requests improves GPU utilization during inference workloads.",
    "Glaciologists measure ice thickness with radar pulses from low flights.",
    "The garden needed watering twice a day during the heat wave.",
    "Static analysis tools catch whole classes of bugs before tests even run.",
    "The festival featured street performers from over thirty countries.",
    "Vector databases use approximate nearest neighbor search to scale retrieval.",
    "The orchard yields more fruit after a couple of cooler summers.",
    "Kernel fusions reduce memory traffic in transformer implementations.",
    "The observatory opens its telescope to the public on clear friday nights.",
    "Tokenizers split rare words into subword pieces, inflating sequence length.",
    "The canal locks raise boats nearly sixteen meters between the lakes.",
    "Fine-tuning a small model often beats prompting a huge one on narrow tasks.",
    "The trail markers fade by late season and need repainting every spring.",
    "Memory mapped files let processes share data without copying pages.",
    "The vineyard harvest started two weeks earlier than the decade average.",
    "Attention heads specialize: some track syntax, others copy tokens forward.",
    "The depot rerouted buses after the bridge closed for repairs.",
    "Profile-guided optimization tailors inlining decisions to real workloads.",
    "The wetland filters nitrates before they reach the drinking water supply.",
    "Zero-shot classifiers need only label names, no labeled training examples.",
    "The foundry cast bronze bells using molds carved from green wood.",
    "SIMD instructions multiply throughput of tight numeric loops dramatically.",
    "The school garden supplies vegetables to the cafeteria twice a week.",
    "Rerankers score query-document pairs jointly for higher relevance quality.",
    "The millrace turned the waterwheel that powered the sawmill upstream.",
    "Garbage collectors tune pause times against throughput for latency goals.",
    "The planetarium projected a rare alignment of the five bright planets.",
    "Layer normalization stabilizes training but adds elementwise memory traffic.",
    "The orchid thrived only after the grower switched to bark substrate.",
    "Lock-free queues reduce contention when many threads enqueue work.",
    "The ferry crossed the strait in forty minutes on a calm morning.",
    "Cross-encoder outputs a single relevance score per query-document pair.",
    "The quarry closed after fossils turned up in the third bench cut.",
    "Tries outperform hash maps for prefix completion over huge dictionaries.",
    "The almanac predicted sunrise at ten past six, give or take a minute.",
    "Quantization trades a little accuracy for much smaller, faster models.",
    "The homestead's smokehouse hung hickory-cured bacon through the winter.",
    "Bloom filters answer membership queries in a fraction of a kilobyte.",
    "The tidal bore reversed the river for nearly an hour each spring.",
    "Weight tying between input and output embeddings saves half the matrix.",
    "The aviary keeps the humidity high so the tropical ferns stay lush.",
    "Escape rooms reward teams that communicate observations out loud quickly.",
    "The peat bog preserved a wooden trackway over five thousand years.",
    "Speculative decoding verifies draft tokens in parallel to go faster.",
    "The canal company still pays an annual dividend from toll income.",
    "Linter rules about unused imports keep diffs clean across refactors.",
    "The truffle hunt ended near the old oak at the north pasture edge.",
    "Connection pooling amortizes the handshake cost of database queries.",
    "The frost damaged the olive grove on the southern slope that winter.",
    "Gradient checkpointing trades recompute for lower peak memory.",
    "The lock keeper still winds the clock tower mechanism by hand.",
    "Column stores compress repeated values with run-length encoding.",
    "The oast house dried hops for the village brewery until 1936.",
    "Approximate search indexes like HNSW shrink retrieval latency scales.",
    "The windmill's sails were rebuilt using the original elm framework.",
    "Consistent hashing keeps cache rebalancing cheap when nodes join.",
    "The tithe barn now hosts the weekly farmers market in summer.",
    "Rotary embeddings let transformers generalize beyond fixed positions.",
    "The dovecot once signaled the estate's right to keep pigeons.",
    "Prefetching vectors while scoring the previous page hides disk latency.",
    "The lychgate marked the start of the funeral path to the church.",
    "Distillation compresses an ensemble into a single smaller student.",
    "The water meadows flood gently each February, feeding the silt.",
    "Token budget routers mix cheap and strong models per query difficulty.",
    "The causeway floods at king tides, cutting the island off briefly.",
    "Early exiting skips later layers for tokens that are already certain.",
    "The quay still bears crane marks from the nineteenth-century wool trade.",
    "Speculative sampling preserves the target model's output distribution.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="dir", required=True, help="model directory containing model.onnx")
    p.add_argument("--mode", choices=["dynamic", "static", "both"], default="dynamic")
    p.add_argument("--out-name", default=None, help="output graph file name (default model_int8[_static].onnx)")
    p.add_argument("--calib-file", default=None, help="optional calibration text, one sentence per line")
    p.add_argument("--calib-samples", type=int, default=256, help="number of calibration rows (default 256)")
    p.add_argument("--max-len", type=int, default=512, help="calibration truncation length")
    p.add_argument("--per-channel", action="store_true", help="dynamic: per-channel weight scales (often slower on ARM)")
    p.add_argument("--exclude", default="", help="comma-separated substrings; nodes whose name matches stay fp32 (e.g. '/attn/MatMul')")
    return p.parse_args()


def load_model(dirn: Path):
    import onnx

    model_file = dirn / "model.onnx"
    if not model_file.exists():
        sys.exit(f"error: {model_file} not found")
    return model_file


def calibration_reader(model_file: Path, tok_dir: Path, samples: int, max_len: int, calib_file: str | None):
    """Builds an ORT CalibrationDataReader feeding real tokenized rows matching the graph inputs."""
    from transformers import AutoTokenizer

    class Reader:
        def __init__(self):
            self.tok = AutoTokenizer.from_pretrained(str(tok_dir))
            texts = open(calib_file, encoding="utf-8").read().splitlines() if calib_file else list(CALIB_SENTENCES)
            texts = [t.strip() for t in texts if t.strip()]
            if len(texts) < samples:
                rng = random.Random(7)
                texts = [texts[rng.randrange(len(texts))] + " " + texts[rng.randrange(len(texts))] for _ in range(samples - len(texts))] + texts
            rng = random.Random(11)
            texts = rng.sample(texts, min(samples, len(texts)))
            self.rows = []
            for t in texts:
                enc = self.tok(t, truncation=True, max_length=max_len, padding=False, return_tensors="np")
                ids = enc["input_ids"][0]
                if ids.size < 2:
                    continue
                # batch=1, no padding: position ids are simply 0..n-1 (cumsum(mask)-1 convention).
                pos = np.arange(ids.size, dtype=np.int64)
                self.rows.append({
                    "input_ids": ids.reshape(1, -1).astype(np.int64),
                    "attention_mask": np.ones((1, ids.size), dtype=np.int64),
                    "position_ids": pos.reshape(1, -1).astype(np.int64),
                })
            self.i = 0

        def get_next(self):
            if self.i >= len(self.rows):
                return None
            row = self.rows[self.i]
            self.i += 1
            return row

        def rewind(self):
            self.i = 0

    return Reader()


def node_names_to_exclude(model_file: Path, patterns: list[str]):
    """Resolves substring patterns to concrete MatMul node names (attention projections etc.)."""
    import onnx

    model = onnx.load(str(model_file), load_external_data=False)
    out = set()
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm") and any(pat in node.name for pat in patterns):
            out.add(node.name)
    return sorted(out)


def quantize_dynamic(model_file: Path, out_file: Path, per_channel: bool, exclude: list[str]) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"dynamic INT8 -> {out_file} (per_channel={per_channel}, excluded nodes: {len(exclude)})")
    quantize_dynamic(
        str(model_file),
        str(out_file),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        nodes_to_exclude=exclude or None,
    )


def quantize_static(model_file: Path, out_file: Path, tok_dir: Path, samples: int, max_len: int, calib_file: str | None, exclude: list[str]) -> None:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    reader = calibration_reader(model_file, tok_dir, samples, max_len, calib_file)
    print(f"static QDQ INT8 -> {out_file} ({len(reader.rows)} calibration rows)")
    quantize_static(
        str(model_file),
        str(out_file),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=True,
        nodes_to_exclude=exclude or None,
    )


def main() -> None:
    args = parse_args()
    dirn = Path(args.dir)
    model_file = load_model(dirn)

    import onnx

    graph = onnx.load(str(model_file), load_external_data=False)
    n_nodes = len(graph.graph.node)
    graph_size = sum(f.stat().st_size for f in dirn.glob("model.onnx*"))
    print(f"model: {model_file} nodes={n_nodes} size={graph_size / 2**20:.0f} MiB")

    if graph_size >= 2**31:
        print("note: graph uses external data; outputs will also use external data")

    patterns = [p.strip() for p in args.exclude.split(",") if p.strip()]
    exclude = node_names_to_exclude(model_file, patterns) if patterns else []

    results = []
    if args.mode in ("dynamic", "both"):
        out = dirn / (args.out_name or "model_int8.onnx")
        quantize_dynamic(model_file, out, args.per_channel, exclude)
        results.append(out)
    if args.mode in ("static", "both"):
        out = dirn / (args.out_name or "model_int8_static.onnx")
        quantize_static(model_file, out, dirn, args.calib_samples, args.max_len, args.calib_file, exclude)
        results.append(out)

    for r in results:
        print(f"\nwrote {r} ({r.stat().st_size / 2**20:.1f} MiB + external data)")
    print("\npoint rsinfer at it, e.g.:")
    print(f"  - name: {dirn.name}-int8")
    print("    kind: embedding")
    print(f"    path: {dirn.resolve()}")
    print(f"    file: {results[-1].name}")


if __name__ == "__main__":
    main()
