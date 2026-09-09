import Link from 'next/link';
import {
  Activity,
  ArrowRight,
  Boxes,
  Cpu,
  Download,
  ShieldCheck,
  Sparkles,
  TextSearch,
} from 'lucide-react';
import { appName, assetBase } from '@/lib/shared';

const models = [
  {
    icon: Boxes,
    title: 'Embeddings',
    routes: '/v1/embeddings · /embed',
    text: 'OpenAI/vLLM-compatible vectors with Matryoshka truncation and base64 responses.',
  },
  {
    icon: TextSearch,
    title: 'Rerankers',
    routes: '/v1/rerank · /v1/score',
    text: 'Cross-encoder scoring for (query, document) pairs — Jina/Cohere-compatible API.',
  },
  {
    icon: ShieldCheck,
    title: 'PII detection',
    routes: '/pii/detect · /pii/redact',
    text: 'Zero-shot NER over configurable labels, with span offsets and redacted output.',
  },
  {
    icon: Sparkles,
    title: 'Zero-shot classification',
    routes: '/classify/zero-shot · /classify/true-false',
    text: 'Classify into arbitrary labels, or reduce NLI entailment to a true/false judgement.',
  },
];

const highlights = [
  {
    icon: Download,
    title: 'Models from the Hub',
    text: 'Declare models in YAML — fetched from Hugging Face at startup, or served from a local directory. No manual export step.',
  },
  {
    icon: Cpu,
    title: 'Any execution backend',
    text: 'One binary, build profiles for CPU, CoreML/ANE on macOS, CUDA and TensorRT on Linux. EPs are matched automatically per model.',
  },
  {
    icon: Activity,
    title: 'Operations-friendly',
    text: 'Prometheus metrics on /metrics, /health and /v1/models probes, request timeouts, body limits, and per-model queues.',
  },
];

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-fd-border">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-60"
          style={{
            background:
              'radial-gradient(ellipse 60% 50% at 50% -10%, color-mix(in oklab, var(--color-fd-primary) 18%, transparent), transparent), radial-gradient(ellipse 40% 40% at 85% 100%, color-mix(in oklab, var(--color-fd-primary) 8%, transparent), transparent)',
          }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black,transparent)]"
          style={{
            backgroundImage:
              'linear-gradient(to right, color-mix(in oklab, var(--color-fd-border) 60%, transparent) 1px, transparent 1px), linear-gradient(to bottom, color-mix(in oklab, var(--color-fd-border) 60%, transparent) 1px, transparent 1px)',
            backgroundSize: '40px 40px',
          }}
        />
        <div className="relative mx-auto flex max-w-5xl flex-col items-center px-6 py-20 text-center md:py-28">
          <img
            src={`${assetBase}/logo.png`}
            alt={`${appName} logo`}
            className="h-24 w-auto md:h-28"
          />
          <div className="mt-8 flex flex-wrap items-center justify-center gap-2 text-xs font-medium">
            <span className="rounded-full border border-fd-border bg-fd-card px-3 py-1 text-fd-muted-foreground">
              Rust
            </span>
            <span className="rounded-full border border-fd-border bg-fd-card px-3 py-1 text-fd-muted-foreground">
              ONNX Runtime
            </span>
            <span className="rounded-full border border-fd-border bg-fd-card px-3 py-1 text-fd-muted-foreground">
              OpenAI / vLLM compatible
            </span>
            <span className="rounded-full border border-fd-primary/30 bg-fd-primary/10 px-3 py-1 text-fd-primary">
              one binary
            </span>
          </div>

          <h1 className="mt-6 text-5xl font-bold tracking-tight md:text-7xl">
            <span className="bg-gradient-to-b from-fd-foreground via-fd-foreground to-fd-muted-foreground bg-clip-text text-transparent">
              {appName}
            </span>
          </h1>
          <p className="mt-5 max-w-xl text-balance text-lg leading-relaxed text-fd-muted-foreground md:text-xl">
            Self-hosted ONNX inference: embeddings, rerankers, PII, and
            zero-shot classification — one binary, one API.
          </p>

          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/docs"
              className="inline-flex items-center gap-1.5 rounded-xl bg-fd-primary px-6 py-3 text-sm font-semibold text-fd-primary-foreground shadow-lg shadow-fd-primary/20 transition hover:opacity-90"
            >
              Get started
              <ArrowRight className="size-4" />
            </Link>
            <Link
              href="/docs/api"
              className="inline-flex items-center rounded-xl border border-fd-border bg-fd-card/70 px-6 py-3 text-sm font-semibold backdrop-blur transition hover:bg-fd-accent"
            >
              Server Endpoints
            </Link>
          </div>

          {/* Terminal preview */}
          <div className="mt-14 w-full max-w-2xl overflow-hidden rounded-2xl border border-fd-border bg-fd-card text-left shadow-2xl shadow-black/10 dark:shadow-black/40">
            <div className="flex items-center gap-1.5 border-b border-fd-border px-4 py-3">
              <span className="size-2.5 rounded-full bg-red-500/80" />
              <span className="size-2.5 rounded-full bg-yellow-500/80" />
              <span className="size-2.5 rounded-full bg-green-500/80" />
              <span className="ml-3 font-mono text-xs text-fd-muted-foreground">
                terminal
              </span>
            </div>
            <pre className="overflow-x-auto p-5 font-mono text-[13px] leading-relaxed">
              <code>
                <span className="text-fd-muted-foreground">$ </span>
                <span className="text-fd-foreground">make cpu</span>
                {'\n'}
                <span className="text-fd-muted-foreground">$ </span>
                <span className="text-fd-foreground">
                  ./target/release/rsinfer-server --config config.yaml
                </span>
                {'\n'}
                <span className="text-green-600 dark:text-green-400">
                  ready
                </span>
                <span className="text-fd-muted-foreground">
                  {'  '}
                  models fetched from HF Hub · listening on :8080
                </span>
                {'\n\n'}
                <span className="text-fd-muted-foreground">$ </span>
                <span className="text-fd-foreground">
                  {'curl -s localhost:8080/v1/embeddings -d '}
                </span>
                <span className="text-fd-primary">
                  {'&#39;{"model":"bge-m3","input":"hello world"}&#39;'}
                </span>
                {'\n'}
                <span className="text-fd-muted-foreground">
                  {'{"data":[{"embedding":[0.012,-0.048,…],"index":0}]}'}
                </span>
              </code>
            </pre>
          </div>
        </div>
      </section>

      {/* Model families */}
      <section className="mx-auto w-full max-w-5xl px-6 py-16 md:py-20">
        <h2 className="text-center text-2xl font-bold tracking-tight md:text-3xl">
          Four model families, one API
        </h2>
        <p className="mx-auto mt-3 max-w-xl text-center text-fd-muted-foreground">
          Mix and match models in <code className="rounded bg-fd-secondary px-1.5 py-0.5 font-mono text-sm">config.yaml</code> —
          each type gets its own routes, defaults, and per-model limits.
        </p>
        <div className="mt-10 grid gap-4 sm:grid-cols-2">
          {models.map((m) => (
            <div
              key={m.title}
              className="group rounded-2xl border border-fd-border bg-fd-card p-6 transition hover:border-fd-primary/40 hover:shadow-lg hover:shadow-fd-primary/5"
            >
              <div className="flex size-11 items-center justify-center rounded-xl bg-fd-primary/10 text-fd-primary transition group-hover:bg-fd-primary group-hover:text-fd-primary-foreground">
                <m.icon className="size-5" />
              </div>
              <h3 className="mt-4 font-semibold">{m.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-fd-muted-foreground">
                {m.text}
              </p>
              <p className="mt-4 font-mono text-xs text-fd-muted-foreground/80">
                {m.routes}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Highlights */}
      <section className="border-y border-fd-border bg-fd-secondary/30">
        <div className="mx-auto grid max-w-5xl gap-10 px-6 py-16 md:grid-cols-3 md:py-20">
          {highlights.map((h) => (
            <div key={h.title}>
              <div className="flex size-10 items-center justify-center rounded-lg border border-fd-border bg-fd-card text-fd-primary">
                <h.icon className="size-5" />
              </div>
              <h3 className="mt-4 font-semibold">{h.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-fd-muted-foreground">
                {h.text}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto max-w-5xl px-6 py-16 text-center md:py-24">
        <h2 className="text-2xl font-bold tracking-tight md:text-3xl">
          Running in under five minutes
        </h2>
        <p className="mx-auto mt-3 max-w-md text-fd-muted-foreground">
          Build for your platform, point at a config, and the models take care
          of themselves.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link
            href="/docs"
            className="inline-flex items-center gap-1.5 rounded-xl bg-fd-primary px-6 py-3 text-sm font-semibold text-fd-primary-foreground transition hover:opacity-90"
          >
            Read the guide
            <ArrowRight className="size-4" />
          </Link>
          <Link
            href="/docs/setup/configuration"
            className="inline-flex items-center rounded-xl border border-fd-border bg-fd-card px-6 py-3 text-sm font-semibold transition hover:bg-fd-accent"
          >
            Configuration reference
          </Link>
        </div>
      </section>
    </div>
  );
}
