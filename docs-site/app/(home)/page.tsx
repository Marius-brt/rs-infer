import Link from 'next/link';
import { appName } from '@/lib/shared';

export default function HomePage() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center text-center px-6 py-24">
      <h1 className="text-4xl font-bold tracking-tight md:text-5xl">{appName}</h1>
      <p className="mt-4 max-w-xl text-lg text-fd-muted-foreground">
        An ONNX Runtime inference server in Rust — embeddings, rerankers, PII
        detection, and zero-shot classification through one HTTP API.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-4">
        <Link
          href="/docs"
          className="inline-flex items-center rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-medium text-fd-primary-foreground hover:opacity-90"
        >
          Get started
        </Link>
        <Link
          href="/docs/configuration"
          className="inline-flex items-center rounded-lg border border-fd-border px-5 py-2.5 text-sm font-medium hover:bg-fd-accent"
        >
          Configuration reference
        </Link>
      </div>
    </div>
  );
}