import { Geist } from 'next/font/google';
import { Provider } from '@/components/provider';
import { appName } from '@/lib/shared';
import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import './global.css';

const geist = Geist({
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://marius-brt.github.io/rs-infer/'),
  title: {
    default: appName,
    template: `%s | ${appName}`,
  },
  description:
    'A Rust inference server for ONNX Runtime — embeddings, rerankers, PII detection, and zero-shot classification through one HTTP API.',
  openGraph: {
    title: appName,
    siteName: appName,
    images: ['/og-image.png'],
  },
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={geist.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}