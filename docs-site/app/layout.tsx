import { Geist } from 'next/font/google';
import { Provider } from '@/components/provider';
import type { ReactNode } from 'react';
import './global.css';

const geist = Geist({
  subsets: ['latin'],
});

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={geist.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}