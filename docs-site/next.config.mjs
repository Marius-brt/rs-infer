import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  // Static export for GitHub Pages (no Node.js server needed).
  output: 'export',
  trailingSlash: true,
  images: { unoptimized: true },
  // Set NEXT_BASE_PATH when deploying to a project site on GitHub Pages
  // (e.g. /my-repo). Leave unset for local dev or user sites.
  ...(process.env.NEXT_BASE_PATH
    ? { basePath: process.env.NEXT_BASE_PATH, assetPrefix: `${process.env.NEXT_BASE_PATH}/` }
    : {}),
};

export default withMDX(config);