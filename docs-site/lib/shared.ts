export const appName = 'RS Infer';

// Prefix for static assets (public/) when served from a project site, matching
// the `basePath` logic in next.config.mjs (NEXT_BASE_PATH=/repo-name).
export const assetBase = process.env.NEXT_BASE_PATH ?? '';

export const docsRoute = '/docs';

// Set NEXT_PUBLIC_GITHUB_REPO (e.g. "marius-brt/rs-infer") to link the docs site
// to the source repository.
export const gitConfig = {
  repo: process.env.NEXT_PUBLIC_GITHUB_REPO ?? 'marius-brt/rs-infer',
  branch: 'main',
};