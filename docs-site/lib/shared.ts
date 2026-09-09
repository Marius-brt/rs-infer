export const appName = 'ort-infer';

export const docsRoute = '/docs';

// Set NEXT_PUBLIC_GITHUB_REPO (e.g. "user/ort-infer") to link the docs site
// to the source repository.
export const gitConfig = {
  repo: process.env.NEXT_PUBLIC_GITHUB_REPO ?? 'user/ort-infer',
  branch: 'main',
};