# rs-infer documentation site

The public documentation is built with [Fumadocs](https://fumadocs.dev)
(a Next.js + Tailwind documentation framework) and published to GitHub Pages.

## Local development

```bash
cd docs-site
npm install
npm run dev          # http://localhost:3000
```

## Adding or editing content

- **Narrative pages** live in [`content/docs/`](content/docs/) as `.mdx` files.
  Add a page and list its slug in
  [`content/docs/meta.json`](content/docs/meta.json) to include it in the
  sidebar. Frontmatter supports `title`, `description`, `icon`, `full`, `tag`.
- **API reference** pages are generated at build time from
  [`openapi.yaml`](openapi.yaml) using
  [Fumadocs OpenAPI](https://www.fumadocs.dev/docs/integrations/openapi).
  Edit that YAML file (add operations, change schemas, group them with `tags`)
  and the sidebar regenerates automatically. The `prebuild` script converts
  the YAML to JSON for the loader — run it manually with `npm run build:openapi`
  if you want to inspect `openapi.json`.

## Build for GitHub Pages

The static export goes to `out/`. When deployed to a project site at
`https://<user>.github.io/<repo>/`, set `NEXT_BASE_PATH=/<repo>` so asset URLs
match:

```bash
NEXT_BASE_PATH=/my-repo npm run build
```

## Deployment

Push to `main` (or `master`) triggers `.github/workflows/docs.yml`, which
builds a static export and publishes it via GitHub Pages. The base path and
the "GitHub" nav link are set automatically from `github.event.repository.name`
and `github.repository`.

In your repo's **Settings → Pages**, set **Source** to **GitHub Actions**
(first deployment only — after that it stays on Actions).