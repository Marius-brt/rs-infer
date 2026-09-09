import { loader } from 'fumadocs-core/source';
import { defineDocs } from 'fumadocs-mdx/macro';
import { openapi } from './openapi';
import { docsRoute } from './shared';

const docs = defineDocs({
  dir: 'content/docs',
});

// Combine Markdown/MDX pages with virtual pages generated from the OpenAPI
// schema. See https://fumadocs.dev/docs/integrations/openapi for details.
export const source = loader(
  {
    docs: docs.toFumadocsSource(),
    openapi: await openapi.staticSource({
      baseDir: 'api/(generated)',
      groupBy: 'tag',
      meta: {
        folderStyle: 'separator',
      },
    }),
  },
  {
    baseUrl: docsRoute,
    plugins: [openapi.loaderPlugin()],
  },
);