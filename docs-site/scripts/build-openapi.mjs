import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { load } from 'js-yaml';

const src = fileURLToPath(new URL('../openapi.yaml', import.meta.url));
const out = fileURLToPath(new URL('../openapi.json', import.meta.url));

const doc = load(readFileSync(src, 'utf8'));
writeFileSync(out, JSON.stringify(doc, null, 2) + '\n');

console.log(`[openapi] wrote ${out}`);