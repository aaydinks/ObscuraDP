// Renders public/og.png (1200x630) from template.html with Playwright.
// If src/assets/brand/logo.png exists it replaces the text wordmark.
import { createRequire } from 'node:module';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');
const template = resolve(here, 'template.html');
const logo = resolve(root, 'src/assets/brand/logo.png');
const out = resolve(root, 'public/og.png');

let html = readFileSync(template, 'utf8');
if (existsSync(logo)) {
  const b64 = readFileSync(logo).toString('base64');
  html = html.replace(/<div class="brand" id="brand">[\s\S]*?<\/div>/, `<div class="brand" id="brand"><img src="data:image/png;base64,${b64}" alt="Obscura DP"></div>`);
}
const tmp = resolve(here, '.rendered.html');
writeFileSync(tmp, html);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(tmp).href);
await page.evaluate(() => document.fonts.ready);
await page.screenshot({ path: out, type: 'png' });
await browser.close();
console.log(`wrote ${out}`);
