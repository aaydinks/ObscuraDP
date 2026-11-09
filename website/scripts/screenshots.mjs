// Full-page and per-section screenshots at 1440x900 and 390x844.
// Usage: node scripts/screenshots.mjs [baseUrl] [outDir]
import { mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';

// Playwright may be installed globally (NODE_PATH) rather than in this package.
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const base = process.argv[2] ?? 'http://localhost:4321';
const out = process.argv[3] ?? 'screenshots';
mkdirSync(out, { recursive: true });

const viewports = [
  { name: 'desktop', width: 1440, height: 900 },
  { name: 'mobile', width: 390, height: 844 },
];
const pages = [
  { path: '/', name: 'home', sections: ['hero', 'problem', 'mission', 'framework', 'research', 'roadmap', 'impact', 'about', 'contact', 'footer'] },
  { path: '/research', name: 'research', sections: [] },
  { path: '/credits', name: 'credits', sections: [] },
  { path: '/does-not-exist', name: '404', sections: [] },
];

const browser = await chromium.launch();
for (const vp of viewports) {
  const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  for (const p of pages) {
    await page.goto(base + p.path, { waitUntil: 'networkidle' });
    await page.evaluate(() => document.fonts.ready);
    // Make draw-in charts visible for the capture.
    await page.evaluate(() => document.querySelectorAll('.chart[data-animate]').forEach((c) => c.classList.add('is-visible')));
    await page.screenshot({ path: `${out}/${p.name}-${vp.name}-full.png`, fullPage: true });
    if (p.sections.length) await page.addStyleTag({ content: '.site-header{position:absolute !important}' });
    for (const s of p.sections) {
      const sel = s === 'hero' ? '.hero' : s === 'footer' ? '.site-footer' : `#${s}`;
      const el = page.locator(sel).first();
      if ((await el.count()) === 0) continue;
      await el.scrollIntoViewIfNeeded();
      await el.screenshot({ path: `${out}/${p.name}-${vp.name}-${s}.png` });
    }
    const scrollW = await page.evaluate(() => document.documentElement.scrollWidth);
    if (scrollW > vp.width) console.log(`horizontal overflow on ${p.path} at ${vp.name}: ${scrollW}px`);
  }
  await ctx.close();
}
await browser.close();
console.log(`screenshots written to ${out}/`);
