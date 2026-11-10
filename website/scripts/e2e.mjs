// Behaviour checks against a running preview: nav links and hash updates, the
// mobile menu, keyboard focus rings, the slider's arrow keys, chart keyboard
// tooltips, the tabs, and the contact form's validation, success and error
// states (with the network call mocked).
// Usage: node scripts/e2e.mjs [baseUrl]
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const base = process.argv[2] ?? 'http://localhost:4321';
let failures = 0;
const check = (ok, msg) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`);
  if (!ok) failures++;
};

const browser = await chromium.launch();

// ---------- Desktop ----------
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(base + '/', { waitUntil: 'networkidle' });

  // Nav links land with the heading visible below the header and the hash updates.
  const ids = ['problem', 'framework', 'research', 'roadmap', 'impact', 'about', 'contact'];
  for (const id of ids) {
    await page.click(`.primary-nav a[data-nav="${id}"]`);
    await page.waitForTimeout(900);
    const r = await page.evaluate((id) => {
      const h = document.querySelector(`#${id} h2`);
      const header = document.getElementById('site-header').getBoundingClientRect();
      const b = h.getBoundingClientRect();
      return { top: b.top, headerBottom: header.bottom, hash: location.hash };
    }, id);
    check(r.top >= r.headerBottom - 1 && r.top < 400, `nav "${id}": heading at ${Math.round(r.top)}px, below header (${Math.round(r.headerBottom)}px)`);
    check(r.hash === `#${id}`, `nav "${id}": hash is ${r.hash}`);
  }

  // Current-section marking.
  await page.click('.primary-nav a[data-nav="roadmap"]');
  await page.waitForTimeout(900);
  check((await page.getAttribute('.primary-nav a[data-nav="roadmap"]', 'aria-current')) === 'true', 'current section marked in nav');

  // Slider arrow keys.
  await page.focus('#eps-range');
  const before = await page.inputValue('#eps-range');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  const after = await page.inputValue('#eps-range');
  const vt = await page.getAttribute('#eps-range', 'aria-valuetext');
  check(Number(after) > Number(before), `slider arrow keys move the value (${before} -> ${after})`);
  check(/^epsilon [\d.]+, /.test(vt ?? ''), `slider aria-valuetext updates: "${vt}"`);
  const errText = await page.textContent('[data-readout="error"]');
  check(errText?.startsWith('±'), `slider readout updates: ${errText?.trim()}`);
  await page.click('[data-redraw]');

  // Hero demo toggle.
  await page.click('.toggle-btn[data-mode="without"]');
  check((await page.getAttribute('.toggle-btn[data-mode="without"]', 'aria-pressed')) === 'true', 'hero demo toggle sets aria-pressed');
  check(await page.locator('[data-hero-demo].is-without').count() === 1, 'hero demo fades the You dot');

  // Tabs.
  await page.click('#research-tabs-tab-1');
  check(await page.isVisible('#research-tabs-panel-1'), 'tab 2 panel shows on click');
  check(await page.isHidden('#research-tabs-panel-0'), 'tab 1 panel hides');
  await page.focus('#research-tabs-tab-1');
  await page.keyboard.press('ArrowRight');
  check(await page.isVisible('#research-tabs-panel-2'), 'tabs move with arrow keys');

  // Chart keyboard tooltips.
  await page.focus('#fig-exp1-recall .chart');
  await page.keyboard.press('ArrowRight');
  const tipVisible = await page.evaluate(() => document.getElementById('chart-tip')?.classList.contains('is-visible'));
  check(Boolean(tipVisible), 'chart tooltip shows on keyboard focus');

  // View data toggle.
  await page.click('#fig-exp1-recall .view-data');
  check(await page.isVisible('#fig-exp1-recall-table'), '"View data" reveals the table');

  // Keyboard pass: every focusable element gets a visible focus ring.
  await page.goto(base + '/', { waitUntil: 'networkidle' });
  const focusables = await page.evaluate(() => {
    const sel = 'a[href], button:not([disabled]), input:not([type="hidden"]), select, textarea, [tabindex="0"]';
    return Array.from(document.querySelectorAll(sel)).filter((el) => el.offsetParent !== null && !el.closest('.hp')).length;
  });
  let ringed = 0;
  let visited = 0;
  const seen = new Set();
  for (let i = 0; i < focusables + 20; i++) {
    await page.keyboard.press('Tab');
    const info = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return null;
      const cs = getComputedStyle(el);
      const key = `${el.tagName}#${el.id}.${el.className}|${el.getAttribute('href') ?? ''}|${(el.textContent ?? '').trim().slice(0, 40)}`;
      const ring = cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) >= 2;
      const shadow = cs.boxShadow && cs.boxShadow !== 'none';
      return { key, ring: ring || shadow || el.tagName === 'INPUT' && el.type === 'range' };
    });
    if (!info || seen.has(info.key)) {
      if (!info) break;
      continue;
    }
    seen.add(info.key);
    visited++;
    if (info.ring) ringed++;
    else console.log(`   no focus ring: ${info.key}`);
  }
  check(ringed === visited && visited > 20, `keyboard pass: ${ringed}/${visited} focused elements show a ring`);

  // Contact form: validation.
  await page.goto(base + '/#contact', { waitUntil: 'networkidle' });
  await page.click('[data-submit]');
  check(await page.isVisible('#cf-name-err'), 'form shows inline validation errors');
  check((await page.getAttribute('#cf-name', 'aria-invalid')) === 'true', 'invalid field has aria-invalid');

  // Contact form: success (mocked network).
  await page.route('**/api/contact', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true }) }));
  await page.fill('#cf-name', 'Test Person');
  await page.fill('#cf-email', 'test@example.org');
  await page.selectOption('#cf-interest', 'Contributing');
  await page.fill('#cf-message', 'Hello, I would like to help benchmark the framework.');
  await page.click('[data-submit]');
  await page.waitForFunction(() => document.querySelector('[data-status]')?.textContent?.includes('Thanks'));
  check(true, 'form success state shows the thank-you message');

  // Contact form: failure.
  await page.unroute('**/api/contact');
  await page.route('**/api/contact', (route) => route.fulfill({ status: 502, contentType: 'application/json', body: JSON.stringify({ ok: false, error: 'The message could not be sent right now.' }) }));
  await page.fill('#cf-name', 'Test Person');
  await page.fill('#cf-email', 'test@example.org');
  await page.selectOption('#cf-interest', 'Contributing');
  await page.fill('#cf-message', 'Hello again, this one should fail on purpose.');
  await page.click('[data-submit]');
  await page.waitForFunction(() => document.querySelector('[data-status]')?.classList.contains('is-error'));
  const errState = await page.textContent('[data-status]');
  check(errState?.includes('could not be sent'), `form error state shows: "${errState?.trim().slice(0, 80)}"`);

  // The real endpoint rejects bad input.
  await page.unroute('**/api/contact');
  const res = await page.evaluate(async () => {
    const fd = new FormData();
    fd.set('name', 'x');
    const r = await fetch('/api/contact', { method: 'POST', body: fd });
    return { status: r.status, body: await r.json() };
  });
  check(res.status === 422 && res.body.fields?.email, `server validation returns 422 with field errors`);
  await page.close();
}

// ---------- Mobile ----------
{
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await page.goto(base + '/', { waitUntil: 'networkidle' });
  const sw = await page.evaluate(() => document.documentElement.scrollWidth);
  check(sw <= 390, `no horizontal scroll at 390px (scrollWidth ${sw})`);
  await page.setViewportSize({ width: 360, height: 780 });
  const sw360 = await page.evaluate(() => document.documentElement.scrollWidth);
  check(sw360 <= 360, `no horizontal scroll at 360px (scrollWidth ${sw360})`);
  await page.setViewportSize({ width: 390, height: 844 });

  await page.click('.menu-btn');
  check(await page.isVisible('#mobile-menu'), 'mobile menu opens');
  check((await page.getAttribute('.menu-btn', 'aria-expanded')) === 'true', 'menu button aria-expanded true');
  // Focus trap: tabbing from the last item wraps to the menu button.
  const items = await page.locator('#mobile-menu a, #mobile-menu button').count();
  for (let i = 0; i < items; i++) await page.keyboard.press('Tab');
  const active = await page.evaluate(() => document.activeElement?.className);
  check(active?.includes('menu-btn'), `focus wraps inside the menu (active: ${active})`);
  await page.keyboard.press('Escape');
  check(await page.isHidden('#mobile-menu'), 'Escape closes the menu');
  await page.click('.menu-btn');
  await page.click('#mobile-menu a[data-nav="research"]');
  await page.waitForTimeout(800);
  check(await page.isHidden('#mobile-menu'), 'tapping a link closes the menu');
  check((await page.evaluate(() => location.hash)) === '#research', 'mobile link updates the hash');

  const rw = await page.goto(base + '/research', { waitUntil: 'networkidle' });
  check(rw?.status() === 200, '/research loads');
  const sw2 = await page.evaluate(() => document.documentElement.scrollWidth);
  check(sw2 <= 390, `/research has no horizontal scroll at 390px (scrollWidth ${sw2})`);
  const nf = await page.goto(base + '/nope', { waitUntil: 'networkidle' });
  check(nf?.status() === 404, `missing page returns 404 (${nf?.status()})`);
  await page.close();
}

await browser.close();
if (failures) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nall checks passed');
