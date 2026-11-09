# obscuradp.org

The public website for Obscura DP. Built with Astro (static HTML, TypeScript islands, build-time KaTeX and SVG charts) and deployed to Cloudflare Workers with static assets. One server route, `/api/contact`, handles the contact form.

## Requirements

- Node 22.23.4 or newer (see `.node-version`).
- npm 9.6.5 or newer.

## Commands

| Command | What it does |
|---|---|
| `npm install` | Install dependencies. |
| `npm run dev` | Dev server at `http://localhost:4321`. |
| `npm run build` | Type-check (`astro check`) and build into `dist/`. |
| `npm run preview` | Serve the build with the Cloudflare runtime (`wrangler dev`). |
| `npm run deploy` | Deploy with Wrangler from your machine (`wrangler deploy`). |
| `npm run sync-data` | Copy the experiment result JSON files from `../Experiment */results/` into `src/data/`. The copies are committed, so builds never depend on the parent folders. Re-run after new experiment results land. |
| `npm run og` | Re-render `public/og.png` (1200×630) from `scripts/og/template.html` with Playwright. |
| `npm run lint:copy` | Fail on em dashes and banned marketing words in site copy and built HTML. |
| `npm run check:contrast` | Check every text/background pair for WCAG AA. |
| `npm run screenshots` | Playwright screenshots of every section at 1440×900 and 390×844 (needs a running preview). |
| `npm run test:e2e` | Behaviour checks: navigation, mobile menu, keyboard, slider, tabs, form states (needs a running preview). |

The Playwright scripts use the globally installed Playwright when the package has none (`createRequire` + `NODE_PATH`).

## Project layout

```
src/
  data/            JSON copied from the experiments (committed)
  assets/brand/    logo.png, mark.png, founder.png (add these; text fallbacks render until then)
  assets/photos/   photo-a.jpg, photo-b.jpg (add these; placeholders render until then)
  styles/          tokens.css (design tokens), base.css, fonts.css
  lib/             site.ts (constants and placeholders), chart.ts (SVG helpers), rng.ts, laplace.ts
  layouts/         Base.astro (head, header, footer, JSON-LD, Open Graph)
  components/      Header, Footer, Figure, Tabs, Math, Photo, Logo, ContactForm, visuals/, charts/
  sections/        One file per home-page section
  pages/           index, research, credits, 404, api/contact.ts
scripts/           sync-data, og/, extract-brand.py, make-icons, lint-copy, check-contrast, screenshots, e2e
```

Everything that still needs a real value is in `src/lib/site.ts`: `contactEmail`, `preregistrationUrl`, `founderLinks`.

## Brand assets

The logo, mark and founder photo come from the slide deck PDF. Once you have the PDF:

```bash
python3 scripts/extract-brand.py path/to/deck.pdf --list   # check image indices
python3 scripts/extract-brand.py path/to/deck.pdf          # writes src/assets/brand/* and the icons
npm run og
```

The script needs `pdfimages` (poppler-utils) and Pillow. Until the files exist, the header shows a text wordmark, the diagrams use a simple placeholder mark, and the About section shows a portrait placeholder. If the raster logo looks soft at 28 to 32 px, replace `src/assets/brand/logo.png` with an SVG export and update `Logo.astro`.

## Photos

Put the two Unsplash photos at `src/assets/photos/photo-a.jpg` (Problem section, community health worker or clinic scene) and `src/assets/photos/photo-b.jpg` (About section, students around a laptop). Use only photos under the free Unsplash License, not Unsplash+. Record photographer, link and license in `CREDITS.md` and in `src/lib/credits.ts` (which renders `/credits`). Astro serves them as AVIF and WebP at several widths.

## Contact form

Flow: browser validation → `POST /api/contact` → honeypot check → server validation → Turnstile `siteverify` → one email through Resend's REST API → nothing stored.

Configuration (all four documented here, none in code):

| Name | Kind | Where | Purpose |
|---|---|---|---|
| `RESEND_API_KEY` | Secret | Cloudflare → Worker → Settings → Variables and Secrets, or `npx wrangler secret put RESEND_API_KEY` | Sends the email. |
| `TURNSTILE_SECRET_KEY` | Secret | Same as above | Verifies the Turnstile token server-side. If unset, verification is skipped (development only). |
| `CONTACT_TO_EMAIL` | Variable | `wrangler.jsonc` `vars`, or the dashboard | Inbox that receives messages. |
| `PUBLIC_TURNSTILE_SITE_KEY` | Variable (public) | `wrangler.jsonc` `vars`, or the dashboard | Renders the Turnstile widget. If unset, no widget is rendered. |
| `CONTACT_MOCK` | Variable (optional) | `.dev.vars` locally | `true` logs the email instead of sending it. |

The `from` address is `contact@obscuradp.org`; verify the `obscuradp.org` domain in Resend first, or change the address in `src/pages/api/contact.ts`.

Local development: copy `.dev.vars.example` to `.dev.vars`. It uses Cloudflare's public Turnstile test keys (the widget always passes) and `CONTACT_MOCK=true`.

## Deploy to Cloudflare

The Astro Cloudflare adapter targets Workers with static assets (Cloudflare Pages is no longer supported by the adapter). `wrangler.jsonc` is already configured: the Worker entry is the adapter's server entrypoint, static files are served from `dist/`, and a missing path returns the built `404.html`.

### One-time setup in the Cloudflare dashboard

1. **Create the Worker from the repository.** Workers & Pages → Create → Workers → *Import a repository* → pick `aaydinks/ObscuraDP` → connect GitHub if asked.
2. **Build settings.**
   - Root directory: `website`
   - Build command: `npm run build`
   - Deploy command: `npx wrangler deploy`
   - Build variable: `NODE_VERSION` = `22.23.4` (the `.node-version` file in `website/` covers this too)
   - Production branch: `main`
3. **Variables and secrets.** Worker → Settings → Variables and Secrets.
   - Add secrets `RESEND_API_KEY` and `TURNSTILE_SECRET_KEY`.
   - Add plain variables `CONTACT_TO_EMAIL` and `PUBLIC_TURNSTILE_SITE_KEY` (or fill the `vars` block in `wrangler.jsonc` and commit).
   - `PUBLIC_TURNSTILE_SITE_KEY` is read at build time, so it must also be present as a build variable.
4. **Turnstile.** Cloudflare dashboard → Turnstile → Add widget → hostname `obscuradp.org` → copy the site key and secret key into the two settings above.
5. **Resend.** Resend → Domains → add and verify `obscuradp.org` (DNS records go into the Cloudflare zone) → API Keys → create a sending key → `RESEND_API_KEY`.
6. **Custom domain.** Worker → Settings → Domains & Routes → Add → Custom domain → `obscuradp.org`. Add `www.obscuradp.org` too, or create a redirect rule from `www` to the apex. The zone must already be on Cloudflare.
7. **Deploy.** Save; Cloudflare builds and deploys. Every later push to `main` redeploys. Pushes to other branches get preview URLs.

### Deploy from a terminal instead

```bash
cd website
npm ci
npm run build
npx wrangler login
npx wrangler secret put RESEND_API_KEY
npx wrangler secret put TURNSTILE_SECRET_KEY
npx wrangler deploy
```

## Privacy and analytics

No cookies, no third-party trackers, no analytics. The only third-party script is Cloudflare Turnstile on the contact form, and only when a site key is configured. If analytics is ever wanted, use Cloudflare Web Analytics (cookieless), enabled in the dashboard without touching the code.

## Quality checks

```bash
npm run build          # astro check + build, zero errors and warnings
npm run lint:copy      # no em dashes, no banned words (mission and vision are exempt)
npm run check:contrast # WCAG AA on every token pair in use
npm run preview        # then, in another terminal:
npm run test:e2e
npm run screenshots
```

Lighthouse (mobile) targets 95 or higher in Performance, Accessibility, Best Practices and SEO on `/` and `/research`.
