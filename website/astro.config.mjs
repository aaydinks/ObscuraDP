// @ts-check
import { defineConfig, envField } from 'astro/config';
import cloudflare from '@astrojs/cloudflare';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://obscuradp.org',
  output: 'static',
  session: false,
  trailingSlash: 'never',
  adapter: cloudflare({
    imageService: 'compile',
  }),
  integrations: [
    sitemap({
      filter: (page) => !page.includes('/api/'),
    }),
  ],
  env: {
    schema: {
      RESEND_API_KEY: envField.string({ context: 'server', access: 'secret', optional: true }),
      TURNSTILE_SECRET_KEY: envField.string({ context: 'server', access: 'secret', optional: true }),
      CONTACT_TO_EMAIL: envField.string({ context: 'server', access: 'public', optional: true }),
      CONTACT_FROM_EMAIL: envField.string({ context: 'server', access: 'public', optional: true, default: 'Obscura DP <onboarding@resend.dev>' }),
      CONTACT_MOCK: envField.boolean({ context: 'server', access: 'public', optional: true, default: false }),
      PUBLIC_TURNSTILE_SITE_KEY: envField.string({ context: 'client', access: 'public', optional: true }),
    },
  },
  build: {
    format: 'file',
    inlineStylesheets: 'auto',
  },
  vite: {
    build: {
      assetsInlineLimit: 0,
    },
  },
});
