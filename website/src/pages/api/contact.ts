// Contact form endpoint. Runs on Cloudflare Workers.
// 1. Validate fields and the honeypot. 2. Check the Turnstile token.
// 3. Email the message with Resend. 4. Store nothing.
import type { APIRoute } from 'astro';
import { RESEND_API_KEY, TURNSTILE_SECRET_KEY, CONTACT_TO_EMAIL, CONTACT_FROM_EMAIL, CONTACT_MOCK } from 'astro:env/server';

export const prerender = false;

const INTERESTS = new Set(['Using the framework', 'Contributing', 'Partnering or piloting', 'Press or other']);
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });

const str = (fd: FormData, key: string, max: number) => {
  const v = fd.get(key);
  return typeof v === 'string' ? v.trim().slice(0, max) : '';
};

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c);

async function verifyTurnstile(token: string, ip: string | null): Promise<boolean> {
  if (!TURNSTILE_SECRET_KEY) return true; // not configured: skip (documented in README)
  const body = new URLSearchParams({ secret: TURNSTILE_SECRET_KEY, response: token });
  if (ip) body.set('remoteip', ip);
  const res = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', { method: 'POST', body });
  if (!res.ok) return false;
  const data = (await res.json()) as { success?: boolean };
  return Boolean(data.success);
}

export const POST: APIRoute = async ({ request }) => {
  let fd: FormData;
  try {
    fd = await request.formData();
  } catch {
    return json({ ok: false, error: 'The form could not be read.' }, 400);
  }

  // Honeypot: bots fill it, people never see it. Pretend success so bots stop.
  if (str(fd, 'website', 10)) return json({ ok: true });

  const name = str(fd, 'name', 120);
  const email = str(fd, 'email', 200);
  const organization = str(fd, 'organization', 160);
  const interest = str(fd, 'interest', 60);
  const message = str(fd, 'message', 4000);

  const fields: Record<string, string> = {};
  if (name.length < 2) fields.name = 'Please enter your name.';
  if (!EMAIL_RE.test(email)) fields.email = 'Please enter a valid email address.';
  if (!INTERESTS.has(interest)) fields.interest = 'Please choose one option.';
  if (message.length < 10) fields.message = 'Please write a few more words (at least 10 characters).';
  if (Object.keys(fields).length) return json({ ok: false, error: 'Please fix the highlighted fields.', fields }, 422);

  const token = str(fd, 'cf-turnstile-response', 4096);
  const ip = request.headers.get('cf-connecting-ip');
  if (TURNSTILE_SECRET_KEY && !token) return json({ ok: false, error: 'Please complete the verification challenge.' }, 400);
  if (!(await verifyTurnstile(token, ip))) return json({ ok: false, error: 'Verification failed. Please try again.' }, 400);

  const to = CONTACT_TO_EMAIL;
  const subject = `[Obscura DP] ${interest}: ${name}`;
  const text = [
    `Name: ${name}`,
    `Email: ${email}`,
    `Organization: ${organization || '(none)'}`,
    `Interested in: ${interest}`,
    '',
    message,
  ].join('\n');
  const html = `<p><strong>Name:</strong> ${escapeHtml(name)}<br><strong>Email:</strong> ${escapeHtml(email)}<br><strong>Organization:</strong> ${escapeHtml(organization || '(none)')}<br><strong>Interested in:</strong> ${escapeHtml(interest)}</p><p>${escapeHtml(message).replace(/\n/g, '<br>')}</p>`;

  if (CONTACT_MOCK || !RESEND_API_KEY) {
    if (!CONTACT_MOCK) return json({ ok: false, error: 'Email delivery is not configured yet.' }, 503);
    console.log('[contact mock] would send:', { to, subject });
    return json({ ok: true, mock: true });
  }
  if (!to) return json({ ok: false, error: 'Email delivery is not configured yet.' }, 503);

  const res = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: { authorization: `Bearer ${RESEND_API_KEY}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      from: CONTACT_FROM_EMAIL,
      to: [to],
      reply_to: email,
      subject,
      text,
      html,
    }),
  });
  if (!res.ok) return json({ ok: false, error: 'The message could not be sent right now.' }, 502);
  return json({ ok: true });
};

export const GET: APIRoute = () => json({ ok: false, error: 'Use POST.' }, 405);
