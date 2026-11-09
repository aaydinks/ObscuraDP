// Fails if site copy contains an em dash (U+2014) or a banned marketing word.
// Scans src/ (excluding data and node code that never renders) and, when
// present, the built HTML in dist/. The mission and vision statements are
// quoted word for word and are exempt (marked with data-copy-exempt).
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, extname, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const BANNED = [
  'revolutioni[sz]e',
  'unlock',
  'unleash',
  'empower',
  'seamless',
  'cutting-edge',
  'game-changing',
  'next-generation',
  'harness',
  'supercharge',
  'leverage',
  'robust',
  'in today’s digital landscape',
  "in today's digital landscape",
  'join us on this journey',
];
const bannedRe = new RegExp(`\\b(${BANNED.join('|')})`, 'i');

function walk(dir, exts, out = []) {
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (name === 'node_modules' || name === 'data' || name === '_astro' || name === '_worker.js' || name === 'server') continue;
      walk(p, exts, out);
    } else if (exts.includes(extname(name))) out.push(p);
  }
  return out;
}

// Remove exempt blocks (mission/vision quotes) from a text before scanning.
function stripExempt(text) {
  return text
    .replace(/<blockquote[^>]*data-copy-exempt[^>]*>[\s\S]*?<\/blockquote>/g, '')
    .replace(/<blockquote[^>]*data-astro-cid-[^>]*>[\s\S]*?<\/blockquote>/g, (m) => (/empower|envision/.test(m) ? '' : m));
}

const files = [
  ...walk(join(root, 'src'), ['.astro', '.ts', '.md']),
  ...walk(join(root, 'dist', 'client'), ['.html']),
  ...walk(join(root, 'scripts', 'og'), ['.html']),
];

let problems = 0;
for (const f of files) {
  if (f.endsWith('lint-copy.mjs') || f.includes('copy-rules')) continue;
  const raw = readFileSync(f, 'utf8');
  const text = stripExempt(raw);
  const lines = text.split('\n');
  lines.forEach((line, i) => {
    // Skip comments that explain the rule itself.
    if (/^\s*\/\//.test(line)) return;
    if (line.includes('—')) {
      problems++;
      console.log(`${f}:${i + 1}: em dash: ${line.trim().slice(0, 100)}`);
    }
    const m = line.match(bannedRe);
    if (m) {
      problems++;
      console.log(`${f}:${i + 1}: banned word "${m[1]}": ${line.trim().slice(0, 100)}`);
    }
  });
}
if (problems) {
  console.error(`\n${problems} problem(s) found.`);
  process.exit(1);
}
console.log(`copy lint passed (${files.length} files)`);
