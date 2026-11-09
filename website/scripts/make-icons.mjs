// Fallback icons from the placeholder mark (used until the real logo mark is
// supplied; extract-brand.py overwrites these). Writes public/favicon.svg,
// public/favicon.ico and public/apple-touch-icon.png.
import sharp from 'sharp';
import { writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const svg = (size, pad, bg) => `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 32 32">
${bg ? `<rect width="32" height="32" rx="6" fill="${bg}"/>` : ''}
<g transform="translate(${pad},${pad}) scale(${(32 - 2 * pad) / 32})">
<circle cx="16" cy="16" r="13" fill="none" stroke="#1a5cf4" stroke-width="2.5"/>
<path d="M16 7.5a8.5 8.5 0 0 1 0 17z" fill="#1a5cf4"/>
</g></svg>`;

writeFileSync(resolve(root, 'public/favicon.svg'), svg(32, 0, null));
await sharp(Buffer.from(svg(180, 3, '#ffffff'))).resize(180, 180).png().toFile(resolve(root, 'public/apple-touch-icon.png'));

// ICO: a 32x32 PNG wrapped in an ICO container (PNG-in-ICO is supported by all modern browsers).
const png32 = await sharp(Buffer.from(svg(32, 0, null))).resize(32, 32).png().toBuffer();
const header = Buffer.alloc(6);
header.writeUInt16LE(0, 0);
header.writeUInt16LE(1, 2);
header.writeUInt16LE(1, 4);
const entry = Buffer.alloc(16);
entry.writeUInt8(32, 0);
entry.writeUInt8(32, 1);
entry.writeUInt8(0, 2);
entry.writeUInt8(0, 3);
entry.writeUInt16LE(1, 4);
entry.writeUInt16LE(32, 6);
entry.writeUInt32LE(png32.length, 8);
entry.writeUInt32LE(22, 12);
writeFileSync(resolve(root, 'public/favicon.ico'), Buffer.concat([header, entry, png32]));
console.log('icons written to public/');
