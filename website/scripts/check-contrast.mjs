// WCAG AA contrast check for every text/background pair the site uses.
const tokens = {
  bg: '#ffffff',
  surface: '#f6f8fb',
  ink: '#101828',
  text: '#344054',
  muted: '#667085',
  blue: '#1a5cf4',
  blueDeep: '#0b3fd6',
  blueSoft: '#8fc0f8',
  blueTint: '#eef4ff',
  error: '#b42318',
  inkText: '#f6f8fb',
  inkMuted: '#98a2b3',
  white: '#ffffff',
};

const lum = (hex) => {
  const c = hex.replace('#', '');
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(c.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => {
  const [l1, l2] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
};

// [foreground, background, minimum, description]
const pairs = [
  ['ink', 'bg', 4.5, 'headings on white'],
  ['ink', 'surface', 4.5, 'headings on surface'],
  ['text', 'bg', 4.5, 'body on white'],
  ['text', 'surface', 4.5, 'body on surface'],
  ['muted', 'bg', 4.5, 'captions on white'],
  ['muted', 'surface', 4.5, 'captions on surface'],
  ['blue', 'bg', 4.5, 'links and stats on white'],
  ['blue', 'surface', 4.5, 'links on surface'],
  ['blue', 'blueTint', 4.5, 'blue text on tint'],
  ['white', 'blue', 4.5, 'primary button text'],
  ['white', 'blueDeep', 4.5, 'primary button hover text'],
  ['error', 'bg', 4.5, 'form errors on white'],
  ['white', 'ink', 4.5, 'headings on dark section'],
  ['inkText', 'ink', 4.5, 'body on dark section'],
  ['inkMuted', 'ink', 4.5, 'captions on dark section'],
  ['blueSoft', 'ink', 4.5, 'numbers on dark section'],
  ['blue', 'bg', 3, 'focus ring (non-text)'],
];

let failed = 0;
for (const [fg, bg, min, what] of pairs) {
  const r = ratio(tokens[fg], tokens[bg]);
  const ok = r >= min;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${r.toFixed(2)}:1  ${fg} on ${bg}  (${what}, min ${min})`);
}
if (failed) {
  console.error(`${failed} pair(s) below the minimum`);
  process.exit(1);
}
console.log('all pairs pass');
