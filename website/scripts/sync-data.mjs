// Copies the experiment result JSON files this site renders into src/data/.
// Run from website/: `npm run sync-data`. The copies are committed so a
// build never depends on the parent folders being present.
import { copyFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const outDir = resolve(here, '..', 'src', 'data');

const files = [
  ['Experiment 1/results/results_multiseed.json', 'exp1-results_multiseed.json'],
  ['Experiment 1/results/results.json', 'exp1-results.json'],
  ['Experiment 2/results/results_multiseed.json', 'exp2-results_multiseed.json'],
  ['Experiment 2/results/results.json', 'exp2-results.json'],
  ['Experiment 3/results/overhead.json', 'exp3-overhead.json'],
  ['Experiment 3/results/spe.json', 'exp3-spe.json'],
  ['Experiment 3/results/accuracy.json', 'exp3-accuracy.json'],
];

mkdirSync(outDir, { recursive: true });
let copied = 0;
for (const [from, to] of files) {
  const src = resolve(repoRoot, from);
  if (!existsSync(src)) {
    console.error(`missing: ${src}`);
    process.exitCode = 1;
    continue;
  }
  copyFileSync(src, resolve(outDir, to));
  copied += 1;
  console.log(`${from}  ->  src/data/${to}`);
}
console.log(`${copied}/${files.length} files copied`);
