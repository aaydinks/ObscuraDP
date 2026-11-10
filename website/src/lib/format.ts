export const pct = (x: number, digits = 2) => `${(x * 100).toFixed(digits)}%`;
export const fixed = (x: number, digits = 2) => x.toFixed(digits);

export function compactNumber(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return `${n}`;
}

export function bytesToMB(bytes: number, digits = 0): string {
  return `${(bytes / 1e6).toFixed(digits)} MB`;
}

export function msToSeconds(ms: number, digits = 2): string {
  return `${(ms / 1000).toFixed(digits)} s`;
}

export function plusMinus(mean: number, std: number, digits = 2, scale = 1): string {
  return `${(mean * scale).toFixed(digits)} ± ${(std * scale).toFixed(digits)}`;
}

export function formatEpsilon(eps: number): string {
  if (eps >= 10) return eps.toFixed(0);
  if (eps >= 1) return eps.toFixed(1);
  return eps.toFixed(2);
}
