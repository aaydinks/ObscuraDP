// Laplace noise with scale b: inverse-CDF sampling from a uniform in (0, 1).
export function laplace(rand: () => number, scale: number): number {
  const u = rand() - 0.5;
  const safe = Math.max(1e-12, Math.abs(u));
  return -scale * Math.sign(u) * Math.log(1 - 2 * safe);
}

// Released count: true count plus Laplace(1/epsilon), rounded and clamped at 0.
export function releaseCount(trueCount: number, epsilon: number, rand: () => number): number {
  return Math.max(0, Math.round(trueCount + laplace(rand, 1 / epsilon)));
}

// Worst-case posterior that a specific person is in the data, from a 50/50 prior.
export function attackerCeiling(epsilon: number): number {
  const e = Math.exp(epsilon);
  return e / (1 + e);
}
