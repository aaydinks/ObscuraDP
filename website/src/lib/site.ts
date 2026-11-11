// Site-wide constants. Placeholders in square brackets are meant to be replaced
// once the real values exist; they are collected here so it is a single edit.
export const SITE = {
  name: 'Obscura DP',
  url: 'https://obscuradp.org',
  title: 'Obscura DP: Mathematically guaranteed privacy for AI',
  description:
    'Obscura DP is an open-source privacy framework for AI that combines differential privacy, federated learning and homomorphic encryption in one pipeline, free for any model training workflow. Research stage, open-source release planned for 2026-27.',
  github: 'https://github.com/aaydinks/ObscuraDP',
  githubTree: 'https://github.com/aaydinks/ObscuraDP/tree/main',
  contactEmail: 'aayushdinker@gmail.com',
  preregistrationUrl: '', // Pre-registration link: empty until supplied
  founderLinks: [] as { label: string; href: string }[], // [FOUNDER LINKS]
  statusLine: 'Research stage · Open-source release planned for 2026–27',
  introVideo: {
    id: 'ez8MoY2MsYU',
    url: 'https://youtu.be/ez8MoY2MsYU',
    title: 'Obscura DP: project introduction',
  },
};

export const NAV = [
  { id: 'problem', label: 'Problem' },
  { id: 'framework', label: 'Framework' },
  { id: 'research', label: 'Research' },
  { id: 'roadmap', label: 'Roadmap' },
  { id: 'impact', label: 'Impact' },
  { id: 'about', label: 'About' },
  { id: 'contact', label: 'Contact' },
];

export const EXPERIMENT_FOLDERS = {
  exp1: `${SITE.githubTree}/Experiment%201`,
  exp2: `${SITE.githubTree}/Experiment%202`,
  exp3: `${SITE.githubTree}/Experiment%203`,
};

export const PAPERS = {
  exp1: { label: 'arXiv:2508.10017', href: 'https://arxiv.org/abs/2508.10017' },
  exp2: {
    label: 'Sci. Rep. 2025, DOI 10.1038/s41598-025-34536-9',
    href: 'https://doi.org/10.1038/s41598-025-34536-9',
  },
  exp3: { label: 'FedML-HE, arXiv:2303.10837', href: 'https://arxiv.org/abs/2303.10837' },
};
