// Header behaviour: scrolled shadow, current-section marking with
// IntersectionObserver, hash updates, and the mobile menu with a focus trap.
export function initNav(): void {
  const header = document.getElementById('site-header');
  if (!header) return;

  const onScroll = () => header.classList.toggle('is-scrolled', window.scrollY > 4);
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  // Mobile menu
  const btn = header.querySelector<HTMLButtonElement>('.menu-btn');
  const menu = header.querySelector<HTMLElement>('#mobile-menu');
  if (btn && menu) {
    const focusables = () =>
      Array.from(
        menu.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'),
      );
    const open = () => {
      menu.hidden = false;
      btn.setAttribute('aria-expanded', 'true');
      btn.setAttribute('aria-label', 'Close menu');
      document.body.style.overflow = 'hidden';
      focusables()[0]?.focus();
    };
    const close = (refocus = true) => {
      menu.hidden = true;
      btn.setAttribute('aria-expanded', 'false');
      btn.setAttribute('aria-label', 'Open menu');
      document.body.style.overflow = '';
      if (refocus) btn.focus();
    };
    btn.addEventListener('click', () => (menu.hidden ? open() : close()));
    menu.addEventListener('click', (e) => {
      if ((e.target as HTMLElement).closest('a')) close(false);
    });
    document.addEventListener('keydown', (e) => {
      if (menu.hidden) return;
      if (e.key === 'Escape') {
        e.preventDefault();
        close();
        return;
      }
      if (e.key === 'Tab') {
        const items = [btn, ...focusables()];
        const first = items[0];
        const last = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    });
    window.matchMedia('(min-width: 1024px)').addEventListener('change', (ev) => {
      if (ev.matches && !menu.hidden) close(false);
    });
  }

  // Current section marking (home page only, where sections exist)
  const links = Array.from(header.querySelectorAll<HTMLAnchorElement>('.primary-nav a[data-nav]'));
  const sections = links
    .map((a) => document.getElementById(a.dataset.nav ?? ''))
    .filter((s): s is HTMLElement => Boolean(s));
  if (!sections.length || !('IntersectionObserver' in window)) return;

  const headerH = header.offsetHeight;
  let current = '';
  const setCurrent = (id: string) => {
    if (id === current) return;
    current = id;
    links.forEach((a) => {
      if (a.dataset.nav === id) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    });
    const hash = id ? `#${id}` : window.location.pathname;
    if (window.location.hash !== hash) history.replaceState(null, '', hash);
  };

  // While a clicked in-page link is scrolling, keep its hash instead of the
  // sections passing by. Released shortly after scrolling stops.
  let lockedTo: string | null = null;
  let releaseTimer = 0;
  header.querySelectorAll<HTMLAnchorElement>('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', () => {
      const id = a.getAttribute('href')!.slice(1);
      if (!document.getElementById(id)) return;
      lockedTo = id;
      setCurrent(id);
    });
  });
  window.addEventListener(
    'scroll',
    () => {
      if (!lockedTo) return;
      window.clearTimeout(releaseTimer);
      releaseTimer = window.setTimeout(() => {
        lockedTo = null;
      }, 250);
    },
    { passive: true },
  );

  const visible = new Map<string, number>();
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        visible.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0);
      }
      // Pick the section whose top is closest to the header line, among visible ones.
      let best = '';
      let bestDist = Infinity;
      for (const s of sections) {
        if ((visible.get(s.id) ?? 0) <= 0) continue;
        const d = Math.abs(s.getBoundingClientRect().top - headerH);
        if (d < bestDist) {
          bestDist = d;
          best = s.id;
        }
      }
      if (lockedTo) return;
      if (best) setCurrent(best);
      else if (window.scrollY < 200) setCurrent('');
    },
    { rootMargin: `-${headerH + 1}px 0px -55% 0px`, threshold: [0, 0.1, 0.5] },
  );
  sections.forEach((s) => io.observe(s));
}
