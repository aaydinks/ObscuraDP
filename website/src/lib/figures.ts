// Client behaviour shared by every Figure: the "View data" toggle, the one-time
// draw-in when a chart scrolls into view, and tooltips on hover and focus.
let done = false;

export function initFigures(): void {
  if (done) return;
  done = true;

  document.querySelectorAll<HTMLButtonElement>('.view-data').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.getAttribute('aria-controls') ?? '');
      if (!target) return;
      const open = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', String(!open));
      target.hidden = open;
      btn.textContent = open ? 'View data' : 'Hide data';
    });
  });

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const charts = document.querySelectorAll<SVGElement>('.chart[data-animate]');
  if (reduced || !('IntersectionObserver' in window)) {
    charts.forEach((c) => c.classList.add('is-visible'));
  } else {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('is-visible');
            io.unobserve(e.target);
          }
        }
      },
      { threshold: 0.25 },
    );
    charts.forEach((c) => io.observe(c));
  }

  const tipEl = document.createElement('div');
  tipEl.className = 'chart-tip';
  tipEl.id = 'chart-tip';
  tipEl.setAttribute('aria-hidden', 'true');
  document.body.appendChild(tipEl);

  function show(target: Element, x?: number, y?: number) {
    const text = target.getAttribute('data-tip');
    if (!text) return;
    tipEl.textContent = text;
    tipEl.classList.add('is-visible');
    let px = x;
    let py = y;
    if (px === undefined || py === undefined) {
      const r = target.getBoundingClientRect();
      px = r.left + r.width / 2;
      py = r.top;
    }
    position(px, py);
  }

  function position(x: number, y: number) {
    const w = tipEl.offsetWidth;
    const h = tipEl.offsetHeight;
    let left = x + 12;
    let top = y - h - 10;
    if (left + w > window.innerWidth - 8) left = x - w - 12;
    if (top < 8) top = y + 16;
    tipEl.style.left = `${Math.max(8, left)}px`;
    tipEl.style.top = `${top}px`;
  }

  function hide() {
    tipEl.classList.remove('is-visible');
  }

  // Hover: any mark with data-tip.
  document.querySelectorAll<Element>('[data-tip]').forEach((el) => {
    el.addEventListener('mouseenter', (ev) => show(el, (ev as MouseEvent).clientX, (ev as MouseEvent).clientY));
    el.addEventListener('mousemove', (ev) => position((ev as MouseEvent).clientX, (ev as MouseEvent).clientY));
    el.addEventListener('mouseleave', hide);
  });

  // Keyboard: the chart itself takes focus; arrow keys step through its marks.
  // Marks are not individually focusable because the SVG is role="img".
  document.querySelectorAll<SVGElement>('.chart[tabindex="0"]').forEach((svg) => {
    const marks = Array.from(svg.querySelectorAll<Element>('[data-tip]'));
    if (!marks.length) return;
    const live = svg.parentElement?.parentElement?.querySelector<HTMLElement>('.figure-live') ?? null;
    let idx = -1;
    const focusMark = (i: number) => {
      idx = (i + marks.length) % marks.length;
      marks.forEach((m) => m.classList.remove('is-focused'));
      const m = marks[idx];
      m.classList.add('is-focused');
      show(m);
      if (live) live.textContent = m.getAttribute('data-tip');
    };
    svg.addEventListener('focus', () => focusMark(idx < 0 ? 0 : idx));
    svg.addEventListener('blur', () => {
      hide();
      marks.forEach((m) => m.classList.remove('is-focused'));
    });
    svg.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        focusMark(idx + 1);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        focusMark(idx - 1);
      } else if (e.key === 'Home') {
        e.preventDefault();
        focusMark(0);
      } else if (e.key === 'End') {
        e.preventDefault();
        focusMark(marks.length - 1);
      }
    });
  });
}
