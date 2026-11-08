// Code tabs
(() => {
  const tabs = [...document.querySelectorAll('.tabs [role="tab"]')];
  const select = (tab) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      select(next);
      next.focus();
    });
  });
})();

// Screenshots: drop a file at the figure's data-src path (public/screenshots/...) and
// it replaces the placeholder automatically. No HTML edits needed.
(() => {
  document.querySelectorAll('[data-src]').forEach((fig) => {
    const src = fig.getAttribute('data-src');
    const img = new Image();
    img.decoding = 'async';
    img.alt = fig.querySelector('figcaption strong')?.textContent || '';
    img.onload = () => {
      const ph = fig.querySelector('.shot-ph');
      if (ph) ph.replaceWith(img);
    };
    img.src = src;
  });
})();
