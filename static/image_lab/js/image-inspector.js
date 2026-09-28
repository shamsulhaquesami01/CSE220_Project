/* Result image inspector */
'use strict';
window.createImageInspector = function () {
  const dialog = document.createElement('dialog');
  dialog.className = 'image-inspector';
  dialog.setAttribute('aria-labelledby', 'inspector-title');
  dialog.innerHTML = `
    <div class="inspector-shell">
      <header class="inspector-header">
        <div><p class="eyebrow">IMAGE INSPECTOR</p><h2 id="inspector-title"></h2>
        <p class="inspector-meta" aria-live="polite"></p></div>
        <button type="button" class="ghost inspector-close" aria-label="Close image inspector" autofocus>×</button>
      </header>
      <div class="inspector-toolbar" aria-label="Image view controls">
        <button type="button" class="ghost" data-view="actual">1:1 Actual pixels</button>
        <button type="button" class="ghost" data-view="fit">Fit to screen</button>
        <button type="button" class="ghost" data-zoom="out" aria-label="Zoom out">−</button>
        <output class="inspector-zoom" aria-live="polite">100%</output>
        <button type="button" class="ghost" data-zoom="in" aria-label="Zoom in">+</button>
        <a class="panel-download">Download PNG</a>
      </div>
      <div class="inspector-viewport" tabindex="0" aria-label="Image canvas. Drag or scroll to pan; arrow keys scroll.">
        <div class="inspector-stage"></div>
      </div>
      <footer class="inspector-footer"><span class="inspector-status" role="status"></span>
        <span>1:1 = one image pixel per CSS pixel · Drag or scroll to pan · Esc to close</span></footer>
    </div>`;
  document.body.append(dialog);
  const find = (selector) => dialog.querySelector(selector);
  const viewport = find('.inspector-viewport');
  const stage = find('.inspector-stage');
  const actual = find('[data-view="actual"]');
  const fit = find('[data-view="fit"]');
  const minus = find('[data-zoom="out"]');
  const plus = find('[data-zoom="in"]');
  const status = find('.inspector-status');
  let image, scale = 1, mode = 'actual', ready = false, opener, oldOverflow, drag;

  function render(center = false) {
    if (ready && mode === 'fit') {
      scale = Math.min(1, viewport.clientWidth / image.naturalWidth,
        viewport.clientHeight / image.naturalHeight);
    }
    actual.setAttribute('aria-pressed', String(mode === 'actual'));
    fit.setAttribute('aria-pressed', String(mode === 'fit'));
    actual.disabled = fit.disabled = !ready;
    minus.disabled = !ready || scale <= 0.05;
    plus.disabled = !ready || scale >= 16;
    if (!ready) return;
    image.style.width = `${image.naturalWidth * scale}px`;
    image.style.height = `${image.naturalHeight * scale}px`;
    find('.inspector-zoom').textContent = `${Math.round(scale * 1000) / 10}%`;
    viewport.classList.toggle('is-pannable', image.naturalWidth * scale > viewport.clientWidth ||
      image.naturalHeight * scale > viewport.clientHeight);
    if (center) {
      viewport.scrollLeft = (viewport.scrollWidth - viewport.clientWidth) / 2;
      viewport.scrollTop = (viewport.scrollHeight - viewport.clientHeight) / 2;
    }
  }

  function zoom(factor) {
    if (!ready) return;
    const x = (viewport.scrollLeft + viewport.clientWidth / 2 - image.offsetLeft) / scale;
    const y = (viewport.scrollTop + viewport.clientHeight / 2 - image.offsetTop) / scale;
    scale = Math.max(0.05, Math.min(16, scale * factor));
    mode = 'zoom';
    render();
    viewport.scrollLeft = image.offsetLeft + x * scale - viewport.clientWidth / 2;
    viewport.scrollTop = image.offsetTop + y * scale - viewport.clientHeight / 2;
  }
  actual.onclick = () => { mode = 'actual'; scale = 1; render(true); };
  fit.onclick = () => { mode = 'fit'; render(true); };
  minus.onclick = () => zoom(1 / 1.25);
  plus.onclick = () => zoom(1.25);
  function close() {
    dialog.close();
    cleanup();
  }
  find('.inspector-close').onclick = close;
  dialog.addEventListener('cancel', (event) => { event.preventDefault(); close(); });
  dialog.addEventListener('keydown', (event) => {
    if (event.key !== 'Tab') return;
    const controls = Array.from(dialog.querySelectorAll('button:not(:disabled), a[href], [tabindex="0"]'));
    const first = controls[0], last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault(); last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault(); first.focus();
    }
  });
  let backdropDown = false;
  dialog.addEventListener('pointerdown', (event) => { backdropDown = event.target === dialog; });
  dialog.addEventListener('click', (event) => {
    if (backdropDown && event.target === dialog) close();
    backdropDown = false;
  });
  function cleanup() {
    if (!image) return;
    document.body.style.overflow = oldOverflow;
    ready = false;
    image = null;
    stage.replaceChildren();
    drag = null;
    viewport.classList.remove('is-dragging');
    if (opener?.isConnected) opener.focus({ preventScroll: true });
    else document.querySelector('.panel-inspect')?.focus({ preventScroll: true });
  }
  // Avoid clearing a reopened image.
  dialog.addEventListener('close', () => { if (!dialog.open) cleanup(); });
  viewport.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || !viewport.classList.contains('is-pannable')) return;
    drag = { x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop };
    viewport.setPointerCapture(event.pointerId);
    viewport.classList.add('is-dragging');
    viewport.focus({ preventScroll: true });
    event.preventDefault();
  });
  viewport.addEventListener('pointermove', (event) => {
    if (!drag) return;
    viewport.scrollLeft = drag.left - event.clientX + drag.x;
    viewport.scrollTop = drag.top - event.clientY + drag.y;
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    viewport.addEventListener(type, () => { drag = null; viewport.classList.remove('is-dragging'); });
  }
  new ResizeObserver(() => { if (dialog.open && ready) render(); }).observe(viewport);

  return function open(panel, filename, trigger) {
    if (dialog.open) return;
    opener = trigger;
    mode = 'actual';
    scale = 1;
    ready = false;
    find('#inspector-title').textContent = panel.label;
    find('.inspector-meta').textContent = 'Loading exported image…';
    find('.inspector-zoom').textContent = '100%';
    const download = find('.panel-download');
    download.href = panel.download_url || panel.url;
    download.download = filename;
    download.setAttribute('aria-label', `Download ${panel.label} as PNG`);
    status.textContent = 'Loading…';
    viewport.classList.remove('is-pannable');
    const next = new Image();
    image = next;
    next.alt = panel.label;
    next.draggable = false;
    next.hidden = true;
    next.onload = () => {
      if (image !== next || !dialog.open) return;
      ready = true;
      next.hidden = false;
      find('.inspector-meta').textContent = `${next.naturalWidth} × ${next.naturalHeight} px · PNG · ${filename}`;
      status.textContent = 'Viewing the downloadable image';
      render(true);
    };
    next.onerror = () => {
      if (image !== next || !dialog.open) return;
      find('.inspector-meta').textContent = 'Image unavailable';
      status.textContent = 'Could not load the image. Close and reopen to retry, or use Download PNG.';
    };
    stage.replaceChildren(next);
    oldOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.showModal();
    render();
    next.src = download.href;
  };
};
