/* Image Lab frontend.
 *
 * Plain ES2020, no framework and no bundler. The whole job is:
 *   1. upload an image and remember its id,
 *   2. collect the current operation's parameters,
 *   3. POST them, and swap in the returned panels.
 *
 * Every control funnels into `scheduleRun`, which debounces so dragging a
 * slider does not fire one request per pixel of travel.
 */
'use strict';

(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const PRESETS = JSON.parse($('#kernel-presets').textContent);
  const CSRF = $('#csrf-holder input[name="csrfmiddlewaretoken"]').value;

  const state = {
    imageId: null,
    op: 'convolve',
    kernel: null,       // 2D array of numbers
    inFlight: null,     // AbortController for the current request
    timer: null,
  };

  // ---------------------------------------------------------------------
  // Kernel editor
  // ---------------------------------------------------------------------

  // Build an n-by-n identity kernel (centre 1, everything else 0).
  function identityKernel(n) {
    const k = Array.from({ length: n }, () => new Array(n).fill(0));
    k[(n - 1) >> 1][(n - 1) >> 1] = 1;
    return k;
  }

  // Rebuild the input grid from state.kernel.
  function renderKernel() {
    const grid = $('#kernel-grid');
    const n = state.kernel.length;
    const cols = state.kernel[0].length;
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    grid.replaceChildren();

    const cy = (n - 1) >> 1;
    const cx = (cols - 1) >> 1;

    state.kernel.forEach((row, y) => {
      row.forEach((value, x) => {
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.value = formatCell(value);
        input.setAttribute('aria-label', `kernel row ${y + 1} column ${x + 1}`);
        if (y === cy && x === cx) input.classList.add('is-center');

        input.addEventListener('input', () => {
          const parsed = parseFloat(input.value);
          state.kernel[y][x] = Number.isFinite(parsed) ? parsed : 0;
          updateKernelSum();
          // Editing a cell means it is no longer one of the named presets.
          $('#kernel-preset').value = '';
          scheduleRun();
        });

        grid.appendChild(input);
      });
    });
    updateKernelSum();
  }

  // Trim float noise so 1/9 shows as 0.1111 rather than 0.11111111111111.
  function formatCell(value) {
    if (Number.isInteger(value)) return String(value);
    return String(Math.round(value * 10000) / 10000);
  }

  function updateKernelSum() {
    const total = state.kernel.flat().reduce((a, b) => a + b, 0);
    $('#kernel-sum').textContent = total.toFixed(4);
  }

  // Resize the kernel grid, keeping any overlapping coefficients.
  function resizeKernel(n) {
    const next = identityKernel(n);
    const old = state.kernel;
    if (old) {
      // Align the two grids on their centres so a 3x3 blur stays centred
      // when promoted to 5x5, instead of drifting into the corner.
      const offset = ((n - old.length) / 2) | 0;
      for (let y = 0; y < old.length; y++) {
        for (let x = 0; x < old[y].length; x++) {
          const ty = y + offset;
          const tx = x + offset;
          if (ty >= 0 && ty < n && tx >= 0 && tx < n) next[ty][tx] = old[y][x];
        }
      }
    }
    state.kernel = next;
    renderKernel();
  }

  // ---------------------------------------------------------------------
  // Parameter collection
  // ---------------------------------------------------------------------

  function currentParams() {
    if (state.op === 'convolve') {
      return {
        kernel: state.kernel,
        normalize: $('#conv-normalize').checked,
        pad_mode: $('#conv-pad').value,
      };
    }
    if (state.op === 'resample') {
      return {
        scale: parseFloat($('#resize-scale').value),
        method: $('#resize-method').value,
        pad_mode: $('#resize-pad').value,
      };
    }
    if (state.op === 'noise') {
      return {
        noise_model: $('#noise-model').value,
        noise_sigma: parseFloat($('#noise-sigma').value),
        noise_amount: parseFloat($('#noise-amount').value),
        clean_filter: $('#clean-filter').value,
        filter_size: parseInt($('#filter-size').value, 10),
        filter_sigma: parseFloat($('#filter-sigma').value),
        seed: parseInt($('#noise-seed').value, 10) || 0,
        pad_mode: $('#noise-pad').value,
      };
    }
    return {};
  }

  // ---------------------------------------------------------------------
  // Networking
  // ---------------------------------------------------------------------

  function scheduleRun(delay = 180) {
    clearTimeout(state.timer);
    state.timer = setTimeout(run, delay);
  }

  async function run() {
    if (!state.imageId) return;

    // Cancel whatever is still in the air; only the newest result matters.
    if (state.inFlight) state.inFlight.abort();
    const controller = new AbortController();
    state.inFlight = controller;

    $('#spinner').hidden = false;
    hideError();

    try {
      const response = await fetch('/api/process/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF },
        body: JSON.stringify({ image_id: state.imageId, op: state.op, params: currentParams() }),
        signal: controller.signal,
      });

      const data = await response.json();
      if (!response.ok) {
        showError(data.error || `Request failed (${response.status}).`);
        return;
      }
      renderResults(data);
    } catch (err) {
      if (err.name !== 'AbortError') showError('Could not reach the server.');
    } finally {
      if (state.inFlight === controller) {
        state.inFlight = null;
        $('#spinner').hidden = true;
      }
    }
  }

  async function uploadFile(file) {
    const status = $('#upload-status');
    status.textContent = 'Uploading…';
    status.className = 'status';
    hideError();

    const body = new FormData();
    body.append('image', file);
    if ($('#opt-grayscale').checked) body.append('grayscale', '1');

    try {
      const response = await fetch('/api/upload/', {
        method: 'POST',
        headers: { 'X-CSRFToken': CSRF },
        body,
      });
      const data = await response.json();

      if (!response.ok) {
        status.textContent = data.error || 'Upload failed.';
        status.className = 'status is-error';
        return;
      }

      state.imageId = data.image_id;
      status.textContent = `Loaded ${data.width} × ${data.height}, ${data.channels === 1 ? 'grayscale' : 'RGB'}.`;
      status.className = 'status is-ok';

      $('#empty-state').hidden = true;
      $('#op-card').hidden = false;
      showOpPanel(state.op);
      run();
    } catch (err) {
      status.textContent = 'Could not reach the server.';
      status.className = 'status is-error';
    }
  }

  // ---------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------

  function renderResults(data) {
    const grid = $('#panel-grid');
    grid.replaceChildren();

    for (const panel of data.panels) {
      const card = document.createElement('div');
      card.className = 'panel';

      const head = document.createElement('div');
      head.className = 'panel-head';
      const title = document.createElement('strong');
      title.textContent = panel.label;
      const dims = document.createElement('span');
      dims.textContent = `${panel.width} × ${panel.height}`;
      head.append(title, dims);

      const figure = document.createElement('figure');
      const img = document.createElement('img');
      img.src = panel.url;
      img.alt = panel.label;
      img.loading = 'lazy';
      figure.appendChild(img);

      if (panel.caption) {
        const caption = document.createElement('figcaption');
        // Captions are server-generated and may carry entities like &times;.
        caption.innerHTML = panel.caption;
        figure.appendChild(caption);
      }

      card.append(head, figure);
      grid.appendChild(card);
    }

    const tbody = $('#metrics-table').querySelector('tbody');
    tbody.replaceChildren();
    for (const metric of data.metrics) {
      const tr = document.createElement('tr');
      const tdLabel = document.createElement('td');
      tdLabel.innerHTML = metric.label;
      const tdValue = document.createElement('td');
      tdValue.innerHTML = metric.value;
      if (metric.hint) {
        const hint = document.createElement('span');
        hint.className = 'hint';
        hint.innerHTML = metric.hint;
        tdValue.appendChild(hint);
      }
      tr.append(tdLabel, tdValue);
      tbody.appendChild(tr);
    }
    $('#metrics-box').hidden = data.metrics.length === 0;

    const notes = $('#notes-box');
    notes.replaceChildren();
    for (const note of data.notes) {
      const p = document.createElement('p');
      p.innerHTML = note;
      notes.appendChild(p);
    }
    notes.hidden = data.notes.length === 0;
  }

  function showError(message) {
    const box = $('#error-box');
    box.textContent = message;
    box.hidden = false;
  }

  function hideError() { $('#error-box').hidden = true; }

  function showOpPanel(opId) {
    $$('.op-panel').forEach((panel) => {
      panel.hidden = panel.dataset.opPanel !== opId;
    });
    $$('[data-op-desc]').forEach((el) => {
      el.hidden = el.dataset.opDesc !== opId;
    });
  }

  // Show only the parameters that apply to the selected noise model / filter.
  function syncNoiseVisibility() {
    const model = $('#noise-model').value;
    $$('[data-noise-param]').forEach((el) => {
      el.hidden = el.dataset.noiseParam !== model;
    });
    const filter = $('#clean-filter').value;
    $('[data-filter-param="window"]').hidden = !(filter === 'mean' || filter === 'median');
    $('[data-filter-param="sigma"]').hidden = filter !== 'gaussian';
  }

  // ---------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------

  function init() {
    state.kernel = identityKernel(3);
    renderKernel();

    // --- upload ---
    const dropzone = $('#dropzone');
    const fileInput = $('#file-input');

    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) uploadFile(fileInput.files[0]);
    });

    ['dragenter', 'dragover'].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add('is-over');
      })
    );
    ['dragleave', 'drop'].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove('is-over');
      })
    );
    dropzone.addEventListener('drop', (e) => {
      const file = e.dataTransfer?.files?.[0];
      if (file) uploadFile(file);
    });

    $('#opt-grayscale').addEventListener('change', () => {
      // Colour mode is decided at ingest, so re-send the same file.
      if (fileInput.files[0]) uploadFile(fileInput.files[0]);
    });

    // --- operation tabs ---
    $$('.tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        $$('.tab').forEach((t) => t.classList.remove('is-active'));
        tab.classList.add('is-active');
        state.op = tab.dataset.op;
        showOpPanel(state.op);
        run();
      });
    });

    // --- kernel controls ---
    $('#kernel-preset').addEventListener('change', (e) => {
      const preset = PRESETS[e.target.value];
      if (!preset) return;
      state.kernel = preset.map((row) => row.slice());
      $('#kernel-size').value = String(preset.length);
      renderKernel();
      scheduleRun(0);
    });

    $('#kernel-size').addEventListener('change', (e) => {
      resizeKernel(parseInt(e.target.value, 10));
      $('#kernel-preset').value = '';
      scheduleRun(0);
    });

    $('#kernel-normalize-now').addEventListener('click', () => {
      const total = state.kernel.flat().reduce((a, b) => a + b, 0);
      if (Math.abs(total) < 1e-12) return;   // zero-sum kernels must not be scaled
      state.kernel = state.kernel.map((row) => row.map((v) => v / total));
      renderKernel();
      scheduleRun(0);
    });

    $('#kernel-clear').addEventListener('click', () => {
      state.kernel = state.kernel.map((row) => row.map(() => 0));
      renderKernel();
      scheduleRun(0);
    });

    $('#conv-normalize').addEventListener('change', () => scheduleRun(0));
    $('#conv-pad').addEventListener('change', () => scheduleRun(0));

    // --- resample controls ---
    $('#resize-scale').addEventListener('input', (e) => {
      $('#resize-scale-out').textContent = `${parseFloat(e.target.value).toFixed(2)}×`;
      scheduleRun();
    });
    $('#resize-method').addEventListener('change', () => scheduleRun(0));
    $('#resize-pad').addEventListener('change', () => scheduleRun(0));

    // --- noise controls ---
    $('#noise-model').addEventListener('change', () => { syncNoiseVisibility(); scheduleRun(0); });
    $('#clean-filter').addEventListener('change', () => { syncNoiseVisibility(); scheduleRun(0); });

    $('#noise-sigma').addEventListener('input', (e) => {
      $('#noise-sigma-out').textContent = parseFloat(e.target.value).toFixed(3);
      scheduleRun();
    });
    $('#noise-amount').addEventListener('input', (e) => {
      $('#noise-amount-out').textContent = `${(parseFloat(e.target.value) * 100).toFixed(1)}%`;
      scheduleRun();
    });
    $('#filter-size').addEventListener('input', (e) => {
      $('#filter-size-out').textContent = e.target.value;
      scheduleRun();
    });
    $('#filter-sigma').addEventListener('input', (e) => {
      $('#filter-sigma-out').textContent = parseFloat(e.target.value).toFixed(1);
      scheduleRun();
    });
    $('#noise-seed').addEventListener('change', () => scheduleRun(0));
    $('#noise-pad').addEventListener('change', () => scheduleRun(0));

    syncNoiseVisibility();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
