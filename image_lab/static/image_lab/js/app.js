/* Main frontend logic */
'use strict';

(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const inspectImage = window.createImageInspector();

  const PRESETS = JSON.parse($('#kernel-presets').textContent);
  const CSRF = $('#csrf-holder input[name="csrfmiddlewaretoken"]').value;

  const state = {
    imageId: null,
    queryImageId: null,
    op: 'convolve',
    kernel: null,
    inFlight: null,
    timer: null,
  };

  function identityKernel(n) {
    const k = Array.from({ length: n }, () => new Array(n).fill(0));
    k[(n - 1) >> 1][(n - 1) >> 1] = 1;
    return k;
  }

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
          $('#kernel-preset').value = '';
          scheduleRun();
        });

        grid.appendChild(input);
      });
    });
    updateKernelSum();
  }

  function formatCell(value) {
    if (Number.isInteger(value)) return String(value);
    return String(Math.round(value * 10000) / 10000);
  }

  function updateKernelSum() {
    const total = state.kernel.flat().reduce((a, b) => a + b, 0);
    $('#kernel-sum').textContent = total.toFixed(4);
  }

  function resizeKernel(n) {
    const next = identityKernel(n);
    const old = state.kernel;
    if (old) {
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
    if (state.op === 'deblur') {
      return {
        input_mode: $('#deblur-input-mode').value,
        motion_length: parseFloat($('#deblur-length').value),
        motion_angle: parseFloat($('#deblur-angle').value),
        noise_sigma: parseFloat($('#deblur-noise').value),
        wiener_log10: parseFloat($('#deblur-wiener').value),
        inverse_floor_log10: parseFloat($('#deblur-floor').value),
        seed: parseInt($('#deblur-seed').value, 10) || 0,
      };
    }
    if (state.op === 'spectral_match') {
      return {
        input_mode: $('#match-input-mode').value,
        query_image_id: state.queryImageId,
        rotation: parseFloat($('#match-rotation').value),
        scale: parseFloat($('#match-scale').value),
        shift_x_fraction: parseFloat($('#match-shift-x').value),
        shift_y_fraction: parseFloat($('#match-shift-y').value),
        use_hann: $('#match-hann').checked,
        subpixel: $('#match-subpixel').checked,
      };
    }
    return {};
  }

  function scheduleRun(delay = 180) {
    clearTimeout(state.timer);
    state.timer = setTimeout(run, delay);
  }

  async function run() {
    if (!state.imageId) return;
    if (
      state.op === 'spectral_match'
      && $('#match-input-mode').value === 'real'
      && !state.queryImageId
    ) {
      showError('Upload a query image to run two-image Spectral Match.');
      return;
    }

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

  async function uploadQueryFile(file) {
    const status = $('#match-query-status');
    status.textContent = 'Uploading query…';
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
        status.textContent = data.error || 'Query upload failed.';
        status.className = 'status is-error';
        return;
      }

      state.queryImageId = data.image_id;
      status.textContent = 'Query loaded ' + data.width + ' × ' + data.height + ', ' + (data.channels === 1 ? 'grayscale' : 'RGB') + '.';
      status.className = 'status is-ok';
      if (state.op === 'spectral_match' && $('#match-input-mode').value === 'real') run();
    } catch (err) {
      status.textContent = 'Could not reach the server.';
      status.className = 'status is-error';
    }
  }
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

      const actions = document.createElement('div');
      actions.className = 'panel-actions';
      const dims = document.createElement('span');
      dims.textContent = `${panel.download_width ?? panel.width} × ${panel.download_height ?? panel.height}`;
      dims.title = 'Downloaded image dimensions';

      const download = document.createElement('a');
      download.className = 'panel-download';
      download.href = panel.download_url || panel.url;
      download.download = `image-lab-${data.op}-${panel.key}.png`;
      download.textContent = 'Download PNG';
      download.setAttribute('aria-label', `Download ${panel.label} as PNG`);

      actions.append(dims, download);
      head.append(title, actions);

      const figure = document.createElement('figure');
      const img = document.createElement('img');
      img.src = panel.url;
      img.alt = panel.label;
      img.loading = 'lazy';
      const inspect = document.createElement('button');
      inspect.type = 'button';
      inspect.className = 'panel-inspect';
      inspect.setAttribute('aria-label', `View ${panel.label} at actual size`);
      inspect.setAttribute('aria-haspopup', 'dialog');
      const affordance = document.createElement('span');
      affordance.className = 'panel-inspect-hint';
      affordance.textContent = '⛶ View actual size';
      affordance.setAttribute('aria-hidden', 'true');
      inspect.append(img, affordance);
      inspect.addEventListener('click', () => inspectImage(panel, download.download, inspect));
      figure.appendChild(inspect);

      if (panel.caption) {
        const caption = document.createElement('figcaption');
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

  function syncNoiseVisibility() {
    const model = $('#noise-model').value;
    $$('[data-noise-param]').forEach((el) => {
      el.hidden = el.dataset.noiseParam !== model;
    });
    const filter = $('#clean-filter').value;
    $('[data-filter-param="window"]').hidden = !(filter === 'mean' || filter === 'median');
    $('[data-filter-param="sigma"]').hidden = filter !== 'gaussian';
  }

  function scientificFromLog(value) {
    return (10 ** parseFloat(value)).toExponential(1);
  }

  function syncDeblurVisibility() {
    const simulating = $('#deblur-input-mode').value === 'simulate';
    $$('[data-deblur-sim]').forEach((el) => { el.hidden = !simulating; });
    $$('[data-deblur-copy]').forEach((el) => {
      el.hidden = el.dataset.deblurCopy !== $('#deblur-input-mode').value;
    });
  }

  function syncMatchVisibility() {
    const real = $('#match-input-mode').value === 'real';
    document.querySelectorAll('[data-match-real]').forEach((el) => { el.hidden = !real; });
    document.querySelectorAll('[data-match-controlled]').forEach((el) => { el.hidden = real; });
    document.querySelectorAll('[data-match-copy]').forEach((el) => {
      el.hidden = el.dataset.matchCopy !== $('#match-input-mode').value;
    });
  }

  function signedPercent(value) {
    const percent = parseFloat(value) * 100;
    return (percent >= 0 ? '+' : '') + percent.toFixed(0) + '%';
  }
  function init() {
    state.kernel = identityKernel(3);
    renderKernel();

    const dropzone = $('#dropzone');
    const fileInput = $('#file-input');
    const queryDropzone = $('#match-query-dropzone');
    const queryFileInput = $('#match-query-file');

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

    queryFileInput.addEventListener('change', () => {
      if (queryFileInput.files[0]) uploadQueryFile(queryFileInput.files[0]);
    });
    ['dragenter', 'dragover'].forEach((evt) =>
      queryDropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        queryDropzone.classList.add('is-over');
      })
    );
    ['dragleave', 'drop'].forEach((evt) =>
      queryDropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        queryDropzone.classList.remove('is-over');
      })
    );
    queryDropzone.addEventListener('drop', (e) => {
      const file = e.dataTransfer?.files?.[0];
      if (file) uploadQueryFile(file);
    });

    $('#opt-grayscale').addEventListener('change', () => {
      if (fileInput.files[0]) uploadFile(fileInput.files[0]);
    });

    $$('.tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        $$('.tab').forEach((t) => t.classList.remove('is-active'));
        tab.classList.add('is-active');
        state.op = tab.dataset.op;
        showOpPanel(state.op);
        run();
      });
    });

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
      if (Math.abs(total) < 1e-12) return;
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

    $('#resize-scale').addEventListener('input', (e) => {
      $('#resize-scale-out').textContent = `${parseFloat(e.target.value).toFixed(2)}×`;
      scheduleRun();
    });
    $('#resize-method').addEventListener('change', () => scheduleRun(0));
    $('#resize-pad').addEventListener('change', () => scheduleRun(0));

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

    $('#deblur-input-mode').addEventListener('change', () => {
      syncDeblurVisibility();
      scheduleRun(0);
    });
    $('#deblur-length').addEventListener('input', (e) => {
      $('#deblur-length-out').textContent = `${parseFloat(e.target.value).toFixed(0)} px`;
      scheduleRun();
    });
    $('#deblur-angle').addEventListener('input', (e) => {
      $('#deblur-angle-out').innerHTML = `${parseFloat(e.target.value).toFixed(0)}&deg;`;
      scheduleRun();
    });
    $('#deblur-noise').addEventListener('input', (e) => {
      $('#deblur-noise-out').textContent = parseFloat(e.target.value).toFixed(3);
      scheduleRun();
    });
    $('#deblur-wiener').addEventListener('input', (e) => {
      $('#deblur-wiener-out').textContent = scientificFromLog(e.target.value);
      scheduleRun();
    });
    $('#deblur-floor').addEventListener('input', (e) => {
      $('#deblur-floor-out').textContent = scientificFromLog(e.target.value);
      scheduleRun();
    });
    $('#deblur-seed').addEventListener('change', () => scheduleRun(0));

    $('#match-input-mode').addEventListener('change', () => {
      syncMatchVisibility();
      if ($('#match-input-mode').value === 'controlled' || state.queryImageId) run();
      else showError('Upload a query image to run two-image Spectral Match.');
    });
    $('#match-rotation').addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      $('#match-rotation-out').innerHTML = (value >= 0 ? '+' : '') + value.toFixed(0) + '&deg;';
      scheduleRun();
    });
    $('#match-scale').addEventListener('input', (e) => {
      $('#match-scale-out').textContent = parseFloat(e.target.value).toFixed(2) + '×';
      scheduleRun();
    });
    $('#match-shift-x').addEventListener('input', (e) => {
      $('#match-shift-x-out').textContent = signedPercent(e.target.value);
      scheduleRun();
    });
    $('#match-shift-y').addEventListener('input', (e) => {
      $('#match-shift-y-out').textContent = signedPercent(e.target.value);
      scheduleRun();
    });
    $('#match-hann').addEventListener('change', () => scheduleRun(0));
    $('#match-subpixel').addEventListener('change', () => scheduleRun(0));

    syncNoiseVisibility();
    syncDeblurVisibility();
    syncMatchVisibility();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
