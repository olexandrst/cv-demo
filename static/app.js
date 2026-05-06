(() => {
  'use strict';

  // ──────────────────────────────  DOM  ──────────────────────────────
  const video       = document.getElementById('video');
  const overlay     = document.getElementById('overlay');
  const ctx         = overlay.getContext('2d');
  const modeLabel   = document.getElementById('modeLabel');
  const fpsLabel    = document.getElementById('fpsLabel');
  const statusDot   = document.getElementById('statusDot');
  const cards       = document.querySelectorAll('.mode-card');
  const zoneBar     = document.getElementById('zoneToolbar');
  const btnNew      = document.getElementById('zoneNew');
  const btnFinish   = document.getElementById('zoneFinish');
  const btnUndo     = document.getElementById('zoneUndo');
  const btnClear    = document.getElementById('zoneClear');
  const legendBody  = document.getElementById('legendBody');

  // ──────────────────────────────  State  ──────────────────────────────
  const MODE_NAMES = {
    detection: 'Детекція людей',
    zones:     'Зони інтересу',
    ppe:       'Засоби захисту',
    profiling: 'Профілювання',
  };

  const LEGENDS = {
    detection: [
      { c: '#00C8FF', t: 'людина (різні кольори)' },
      { c: '#FFC800', t: 'людина' },
      { c: '#FF64C8', t: 'людина' },
    ],
    zones: [
      { c: '#FFDC00', t: 'спокійна зона' },
      { c: '#FF4D6A', t: 'порушення зони' },
      { c: '#00C8FF', t: 'людина поза зоною' },
    ],
    ppe: [
      { c: '#22C55E', t: 'у шоломі' },
      { c: '#FF4D6A', t: 'без шолома' },
    ],
    profiling: [
      { c: '#A0A0A0', t: 'нормальний' },
      { c: '#22C55E', t: 'веселий' },
      { c: '#3B82F6', t: 'сумний' },
      { c: '#EF4444', t: 'сердитий' },
      { c: '#FFD700', t: 'наляканий' },
      { c: '#FFA500', t: 'здивований' },
      { c: '#A020F0', t: 'роздратований' },
    ],
  };

  let currentMode = 'detection';

  // zones (normalized 0..1 coords)
  let zones = [];                 // [[ [x,y], ... ], ...]
  let drawingPoly = null;         // current polygon being drawn (px on canvas)
  let drawing = false;

  // ──────────────────────────────  Helpers  ──────────────────────────────
  function syncCanvasSize() {
    const r = video.getBoundingClientRect();
    overlay.width  = Math.max(1, r.width  | 0);
    overlay.height = Math.max(1, r.height | 0);
    drawZones();
  }
  window.addEventListener('resize', syncCanvasSize);
  if (video.complete) syncCanvasSize();
  video.addEventListener('load', syncCanvasSize);

  // FPS estimator: count <img> redraws via requestAnimationFrame + onload
  // We use a synthetic approach: increase counter when MJPEG frame arrives.
  // The MJPEG <img> doesn't emit 'load' per frame on most browsers, so use rAF
  // tied to a hidden canvas snapshot diff — simple and good enough for HUD.
  let fpsFrames = 0, fpsLast = performance.now();
  const fpsCanvas = document.createElement('canvas');
  fpsCanvas.width = 32; fpsCanvas.height = 18;
  const fpsCtx = fpsCanvas.getContext('2d', { willReadFrequently: true });
  let prevHash = 0;
  function tickFps() {
    try {
      fpsCtx.drawImage(video, 0, 0, 32, 18);
      const data = fpsCtx.getImageData(0, 0, 32, 18).data;
      let h = 0;
      for (let i = 0; i < data.length; i += 257) h = (h * 31 + data[i]) | 0;
      if (h !== prevHash) { fpsFrames++; prevHash = h; }
    } catch (_) { /* video not yet decodable */ }
    const now = performance.now();
    if (now - fpsLast >= 1000) {
      fpsLabel.textContent = fpsFrames + ' fps';
      fpsFrames = 0;
      fpsLast = now;
    }
    requestAnimationFrame(tickFps);
  }
  requestAnimationFrame(tickFps);

  // ──────────────────────────────  Mode switching  ──────────────────────────────
  cards.forEach(card => {
    card.addEventListener('click', () => setMode(card.dataset.mode));
  });

  function setMode(mode) {
    if (!MODE_NAMES[mode]) return;
    currentMode = mode;
    cards.forEach(c => c.classList.toggle('is-active', c.dataset.mode === mode));
    modeLabel.textContent = MODE_NAMES[mode];
    zoneBar.classList.toggle('hidden', mode !== 'zones');
    overlay.classList.remove('is-drawing');
    drawing = false;
    drawingPoly = null;
    updateZoneButtons();
    renderLegend();
    drawZones();

    fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    }).catch(() => statusDot.classList.replace('chip--ok', 'chip--warn'));
  }

  function renderLegend() {
    const items = LEGENDS[currentMode] || [];
    legendBody.innerHTML = items.map(it =>
      `<div class="legend-row">
         <span class="legend-swatch" style="background:${it.c}"></span>
         <span>${it.t}</span>
       </div>`
    ).join('');
  }
  renderLegend();

  // ──────────────────────────────  Zones  ──────────────────────────────
  function videoDisplayRect() {
    // The video uses object-fit: contain; compute the actual displayed rect.
    const cw = overlay.width, ch = overlay.height;
    const nw = video.naturalWidth || 1280;
    const nh = video.naturalHeight || 720;
    const scale = Math.min(cw / nw, ch / nh);
    const w = nw * scale, h = nh * scale;
    const x = (cw - w) / 2, y = (ch - h) / 2;
    return { x, y, w, h };
  }

  function clientToNorm(clientX, clientY) {
    const r = overlay.getBoundingClientRect();
    const cx = clientX - r.left;
    const cy = clientY - r.top;
    // canvas px -> css px (canvas size matches CSS size, so 1:1)
    const vd = videoDisplayRect();
    if (cx < vd.x || cy < vd.y || cx > vd.x + vd.w || cy > vd.y + vd.h) return null;
    return [(cx - vd.x) / vd.w, (cy - vd.y) / vd.h];
  }

  function normToCanvas(p) {
    const vd = videoDisplayRect();
    return [vd.x + p[0] * vd.w, vd.y + p[1] * vd.h];
  }

  function drawZones() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (currentMode !== 'zones') return;

    // committed zones — yellow outlines (the server overrides with red on alarm
    // in the baked stream, but here we just hint at locations)
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = 'rgba(255,220,0,.7)';
    ctx.fillStyle   = 'rgba(255,220,0,.06)';
    for (const z of zones) {
      ctx.beginPath();
      z.forEach((p, i) => {
        const [x, y] = normToCanvas(p);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // active polygon being drawn
    if (drawingPoly && drawingPoly.length) {
      ctx.strokeStyle = '#38e1ff';
      ctx.fillStyle   = 'rgba(56,225,255,.12)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      drawingPoly.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.stroke();
      // dots
      for (const [x, y] of drawingPoly) {
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#38e1ff';
        ctx.fill();
      }
    }
  }

  function updateZoneButtons() {
    const drawingHas = drawingPoly && drawingPoly.length > 0;
    btnFinish.disabled = !(drawing && drawingPoly && drawingPoly.length >= 3);
    btnUndo.disabled   = !drawingHas;
    btnNew.disabled    = drawing;
  }

  function startDrawing() {
    drawing = true;
    drawingPoly = [];
    overlay.classList.add('is-drawing');
    updateZoneButtons();
    drawZones();
  }

  function finishDrawing() {
    if (!drawing || !drawingPoly || drawingPoly.length < 3) return;
    const norm = drawingPoly.map(([x, y]) => {
      const vd = videoDisplayRect();
      return [(x - vd.x) / vd.w, (y - vd.y) / vd.h];
    });
    zones.push(norm);
    drawing = false;
    drawingPoly = null;
    overlay.classList.remove('is-drawing');
    updateZoneButtons();
    drawZones();
    pushZones();
  }

  function undoPoint() {
    if (drawingPoly && drawingPoly.length) {
      drawingPoly.pop();
      updateZoneButtons();
      drawZones();
    }
  }

  function clearAll() {
    zones = [];
    drawing = false;
    drawingPoly = null;
    overlay.classList.remove('is-drawing');
    updateZoneButtons();
    drawZones();
    pushZones();
  }

  function pushZones() {
    fetch('/api/zones', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zones }),
    }).catch(() => {});
  }

  overlay.addEventListener('click', (e) => {
    if (!drawing) return;
    const r = overlay.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const vd = videoDisplayRect();
    if (x < vd.x || y < vd.y || x > vd.x + vd.w || y > vd.y + vd.h) return;
    drawingPoly.push([x, y]);
    updateZoneButtons();
    drawZones();
  });
  overlay.addEventListener('dblclick', (e) => {
    e.preventDefault();
    finishDrawing();
  });

  btnNew.addEventListener('click', startDrawing);
  btnFinish.addEventListener('click', finishDrawing);
  btnUndo.addEventListener('click', undoPoint);
  btnClear.addEventListener('click', clearAll);

  // keyboard helpers
  window.addEventListener('keydown', (e) => {
    if (currentMode !== 'zones') return;
    if (e.key === 'Enter' && drawing) finishDrawing();
    else if (e.key === 'Escape') {
      drawing = false;
      drawingPoly = null;
      overlay.classList.remove('is-drawing');
      updateZoneButtons();
      drawZones();
    } else if ((e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)) {
      undoPoint();
    }
  });

  // ──────────────────────────────  Boot  ──────────────────────────────
  setMode('detection');
  // Re-sync after first MJPEG frame is decoded so naturalWidth/Height are valid.
  setInterval(() => {
    if (overlay.width !== overlay.clientWidth || overlay.height !== overlay.clientHeight) {
      syncCanvasSize();
    }
  }, 500);
})();
