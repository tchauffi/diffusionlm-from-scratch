/* Replays real masked-diffusion generation trajectories captured from the model.
 * Each frame is a list of [position, token-id] events (id -1 = re-masked). Plain
 * orders only reveal; the "corrected" (predictor-corrector) order also re-masks
 * low-confidence commits and rewrites them mid-generation. Tokens are colored by
 * id (tiktoken-style). */

const PHI = 0.61803398875;

/** Deterministic pastel color from a token id (same token -> same color). */
function tokenColor(id) {
  const hue = ((id * PHI * 360) % 360);
  const sat = 62 + (id * 7) % 18;           // 62–80%
  const light = 66 + (id * 13) % 12;         // 66–78% -> readable with dark text
  return `hsl(${hue.toFixed(0)} ${sat}% ${light}%)`;
}

const el = (id) => document.getElementById(id);
const canvas = el('canvas');

const ORDERS = {
  confidence: { note: 'Surest positions first — decodes roughly <b>front-to-back</b>.' },
  confidence_weighted: { note: 'Confidence-biased but <b>spread</b> across the sequence.' },
  random: { note: 'Scattered fill, ignoring confidence — <b>messier</b> text.' },
  corrected: { note: 'Predictor-corrector: low-confidence commits flicker back to <b>[MASK] and get rewritten</b>.' },
};

const state = {
  data: null, vocab: null,
  order: 'confidence',
  samples: [], sample: 0,
  events: [], len: 0,
  slots: [], curId: [],      // per-position DOM span + current id (-1 = masked)
  lastWord: [],              // id a position held before being re-masked (for the rewrite flash)
  step: 0, revealedCount: 0,
  playing: true, speed: 1, timer: null,
};

/* ---------- rendering ---------- */

function tokenText(id) {            // -> { text, space } with newlines stripped for chips
  const [t, s] = state.vocab[id];
  return { text: t.replace(/\n+/g, ''), space: !!s, nl: t.includes('\n') };
}

function renderSample(s) {
  canvas.innerHTML = '';
  const len = s.len;
  // Replay events to find each position's final id and its widest text, so slots
  // can reserve width once (monospace) and re-masking never reflows the paragraph.
  const finalId = new Array(len).fill(-1);
  const maxw = new Array(len).fill(1);
  s.events.forEach((fr) => fr.forEach(([p, id]) => {
    if (id >= 0) {
      finalId[p] = id;
      const { text, space } = tokenText(id);
      maxw[p] = Math.max(maxw[p], (space ? 1 : 0) + text.length);
    }
  }));

  const slots = new Array(len);
  for (let p = 0; p < len; p++) {
    const fid = finalId[p];
    const info = fid >= 0 ? tokenText(fid) : { text: '', space: false, nl: false };
    const pureNL = fid >= 0 && info.text === '' && info.nl;
    const span = document.createElement('span');
    if (pureNL) {
      canvas.appendChild(document.createElement('br'));
      span.className = 'tok nl';
      canvas.appendChild(span);
    } else {
      span.className = 'tok';
      span.style.minWidth = maxw[p] + 'ch';
      span.textContent = ' ';
      canvas.appendChild(span);
      if (fid >= 0 && info.nl) canvas.appendChild(document.createElement('br'));
    }
    slots[p] = span;
  }
  state.slots = slots;
  state.curId = new Array(len).fill(-1);
  state.lastWord = new Array(len).fill(-1);
}

function loadSample(idx) {
  const s = state.samples[idx];
  state.sample = idx;
  state.events = s.events;
  state.len = s.len;
  state.step = 0;
  state.revealedCount = 0;
  renderSample(s);
  document.querySelectorAll('.sdot').forEach((d, i) => d.classList.toggle('active', i === idx));
  updateReadouts();
}

function applyOp(p, id) {
  const span = state.slots[p];
  const old = state.curId[p];
  if (id < 0) {                                   // re-mask
    if (old >= 0) { state.revealedCount--; state.lastWord[p] = old; }   // remember the word
    state.curId[p] = -1;
    span.classList.remove('revealed', 'corrected', 'just');
    span.style.background = '';
    span.textContent = ' ';
    span.classList.add('remasking');
    span.addEventListener('animationend', () => span.classList.remove('remasking'), { once: true });
    return;
  }
  if (old < 0) state.revealedCount++;
  // A rewrite = a re-fill landing on a DIFFERENT token than before the re-mask
  // (the model changed its mind). Flash those pink; first-time reveals just pop.
  const changed = state.lastWord[p] >= 0 && state.lastWord[p] !== id;
  state.lastWord[p] = -1;
  state.curId[p] = id;
  const { text, space } = tokenText(id);
  span.textContent = (space ? ' ' : '') + text || ' ';
  span.style.background = tokenColor(id);
  span.classList.remove('remasking', 'just', 'corrected');
  span.classList.add('revealed', changed ? 'corrected' : 'just');
  const cls = changed ? 'corrected' : 'just';
  span.addEventListener('animationend', () => span.classList.remove(cls), { once: true });
}

function nextFrame() {
  if (state.step >= state.events.length) { setPlaying(false); return; }
  state.events[state.step].forEach(([p, id]) => applyOp(p, id));
  flashScanline();
  state.step++;
  updateReadouts();
}

function updateReadouts() {
  const total = state.len || 1;
  const m = Math.max(0, total - state.revealedCount) / total;     // masked fraction
  // Model is conditioned on t = sigma^{-1}(masked fraction) (cosine), so on screen
  // masked% = sigma(t). With the corrector, m (and t) tick back up then down.
  const t = (2 / Math.PI) * Math.acos(Math.min(1, Math.max(-1, 1 - m)));
  el('stepOut').textContent = state.step;
  el('tOut').textContent = t.toFixed(2);
  el('maskOut').textContent = Math.round(m * 100) + '%';
  el('trackFill').style.width = ((1 - m) * 100).toFixed(1) + '%';
}

let scanTimer = null;
function flashScanline() {
  const sl = el('scanline');
  sl.style.transition = 'none';
  sl.style.opacity = '1';
  sl.style.top = (10 + Math.random() * 60) + '%';
  clearTimeout(scanTimer);
  scanTimer = setTimeout(() => { sl.style.transition = 'opacity .5s'; sl.style.opacity = '0'; }, 80);
}

/* ---------- loop ---------- */

function tick() {
  if (!state.playing) return;
  nextFrame();
  state.timer = setTimeout(tick, 130 / state.speed);     // ms per frame at speed 1
}

function setPlaying(p) {
  state.playing = p;
  el('playBtn').textContent = p ? '❚❚ Pause' : '▶ Play';
  clearTimeout(state.timer);
  if (p) {
    if (state.step >= state.events.length) restart(false);
    tick();
  }
}

function restart(autoplay = true) {
  clearTimeout(state.timer);
  loadSample(state.sample);
  if (autoplay) setPlaying(true);
}

function buildSampleDots() {
  const dots = el('sampleDots');
  dots.innerHTML = '';
  state.samples.forEach((_, i) => {
    const b = document.createElement('button');
    b.className = 'sdot'; b.textContent = i + 1;
    b.addEventListener('click', () => { state.sample = i; restart(true); });
    dots.appendChild(b);
  });
}

function setOrder(order) {
  if (!state.data.orders || !state.data.orders[order]) return;
  state.order = order;
  state.samples = state.data.orders[order];
  state.sample = 0;
  document.querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.order === order));
  el('orderNote').innerHTML = ORDERS[order] ? ORDERS[order].note : '';
  buildSampleDots();
  restart(true);
}

/* ---------- autoregressive vs diffusion mini-demo ---------- */

const MINI = "The little fox found a warm hole".split(' ');
function setupMiniDemos() {
  const ar = el('arDemo'), diff = el('diffDemo');
  const arSpans = [], diffSpans = [];
  MINI.forEach((w) => {
    const a = document.createElement('span');
    a.className = 'ar-tok'; a.textContent = w; ar.appendChild(a); arSpans.push(a);
    const d = document.createElement('span');
    d.className = 'ar-tok'; d.textContent = w; diff.appendChild(d); diffSpans.push(d);
  });
  const diffOrder = [3, 0, 5, 1, 6, 2, 4];
  const color = (span, i) => { span.style.background = tokenColor((i + 3) * 421); span.classList.add('on'); };
  const reset = (spans) => spans.forEach((s) => { s.classList.remove('on'); s.style.background = ''; });
  const onScreen = (node) => { const r = node.getBoundingClientRect(); return r.top < innerHeight && r.bottom > 0; };
  function loop() {
    reset(arSpans); reset(diffSpans);
    let i = 0;
    const iv = setInterval(() => {
      if (i < MINI.length) { color(arSpans[i], i); color(diffSpans[diffOrder[i]], diffOrder[i]); i++; }
      else { clearInterval(iv); setTimeout(() => { if (onScreen(ar)) loop(); }, 1400); }
    }, 360);
  }
  const io = new IntersectionObserver((e) => { if (e[0].isIntersecting) { loop(); io.disconnect(); } }, { threshold: .4 });
  io.observe(ar);
}

/* ---------- boot ---------- */

async function boot() {
  let data;
  try {
    data = await (await fetch('trajectories.json')).json();
  } catch (e) {
    canvas.textContent = 'Could not load trajectories.json';
    return;
  }
  state.data = data;
  state.vocab = data.vocab;
  el('modelTag').textContent = data.model || '';

  document.querySelectorAll('.seg-btn').forEach((b) => {
    b.addEventListener('click', () => setOrder(b.dataset.order));
    b.style.display = data.orders[b.dataset.order] ? '' : 'none';
  });

  setupMiniDemos();
  el('playBtn').addEventListener('click', () => setPlaying(!state.playing));
  el('restartBtn').addEventListener('click', () => restart(true));
  el('speed').addEventListener('input', (e) => { state.speed = +e.target.value; });

  setOrder('confidence');
}

boot();
