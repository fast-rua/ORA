'use strict';
/* ORA run dashboard: the single-page run browser. Ported from Shaoxiong Duan's code-vs-tools dashboard
   (snapshot 2026-10-31). All URLs are relative, so the page also works behind a proxy at a sub-path. */

const $ = (s, r) => (r || document).querySelector(s);
const esc = s => s == null ? '' : String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmtS = s => s == null || isNaN(s) ? '–' : s < 90 ? s.toFixed(0) + 's' : s < 5400 ? (s / 60).toFixed(1) + 'm' : (s / 3600).toFixed(1) + 'h';
const fmtN = n => n == null ? '–' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e4 ? (n / 1e3).toFixed(1) + 'k' : String(Math.round(n * 100) / 100);
const fileUrl = p => 'api/file?path=' + encodeURIComponent(p);
const qsOf = o => Object.entries(o).filter(([, v]) => v != null && v !== '').map(([k, v]) => k + '=' + encodeURIComponent(v)).join('&');
const utcMs = s => { if (!s) return null; const t = Date.parse(s); return isNaN(t) ? null : t; };
const hhmm = s => { const t = typeof s === 'number' ? s * 1000 : utcMs(s); if (t == null) return '–'; return new Date(t).toISOString().slice(5, 19).replace('T', ' '); };
const median = xs => { xs = xs.filter(x => typeof x === 'number').sort((a, b) => a - b); if (!xs.length) return null; const m = xs.length >> 1; return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2; };
const STATES = ['running', 'done', 'failed', 'fault'];
/* Each condition of a cohort gets a letter (A, B, ...) for the matrix; the legend names them. */
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
function condLetters(conds) { const m = {}; conds.forEach((c, i) => { m[c] = LETTERS[i] || String(i + 1); }); return m; }
function stateMark(it) { return !it ? '' : it.verdict === 'success' ? '✓' : it.verdict === 'late' ? '~' : it.verdict === 'fail' ? '✗' : it.state === 'running' ? '…' : it.state === 'fault' ? '!' : ''; }

const S = {open: {}, tab: {}, sort: {key: 'started_utc', dir: -1}, filt: {q: '', state: '', cond: ''}, last: {}, view: null, sel: {}};

function det(key, def) { const v = S.open[key]; return (v === undefined ? def : v) ? 'open' : ''; }
document.addEventListener('toggle', e => { const d = e.target; if (d && d.dataset && d.dataset.k) S.open[d.dataset.k] = d.open; }, true);

async function getJSON(path) {
  const r = await fetch('api/' + path, {cache: 'no-store'});
  const t = await r.text();
  let j; try { j = JSON.parse(t); } catch (e) { throw new Error('bad response from ' + path + ': ' + t.slice(0, 200)); }
  if (!r.ok) throw new Error(j.error || r.status);
  return [t.replace(/"now":[0-9.e+]+/g, ''), j];
}
async function getText(path, opts) {
  const r = await fetch(fileUrl(path) + (opts || ''), {cache: 'no-cache'});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.text();
}
function setUpdated(err) {
  $('#updated').innerHTML = err ? '<span class="err">' + esc(err) + '</span>' : 'updated ' + new Date().toLocaleTimeString();
}
function keepScroll(fn) { const y = window.scrollY; fn(); window.scrollTo(0, y); }
/* Replace only changed children: items [{key, html}] */
function patchList(container, items) {
  const existing = new Map();
  for (const el of Array.from(container.children)) existing.set(el.dataset.key, el);
  let prev = null;
  const keep = new Set();
  for (const it of items) {
    keep.add(it.key);
    let el = existing.get(it.key);
    if (!el || el._html !== it.html) {
      const tmp = document.createElement('div'); tmp.innerHTML = it.html;
      const nel = tmp.firstElementChild; nel.dataset.key = it.key; nel._html = it.html;
      if (el) el.replaceWith(nel); else if (prev) prev.after(nel); else container.prepend(nel);
      el = nel;
    }
    prev = el;
  }
  for (const [k, el] of existing) if (!keep.has(k)) el.remove();
}

function badge(state, verdict) {
  if (state === 'done' && verdict) return `<span class="badge s-${verdict === 'success' || verdict === 'late' ? verdict : 'fail'}">${esc(verdict)}</span>`;
  return `<span class="badge s-${esc(state || 'queued')}">${esc(state || '–')}</span>`;
}
function cellClass(it) {
  if (!it) return 'c-none';
  if (it.state === 'done') return it.verdict === 'success' ? 'c-success' : it.verdict === 'late' ? 'c-late' : it.verdict === 'fail' ? 'c-fail' : 'c-done';
  if (it.state === 'failed') return it.verdict === 'success' ? 'c-success' : 'c-fail';
  return 'c-' + (it.state || 'queued');
}
function progressBar(counts, total) {
  if (!total) return '<div class="bar"></div>';
  const segs = ['done', 'failed', 'fault', 'running'];
  return '<div class="bar" data-tip="' + esc(Object.entries(counts).map(([k, v]) => k + ': ' + v).join('\n')) + '">' +
    segs.map(s => counts[s] ? `<span class="bg-${s}" style="width:${100 * counts[s] / total}%"></span>` : '').join('') + '</div>';
}
function countChips(counts) {
  return STATES.filter(s => counts[s]).map(s => `<span class="badge s-${s}">${s} ${counts[s]}</span>`).join(' ') +
    Object.keys(counts).filter(s => !STATES.includes(s)).map(s => ` <span class="badge">${esc(s)} ${counts[s]}</span>`).join('');
}
function condScore(c) { return c ? `<b>${c.success}</b>/${c.n}` + (c.episodes > c.n ? ` <span class="muted small">(${c.episodes} runs)</span>` : '') : '–'; }
function condScores(conds) { return Object.entries(conds || {}).map(([k, c]) => `<span>${esc(k)} ${condScore(c)}</span>`).join(' · ') || '–'; }
function runHref(runId) { return '#/run/' + encodeURIComponent(runId); }
function cohortHref(exp, name) { return '#/cohort?' + qsOf({experiment: exp, name}); }
const ALL = '*';  // the pseudo-cohort holding every run of an experiment
const cohortName = n => n === ALL ? 'all cohorts' : n;

/* ----------------------------------------------------------------- tooltip & lightbox */
document.addEventListener('mouseover', e => {
  const el = e.target.closest && e.target.closest('[data-tip]'); const tip = $('#tip');
  if (!el) { tip.hidden = true; return; }
  tip.textContent = el.dataset.tip; tip.hidden = false;
});
document.addEventListener('mousemove', e => {
  const tip = $('#tip'); if (tip.hidden) return;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8), y = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
});
const LB = {list: [], i: 0};
function lbShow() {
  const it = LB.list[LB.i]; if (!it) return;
  $('#lb-img').src = it.src; $('#lb-cap').textContent = it.cap + `  (${LB.i + 1}/${LB.list.length})`; $('#lb-open').href = it.src;
  $('#lightbox').hidden = false;
}
document.addEventListener('click', e => {
  const img = e.target.closest && e.target.closest('img[data-full]');
  if (img) {
    const scope = img.closest('[data-gal]') || document;
    const imgs = Array.from(scope.querySelectorAll('img[data-full]'));
    LB.list = imgs.map(x => ({src: x.dataset.full, cap: x.dataset.cap || ''})); LB.i = imgs.indexOf(img); lbShow(); return;
  }
  const lb = e.target.closest && e.target.closest('#lightbox');
  if (lb && (e.target.id === 'lb-close' || e.target.id === 'lightbox')) { $('#lightbox').hidden = true; }
  if (e.target.id === 'lb-prev') { LB.i = (LB.i - 1 + LB.list.length) % LB.list.length; lbShow(); }
  if (e.target.id === 'lb-next') { LB.i = (LB.i + 1) % LB.list.length; lbShow(); }
});
document.addEventListener('keydown', e => {
  if ($('#lightbox').hidden) return;
  if (e.key === 'Escape') $('#lightbox').hidden = true;
  if (e.key === 'ArrowLeft') { LB.i = (LB.i - 1 + LB.list.length) % LB.list.length; lbShow(); }
  if (e.key === 'ArrowRight') { LB.i = (LB.i + 1) % LB.list.length; lbShow(); }
});
function thumb(src, cap, cls) {
  return `<figure><img loading="lazy" src="${esc(src)}" data-full="${esc(src)}" data-cap="${esc(cap)}" alt="${esc(cap)}"${cls ? ' class="' + cls + '"' : ''}><figcaption>${esc(cap.split(' · ').pop())}</figcaption></figure>`;
}

/* ----------------------------------------------------------------- overview */
const Overview = {
  live: true,
  mount() { $('#crumb').textContent = ''; $('#app').innerHTML = '<div id="ov"></div>'; },
  async load() {
    const [t, o] = await getJSON('overview' + (S.showHidden ? '?hidden=1' : ''));
    if (S.last.ov === t) return; S.last.ov = t;
    let h = `<h1>Overview</h1><p class="muted small">outputs <span class="mono">${esc(o.outputs)}</span>${o.experiment ? ' · experiment ' + esc(o.experiment) : ''}
      · condition: ${o.condition_path ? 'plan <span class="mono">' + esc(o.condition_path) + '</span>' : 'code mode (code) or tool calls (tools)'}
      · ${o.experiments.map(e => esc(e.name) + ' (' + e.runs + ' runs)').join(', ') || 'no experiments with runs/'}</p>`;
    const hideToggle = o.hidden && o.hidden.cohorts ? `<label class="small muted"><input type="checkbox" ${S.showHidden ? 'checked' : ''} onchange="S.showHidden = this.checked; Overview.load()"> show ${o.hidden.cohorts} hidden cohorts (${o.hidden.runs} runs)</label>` : '';
    h += '<section class="panel"><h2>Cohorts</h2>' + hideToggle + '<div class="scroll"><table><tr><th>cohort</th><th>backends</th><th>last start</th><th>progress</th><th>states</th><th>success by condition</th></tr>';
    for (const b of o.cohorts) {
      h += `<tr class="click" onclick="location.hash='${esc(cohortHref(b.experiment, b.name))}'"><td><a href="${esc(cohortHref(b.experiment, b.name))}"><b>${esc(cohortName(b.name))}</b></a>
        <div class="muted small">${esc(b.experiment)} · ${b.tasks.length} tasks × ${b.columns.length} seeds/layouts</div></td>
        <td class="small">${esc(b.backends.join(', '))}</td><td class="small">${esc(hhmm(b.last_utc))}</td>
        <td style="min-width:140px">${progressBar(b.counts, b.total)}<div class="small muted">${b.finished}/${b.total} finished</div></td>
        <td>${countChips(b.counts)}</td><td class="small">${condScores(b.conditions)}</td></tr>`;
    }
    if (!o.cohorts.length) h += '<tr><td colspan="6" class="empty">no runs under this outputs root</td></tr>';
    h += '</table></div></section><section class="panel"><h2>Recent runs</h2>' + runsTable(o.recent, true) + '</section>';
    keepScroll(() => { $('#ov').innerHTML = h; });
  },
};
function runsTable(rows, withCohort) {
  if (!rows || !rows.length) return '<div class="empty">no runs</div>';
  return '<div class="scroll"><table><tr><th>started</th>' + (withCohort ? '<th>cohort</th>' : '') + '<th>run</th><th>backend</th><th>task</th><th>seed/layout</th><th>condition</th><th>state</th></tr>' +
    rows.map(r => `<tr><td class="small mono">${esc(hhmm(r.started_utc))}</td>${withCohort ? `<td class="small"><a href="${esc(cohortHref(r.experiment, r.cohort))}">${esc(r.cohort)}</a></td>` : ''}
      <td class="mono small"><a href="${esc(runHref(r.run_id))}">${esc(r.short_id || r.run_id)}</a></td><td class="small">${esc(r.backend_name || '')}</td>
      <td>${esc(r.task)}</td><td>${esc(r.column)}</td><td>${esc(r.condition)}</td><td>${badge(r.state, r.verdict)}</td></tr>`).join('') + '</table></div>';
}

/* ----------------------------------------------------------------- cohort */
const ITEM_COLS = [
  ['short_id', 'run'], ['task', 'task'], ['column', 'seed/layout'], ['condition', 'condition'], ['state', 'state'], ['verdict', 'verdict'],
  ['nodecard', 'host/card'], ['wall_s', 'wall', 1], ['model_calls', 'calls', 1], ['native_actions', 'native', 1], ['started_utc', 'started'], ['fault', 'fault'],
];
const Cohort = {
  live: true,
  mount(params) {
    this.exp = params.experiment; this.name = params.name; this.data = null;
    $('#crumb').innerHTML = `/ ${esc(this.exp)} / <b>${esc(cohortName(this.name))}</b>`;
    $('#app').innerHTML = `<div id="rhead"></div><div id="rmatrix"></div>
      <section class="panel"><h2>All runs</h2><div class="filters">
        <input type="text" id="fq" placeholder="filter (any field)" value="${esc(S.filt.q)}" size="28">
        <select id="fstate"><option value="">all states</option>${STATES.map(s => `<option${S.filt.state === s ? ' selected' : ''}>${s}</option>`).join('')}</select>
        <select id="fcond"><option value="">all conditions</option></select>
        <span id="fcount" class="muted"></span></div><div id="ritems"></div></section>`;
    for (const [id, k] of [['fq', 'q'], ['fstate', 'state'], ['fcond', 'cond']]) {
      $('#' + id).addEventListener('input', e => { S.filt[k] = e.target.value; this.renderItems(); });
    }
  },
  async load() {
    const [t, d] = await getJSON('cohort?' + qsOf({experiment: this.exp, name: this.name}));
    if (S.last.cohort === t) return; S.last.cohort = t; this.data = d;
    const conds = Object.keys(d.brief.conditions);
    $('#fcond').innerHTML = '<option value="">all conditions</option>' + conds.map(c => `<option${S.filt.cond === c ? ' selected' : ''}>${esc(c)}</option>`).join('');
    keepScroll(() => { this.renderHead(); this.renderMatrix(); this.renderItems(); });
  },
  renderHead() {
    const d = this.data, b = d.brief;
    $('#rhead').innerHTML = `<div class="head"><h1>${b.name === ALL ? 'All cohorts' : 'Cohort ' + esc(b.name)}</h1><span class="muted">${esc(b.experiment)}</span>
      ${b.name !== ALL ? `<a class="small" href="${esc(cohortHref(b.experiment, ALL))}">all cohorts of this experiment</a>` : ''}</div>
      <section class="panel"><div class="kv">
        <div><div class="k">backends</div><div class="v">${esc(b.backends.join(', '))}</div></div>
        <div><div class="k">first start</div><div class="v">${esc(hhmm(b.created_utc))}</div></div>
        <div><div class="k">last start</div><div class="v">${esc(hhmm(b.last_utc))}</div></div>
        <div><div class="k">condition</div><div class="v">${d.condition_path ? 'plan ' + esc(d.condition_path) : 'code mode or tool calls'}</div></div>
        <div><div class="k">success by condition</div><div class="v">${condScores(b.conditions)}</div></div>
        <div><div class="k">configs</div><div class="v small" data-tip="${esc(b.configs.join('\n'))}">${b.configs.length}</div></div>
      </div>
      <div class="row" style="margin-top:8px;align-items:center"><div style="flex:1">${progressBar(b.counts, b.total)}</div><div>${b.finished}/${b.total} finished</div><div>${countChips(b.counts)}</div></div></section>`;
  },
  renderMatrix() {
    const d = this.data, b = d.brief, eps = d.items;
    const tasks = b.tasks, cols = b.columns, conds = Object.keys(b.conditions), L = condLetters(conds);
    const pick = (t, c, k) => {
      const xs = eps.filter(i => String(i.task) === t && i.column === c && i.condition === k);
      xs.sort((x, y) => (x.started_utc || '').localeCompare(y.started_utc || ''));
      return [xs[xs.length - 1], xs.length];
    };
    let h = `<section class="panel"><h2>Matrix <span class="muted small">task × seed or layout; one square per condition (latest run; ✓ success, ✗ fail, … running, ! fault)</span></h2>
      <div class="legend">${conds.map(c => `<span><b>${esc(L[c])}</b> ${esc(c)}</span>`).join('')}` +
      [['success', 'var(--ok)'], ['fail', 'var(--bad)'], ['running', 'var(--run)'], ['fault', 'var(--fault)']]
        .map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join('') + '</div><div class="scroll"><table class="matrix"><tr><th></th>' +
      cols.map(c => `<th>${esc(c)}</th>`).join('') + '</tr>';
    for (const t of tasks) {
      h += `<tr><th class="task">${esc(t)}</th>`;
      for (const c of cols) {
        h += `<td><div class="cell" style="width:${Math.max(1, conds.length) * 38}px">`;
        for (const k of conds) {
          const [it, n] = pick(t, c, k);
          if (!it) { h += `<div class="half c-none">${esc(L[k])}</div>`; continue; }
          const tip = `${it.run_id}\n${k} · ${it.state}${it.verdict ? ' · ' + it.verdict : ''}${it.baseline ? '\n' + it.baseline : ''}\nwall ${fmtS(it.wall_s)} · calls ${fmtN(it.model_calls)} · native ${fmtN(it.native_actions)}` +
            `\n${it.node ?? '–'}/${it.card ?? '–'}${n > 1 ? '\n' + n + ' runs in this cell (latest shown)' : ''}${it.fault ? '\nfault: ' + it.fault : ''}`;
          h += `<div class="half ${cellClass(it)}" data-tip="${esc(tip)}" onclick="location.hash='${esc(runHref(it.run_id))}'">${esc(L[k])}${stateMark(it)}${n > 1 ? '<sup>' + n + '</sup>' : ''}</div>`;
        }
        h += '</div></td>';
      }
      h += '</tr>';
    }
    h += '</table></div>';
    // per task, every condition of the cohort
    h += '<h3>Per task</h3><div class="scroll"><table><tr><th>task</th>' + conds.map(k => `<th class="num">${esc(k)} succ/n</th>`).join('') +
      conds.map(k => `<th class="num">${esc(k)} median wall</th>`).join('') + conds.map(k => `<th class="num">${esc(k)} median calls</th>`).join('') +
      conds.map(k => `<th class="num">${esc(k)} median native</th>`).join('') + '</tr>';
    for (const t of tasks) {
      const g = k => eps.filter(i => String(i.task) === t && i.condition === k);
      const sc = k => { const vs = g(k).flatMap(i => i.round_verdicts || [i.verdict]).filter(v => v === 'success' || v === 'late' || v === 'fail'); return `${vs.filter(v => v === 'success').length}/${vs.length}`; };
      const md = (k, f) => median(g(k).filter(i => i.state === 'done').map(i => i[f]));
      h += `<tr><td>${esc(t)}</td>` + conds.map(k => `<td class="num">${sc(k)}</td>`).join('') + conds.map(k => `<td class="num">${fmtS(md(k, 'wall_s'))}</td>`).join('') +
        conds.map(k => `<td class="num">${fmtN(md(k, 'model_calls'))}</td>`).join('') + conds.map(k => `<td class="num">${fmtN(md(k, 'native_actions'))}</td>`).join('') + '</tr>';
    }
    $('#rmatrix').innerHTML = h + '</table></div></section>';
  },
  renderItems() {
    if (!this.data) return;
    const f = S.filt, q = f.q.toLowerCase();
    let xs = this.data.items.map(i => Object.assign({nodecard: (i.node ?? '') + '/' + (i.card ?? '')}, i)).filter(i =>
      (!f.state || i.state === f.state) && (!f.cond || i.condition === f.cond) &&
      (!q || Object.values(i).some(v => v != null && String(v).toLowerCase().includes(q))));
    const {key, dir} = S.sort;
    xs.sort((a, b) => { const x = a[key], y = b[key]; if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
      return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * dir; });
    $('#fcount').textContent = xs.length + ' / ' + this.data.items.length;
    let h = '<div class="scroll"><table><tr>' + ITEM_COLS.map(([k, n, num]) => `<th class="sort${num ? ' num' : ''}" data-sort="${k}">${n}${S.sort.key === k ? (S.sort.dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('') + '</tr>';
    for (const i of xs) {
      h += `<tr class="click" onclick="location.hash='${esc(runHref(i.run_id))}'"><td class="mono small">${esc(i.short_id)}</td><td>${esc(i.task)}</td><td>${esc(i.column)}</td>
        <td>${esc(i.condition)}</td><td>${badge(i.state)}</td><td>${i.verdict ? `<span class="badge s-${i.verdict}">${i.verdict}</span>` : ''}${i.round_verdicts ? ` <span class="small muted">${esc(i.round_verdicts.join(' '))}</span>` : ''}</td>
        <td class="small">${esc(i.nodecard)}</td><td class="num">${fmtS(i.wall_s)}</td><td class="num">${fmtN(i.model_calls)}</td>
        <td class="num">${fmtN(i.native_actions)}</td><td class="small mono">${esc(hhmm(i.started_utc))}</td><td class="small err">${esc(i.fault || '')}</td></tr>`;
    }
    $('#ritems').innerHTML = h + '</table></div>';
    for (const th of $('#ritems').querySelectorAll('th[data-sort]')) th.onclick = () => {
      const k = th.dataset.sort; S.sort = {key: k, dir: S.sort.key === k ? -S.sort.dir : 1}; this.renderItems(); };
  },
};

/* ----------------------------------------------------------------- run detail */
function kvBlock(pairs) { return '<div class="kv">' + pairs.map(([k, v, tip]) => `<div${tip ? ` data-tip="${esc(tip)}"` : ''}><div class="k">${esc(k)}</div><div class="v">${v == null || v === '' ? '–' : v}</div></div>`).join('') + '</div>'; }
function verdictBadge(s) {
  if (s.verdict === 'success') return '<span class="verdict s-success">SUCCESS</span>';
  if (s.verdict === 'fail') return '<span class="verdict s-fail">FAIL</span>';
  return `<span class="verdict s-${esc(s.state)}">${esc((s.state || 'unknown').toUpperCase())}</span>`;
}
function summaryKV(s, compact) {
  const pairs = [
    ['termination', esc(s.termination) + (s.termination_authority ? ` <span class="muted small">(${esc(s.termination_authority)})</span>` : '')],
    ['condition / baseline', `<b>${esc(s.condition || '?')}</b> ${esc(s.baseline || '')}`],
    ['backend / task / seed', `${esc(s.backend_name || s.backend)} / ${esc(s.task)} / ${esc(s.column)}`],
    ['host / card', `<span class="small">${esc(s.node ?? '–')}</span> / ${esc(s.card ?? '–')}`],
    ['wall', fmtS(s.wall_s)],
    ['model calls', fmtN(s.model_calls)],
    ['tool calls', fmtN(s.tool_calls) + (s.script_calls ? ` <span class="muted small">(+${s.script_calls} in scripts)</span>` : '')],
    ['robot actions', fmtN(s.actions) + (s.refused_proposals ? ` <span class="muted small">(${s.refused_proposals} refused)</span>` : '')],
    ['native actions', fmtN(s.native_actions) + (s.native_action_limit ? ` <span class="muted small">/ ${s.native_action_limit}</span>` : '')],
    ['tokens in / out', `${fmtN(s.tokens_in)} / ${fmtN(s.tokens_out)}`, s.tokens_reasoning ? 'reasoning ' + s.tokens_reasoning : ''],
    ['observations', fmtN(s.steps)],
  ];
  if (s.round_verdicts) pairs.push(['rounds', esc(s.round_verdicts.join(' '))]);
  if (!compact) pairs.push(['cohort', `<a href="${esc(cohortHref(s.experiment, s.cohort))}">${esc(s.cohort)}</a>`], ['config', `<span class="mono small">${esc(s.config || '–')}</span>`],
    ['started', esc(hhmm(s.started_utc))], ['finished', esc(hhmm(s.finished_utc))], ['decisions', fmtN(s.decisions)], ['model', esc(s.model || '')],
    ['sealed', s.sealed ? 'yes' : 'no']);
  if (s.fault) pairs.push(['fault', `<span class="err">${esc(s.fault)}</span>`]);
  return kvBlock(pairs);
}
function actionRow(a, key) {
  const r = a.receipt || {};
  const st = r.status || (a.refusal ? 'refused' : a.receipt_art ? 'receipt' : 'no receipt');
  const cls = st === 'reached' || st === 'executed' ? 's-done' : st === 'refused' || st === 'no receipt' ? 's-fail' : 's-fault';
  const dur = r.started && r.finished ? fmtS(r.finished - r.started) : '';
  return `<div class="act"><span class="badge s-moved">${esc(r.action || a.kind || 'action')}</span><span class="badge ${cls}">${esc(st)}</span>
    ${r.failure_code ? `<span class="err mono small">${esc(r.failure_code)}${r.failure_stage ? ' @' + esc(r.failure_stage) : ''}</span>` : ''}
    <span class="small">native ${fmtN(r.native_actions)}</span>${r.target_arms ? `<span class="small muted">arms ${esc(r.target_arms.join(','))}</span>` : ''}
    ${r.motion_strategy ? `<span class="small muted">${esc(r.motion_strategy)}</span>` : ''}${dur ? `<span class="small muted">${dur}</span>` : ''}
    ${r.terminal_after_action ? '<span class="badge s-done">terminal</span>' : ''}${a.call_index != null ? `<span class="small muted">call #${a.call_index}</span>` : ''}
    <span class="mono small muted">${esc(a.action_id)}</span>
    ${r.arguments ? `<details class="fold" data-k="${key}:args" ${det(key + ':args', false)} style="flex-basis:100%"><summary>arguments</summary><pre>${esc(r.arguments)}</pre></details>` : ''}
    ${a.refusal ? `<pre class="err" style="flex-basis:100%">${esc(a.refusal)}</pre>` : ''}</div>`;
}
function scriptCalls(c, ctx) {
  const xs = c.script_calls || [];
  if (!xs.length) return '';
  return `<div class="small muted">the script's own calls (${xs.length})</div><ol class="script">` + xs.map(sc =>
    `<li><span class="tool">${esc(sc.name)}</span> ${sc.ok === false ? '<span class="badge s-fail">error</span>' : ''}${sc.moved ? '<span class="badge s-moved">moved</span>' : ''}
      <span class="mono small muted">${esc(JSON.stringify(sc.args || {}).slice(0, 240))}</span>${sc.error ? `<pre class="err">${esc(sc.error)}</pre>` : ''}
      ${sc.images && sc.images.length ? '<div class="thumbs sm">' + sc.images.map((p, i) => thumb(fileUrl(ctx.dir + '/' + p), `${sc.name} image ${i}`)).join('') + '</div>' : ''}</li>`).join('') + '</ol>';
}
function callRow(c, ctx, key) {
  const b = c.b;
  const moved = c.moved;
  const err = b ? (b.is_error || (b.exit_code != null && b.exit_code !== 0)) : c.ok === false;
  let title, body = '';
  if (b) {
    const inp = b.input || {};
    const main = inp.command != null ? inp.command : inp.path != null ? inp.path : JSON.stringify(inp);
    const first = String(main).split('\n').find(l => l.trim() && !/^cd \/workspace\s*(&&)?\s*$/.test(l.trim())) || String(main);
    const wrote = [...(b.changes.added || []), ...(b.changes.modified || [])];
    const dur = b.started && b.finished ? fmtS(b.finished - b.started) : '';
    title = `<span class="muted small">#${esc(b.index)}</span><span class="tool">${esc(b.tool)}</span><span class="cmd mono">${esc(first.slice(0, 300))}</span>
      ${b.exit_code != null && b.exit_code !== 0 ? `<span class="badge s-fail">exit ${esc(b.exit_code)}</span>` : ''}${b.is_error ? '<span class="badge s-fail">error</span>' : ''}
      ${moved ? `<span class="badge s-moved">moved · ${fmtN(b.native_actions)} na</span>` : ''}${wrote.length ? `<span class="chip w" data-tip="${esc(wrote.join('\n'))}">wrote ${wrote.length}</span>` : ''}
      <span class="small muted">${dur}</span>`;
    if (inp.command != null) body += `<pre>${esc(inp.command)}</pre>`;
    else for (const [k, v] of Object.entries(inp)) body += `<div class="small muted">${esc(k)}</div><pre>${esc(v)}</pre>`;
    if (b.input_truncated) body += `<div class="small muted">input truncated</div>`;
    body += `<div class="small muted">output ${fmtN(b.output_chars)} chars${b.output_truncated ? ` · <button class="small" onclick="loadFullCall(this, ${b.index}, '${esc(ctx.run)}')">load full</button>` : ''}</div><pre class="out">${esc(b.output)}</pre>`;
    const ic = Object.entries(b.interface_calls || {});
    if (ic.length) body += '<div>' + ic.map(([k, v]) => `<span class="chip">${esc(k)} ×${esc(v)}</span>`).join('') + '</div>';
    if (wrote.length || (b.changes.removed || []).length) body += '<div>' + wrote.map(f => `<span class="chip w">${esc(f)}</span>`).join('') +
      (b.changes.removed || []).map(f => `<span class="chip">removed ${esc(f)}</span>`).join('') + '</div>';
    for (const a of b.actions || []) body += `<div class="small">→ ${esc(a.kind || 'action')} <span class="badge ${a.status === 'reached' ? 's-done' : 's-fault'}">${esc(a.status || (a.refusal ? 'refused' : '?'))}</span> ${esc(a.failure_code || '')} native ${fmtN(a.native_actions)} · step ${esc(a.before)} → ${esc(a.after)}</div>`;
  } else {
    const shown = c.code != null ? c.code : (c.args == null ? '' : JSON.stringify(c.args));
    const first = String(shown).split('\n').find(l => l.trim()) || '';
    const nScript = (c.script_calls || []).length;
    title = `<span class="tool">${esc(c.name)}</span><span class="cmd mono">${esc(first.slice(0, 300))}</span>
      ${c.ok === false ? '<span class="badge s-fail">error</span>' : c.ok == null ? '<span class="badge s-running">pending</span>' : ''}
      ${moved ? `<span class="badge s-moved">moved${c.native_actions ? ' · ' + fmtN(c.native_actions) + ' na' : ''}</span>` : ''}
      ${nScript ? `<span class="chip">${nScript} script call${nScript > 1 ? 's' : ''}</span>` : ''}
      ${c.effect && c.effect !== 'read_only' ? `<span class="chip">${esc(c.effect)}</span>` : ''}${c.images && c.images.length ? `<span class="chip">${c.images.length} img</span>` : ''}`;
    body += c.code != null ? `<pre>${esc(c.code)}</pre>` : `<pre>${esc(JSON.stringify(c.args, null, 1))}</pre>`;
    body += scriptCalls(c, ctx);
    if (c.error) body += `<pre class="err">${esc(c.error)}</pre>`;
    if (c.result_art) body += `<div class="small muted">result ${fmtN(c.result_bytes)} B · <button class="small" onclick="loadText(this, '${esc(ctx.dir + '/' + c.result_art)}', true)">load full</button> <a target="_blank" href="${esc(fileUrl(ctx.dir + '/' + c.result_art))}">raw</a></div>`;
    if (c.result_preview) body += `<pre class="out">${esc(c.result_preview)}</pre>`;
    if (c.images && c.images.length) body += '<div class="thumbs sm">' + c.images.map((p, i) => thumb(fileUrl(ctx.dir + '/' + p), `${c.name} image ${i}`)).join('') + '</div>';
  }
  return `<details class="call${moved ? ' moved' : ''}${err ? ' err' : ''}" data-k="${key}" ${det(key, false)}><summary>${title}</summary><div class="body">${body}</div></details>`;
}
function turnBlock(t, ctx) {
  const key = ctx.id + ':' + t.mc;
  const u = t.usage || {};
  let h = `<div class="turn"><div class="meta">${esc(t.mc)} · ${esc(hhmm(t.ts))}${u.input_tokens != null ? ` · in ${fmtN(u.input_tokens)} / out ${fmtN(u.output_tokens)}` : ''}${t.finish ? ' · ' + esc(t.finish) : ''}${t.error ? ' · <span class="err">' + esc(typeof t.error === 'string' ? t.error : JSON.stringify(t.error)) + '</span>' : ''}</div>`;
  if (t.reasoning) h += `<details class="fold" data-k="${key}:r" ${det(key + ':r', false)}><summary>reasoning (${fmtN(t.reasoning.length)} chars)</summary><pre>${esc(t.reasoning)}</pre></details>`;
  if (t.content && t.content.trim()) h += `<div class="content">${esc(t.content.trim())}</div>`;
  h += t.calls.map((c, i) => callRow(c, ctx, key + ':c' + i)).join('');
  return h + '</div>';
}
function stepBlock(st, ctx, compact) {
  const key = ctx.id + ':step:' + st.step;
  const ncalls = st.turns.reduce((n, t) => n + t.calls.length, 0);
  const moved = st.turns.some(t => t.calls.some(c => c.moved));
  const chips = st.actions.map(a => { const r = a.receipt || {}; const s = r.status || (a.refusal ? 'refused' : '–');
    return `<span class="badge ${s === 'reached' || s === 'executed' ? 's-done' : 's-fault'}" data-tip="${esc([r.failure_code, r.native_actions != null ? 'native ' + r.native_actions : null].filter(Boolean).join('\n'))}">${esc(r.action || a.kind || 'action')}: ${esc(s)}</span>`; }).join(' ');
  let h = `<details class="step" data-k="${key}" ${det(key, !compact)}><summary><b>${st.step == null ? 'unbound' : 'Step ' + st.step}</b>
    <span class="muted small">${st.turns.length} model calls · ${ncalls} tool calls</span>${moved ? '<span class="badge s-moved">moved</span>' : ''} ${chips}</summary><div class="step-body">`;
  const cams = Object.entries(st.images || {});
  if (cams.length) h += `<div class="thumbs${compact ? ' sm' : ''}">` + cams.map(([cam, rel]) => thumb(fileUrl(ctx.dir + '/' + rel), `step ${st.step} · ${cam}`)).join('') + '</div>';
  else if (st.step != null) h += '<div class="muted small">no observation images</div>';
  if (st.turns.length) {
    if (compact) h += `<details class="fold" data-k="${key}:turns" ${det(key + ':turns', false)}><summary>agent activity</summary>${st.turns.map(t => turnBlock(t, ctx)).join('')}</details>`;
    else h += st.turns.map(t => turnBlock(t, ctx)).join('');
  }
  if (st.actions.length) h += '<div>' + st.actions.map((a, i) => actionRow(a, key + ':a' + i)).join('') + '</div>';
  return h + '</div></details>';
}
async function loadFullCall(btn, index, runId) {
  const pre = btn.closest('.body').querySelector('pre.out');
  btn.disabled = true;
  try { const [, d] = await getJSON('run/call?' + qsOf({id: runId, index})); pre.textContent = typeof d.output === 'string' ? d.output : JSON.stringify(d.output, null, 1); btn.textContent = 'full'; }
  catch (e) { pre.textContent = 'error: ' + e; }
}
async function loadText(btn, path, pretty) {
  const pre = btn.closest('.body').querySelector('pre.out') || btn.parentElement.nextElementSibling;
  btn.disabled = true;
  try { let t = await getText(path); if (pretty) { try { t = JSON.stringify(JSON.parse(t), null, 1); } catch (e) { /* keep raw */ } } pre.textContent = t; }
  catch (e) { pre.textContent = 'error: ' + e; }
}
window.loadFullCall = loadFullCall; window.loadText = loadText;

function filesPane(files, runId) {
  if (!files.length) return '<div class="empty">The agent wrote no files (no code-mode workspace in this run).</div>';
  return `<p class="small muted">Files the agent wrote in its code-mode workspace, with the outline of the newest version (module docstring and functions, as the agent wrote them). Click a version to see its source.</p>` + files.map(f => {
    const o = f.outline;
    const defs = o && o.defs && o.defs.length ? `<ul>${o.defs.map(d => `<li><code>${esc(d.kind === 'class' ? 'class ' + d.name : d.name + '(' + d.sig + ')')}</code>${d.doc ? ' — ' + esc(d.doc) : ''}</li>`).join('')}</ul>` : '';
    const vers = f.versions.map((v, k) => `<button class="small" data-blob="${esc(v.sha || '')}" title="call ${esc(v.call)}">v${k + 1} · call ${esc(v.call)}</button>`).join(' ');
    return `<section class="panel tl-f"><h3 class="mono">${esc(f.path)} <span class="muted small">${f.size ?? '?'} B${f.removed != null ? ' · removed at call ' + f.removed : ''}</span></h3>
      ${o && o.doc ? `<div class="tl-doc">${esc(o.doc)}</div>` : ''}${defs}<div class="small">versions: ${vers}</div><pre class="tl-src" hidden></pre></section>`;
  }).join('');
}

const Run = {
  live: true,
  mount(id, params) {
    this.id = id; this.params = params; this.data = null; this.ctx = null; this.live = true; this.files = null;
    if (params.tab) S.tab[id] = params.tab;
    $('#crumb').innerHTML = '/ run ' + esc(id);
    $('#app').innerHTML = '<div id="runhead"></div><div class="tabs" id="tabs"></div><div id="tabbody"></div>';
  },
  tab() { return S.tab[this.id] || 'timeline'; },
  async load() {
    const [t, d] = await getJSON('run?' + qsOf({id: this.id}));
    this.live = !d.summary || d.summary.state === 'running';
    if (S.last.run === t) return; S.last.run = t;
    this.data = d; this.ctx = {id: this.id, run: this.id, dir: d.summary.run_dir};
    const s = d.summary;
    $('#crumb').innerHTML = `/ <a href="${esc(cohortHref(s.experiment, s.cohort))}">${esc(s.cohort)}</a> / run ${esc(s.short_id || s.run_id)}`;
    keepScroll(() => { this.renderHead(); this.renderTabs(); this.renderTab(); });
  },
  renderHead() {
    const d = this.data, s = d.summary;
    let h = `<div class="head">${verdictBadge(s)}<h1 class="mono runid">${esc(s.run_id)}</h1>${badge(s.state)}
      <a class="small" href="#/compare?${qsOf({a: s.run_id})}">compare…</a></div>`;
    if (s.task_instruction) h += `<div class="muted">“${esc(s.task_instruction)}”</div>`;
    h += `<section class="panel">${summaryKV(s)}</section>`;
    h += '<div class="small rawlinks">raw: ' + d.raw.map(r => `<a target="_blank" href="${esc(fileUrl(s.run_dir + '/' + r.rel))}">${esc(r.rel)}</a> <span class="muted">${fmtN(r.size)}B</span>`).join(' · ') + '</div>';
    $('#runhead').innerHTML = h;
  },
  renderTabs() {
    const s = this.data.summary;
    const tabs = [['timeline', 'Timeline'], ['replay', '3D replay']];
    if (s.code_mode) tabs.push(['files', "Agent's files"]);
    tabs.push(['logs', 'Logs'], ['events', 'Run events']);
    if (!tabs.some(([k]) => k === this.tab())) S.tab[this.id] = 'timeline';
    $('#tabs').innerHTML = tabs.map(([k, n]) => `<button type="button" data-tab="${k}" class="${this.tab() === k ? 'on' : ''}">${n}</button>`).join('');
    for (const b of $('#tabs').querySelectorAll('button')) b.onclick = () => { S.tab[this.id] = b.dataset.tab; this.renderTabs(); $('#tabbody').innerHTML = ''; this.renderTab(); };
  },
  renderTab() {
    const tab = this.tab(), d = this.data, body = $('#tabbody'), s = d.summary;
    if (tab === 'timeline') {
      if (!body.querySelector('#steps')) body.innerHTML = `<div class="filters"><button class="small" type="button" id="openall">open all steps</button><button class="small" type="button" id="closeall">close all steps</button>
        <span class="muted small">${d.steps.length} steps · purple border = call moved the robot · click images to enlarge (← → to browse)</span></div><div id="steps" data-gal="1"></div>`;
      const setAll = v => { for (const st of d.steps) S.open[this.id + ':step:' + st.step] = v; S.last.run = null; this.load(); };
      $('#openall').onclick = () => setAll(true); $('#closeall').onclick = () => setAll(false);
      if (!d.steps.length) { $('#steps').innerHTML = '<div class="empty">no observations or agent activity recorded' + (s.fault ? ': ' + esc(s.fault) : '') + '</div>'; return; }
      patchList($('#steps'), d.steps.map(st => ({key: 's' + st.step, html: stepBlock(st, this.ctx)})));
    } else if (tab === 'replay') {
      if (s.backend === 'synthetic') {
        body.innerHTML = `<div class="panel notice">A Synthetic run has no robot, so it has no 3D replay. Its model calls and receipts are on the Timeline tab.</div>`;
        return;
      }
      // the replay page reloads its own data (Reload button); auto-refresh must not rebuild the iframe
      if (!body.querySelector('#replay')) body.innerHTML = `<div class="small"><a target="_blank" href="static/replay.html?run=${encodeURIComponent(this.id)}">open in its own tab</a>
        <span class="muted">· layers a backend has nothing for are disabled, each with its reason</span></div>
        <iframe id="replay" class="replay-frame" src="static/replay.html?run=${encodeURIComponent(this.id)}"></iframe>`;
    } else if (tab === 'files') {
      const show = files => {
        body.innerHTML = filesPane(files, this.id);
        for (const btn of body.querySelectorAll('button[data-blob]')) btn.onclick = async () => {
          const box = btn.closest('.tl-f').querySelector('.tl-src');
          if (!btn.dataset.blob) { box.hidden = false; box.textContent = '(version not recorded)'; return; }
          box.hidden = false; box.textContent = 'loading…';
          const r = await fetch('api/blob?' + qsOf({id: this.id, sha: btn.dataset.blob}));
          box.textContent = r.ok ? await r.text() : 'HTTP ' + r.status;
        };
      };
      if (this.files) { show(this.files); return; }
      body.innerHTML = '<div class="muted">loading…</div>';
      getJSON('run/files?' + qsOf({id: this.id})).then(([, f]) => { this.files = f.files; if (this.tab() === 'files') show(f.files); })
        .catch(e => { body.innerHTML = `<div class="err">${esc(e)}</div>`; });
    } else if (tab === 'logs') {
      let h = `<section class="panel"><h2>supervisor.log (tail)</h2>${d.supervisor_tail ? `<pre style="max-height:60vh">${esc(d.supervisor_tail)}</pre>` : '<div class="empty">empty or missing</div>'}</section>`;
      if (d.observations_custom && d.observations_custom.length) h += `<section class="panel"><h2>Agent-core messages (last ${d.observations_custom.length})</h2>` +
        d.observations_custom.map(c => `<div class="small muted">${esc(c.ts)} · ${esc(c.type)}</div><pre>${esc(c.content)}</pre>`).join('') + '</section>';
      keepScroll(() => { body.innerHTML = h; });
    } else if (tab === 'events') {
      body.innerHTML = '<section class="panel"><div class="scroll"><table><tr><th>utc</th><th>type</th><th>fields</th></tr>' + (d.run_events || []).map(e => `<tr><td class="small mono">${esc(hhmm(e.utc))}</td><td>${esc(e.type)}</td>
        <td class="small mono">${esc(Object.entries(e.fields || {}).map(([k, v]) => k + '=' + (typeof v === 'string' ? v : JSON.stringify(v))).join('  '))}</td></tr>`).join('') + '</table></div></section>';
    }
  },
};

/* ----------------------------------------------------------------- compare */
const Compare = {
  live: true,
  mount(params) {
    this.params = params; $('#crumb').textContent = '/ compare';
    $('#app').innerHTML = '<div class="filters"><b>A</b><select id="ca"></select><b>B</b><select id="cb"></select><button class="small" type="button" id="cswap">swap</button></div><div class="cmp"><div id="c0"></div><div id="c1"></div></div>';
    this.runs = null; this.last = {}; this.live = true;
  },
  async load() {
    if (!this.runs) {
      const [, r] = await getJSON('runs'); this.runs = r.runs;
      let {a, b} = this.params;
      if (!a && this.runs.length) a = this.runs[this.runs.length - 1].run_id;
      if (a && !b) {
        const ra = this.runs.find(x => x.run_id === a);
        if (ra) { const m = this.runs.filter(x => x.run_id !== a && x.cohort === ra.cohort && x.task === ra.task && x.column === ra.column && x.condition !== ra.condition); b = (m[m.length - 1] || {}).run_id; }
      }
      this.a = a; this.b = b;
      const opts = sel => '<option value="">–</option>' + this.runs.map(x => `<option value="${esc(x.run_id)}"${x.run_id === sel ? ' selected' : ''}>${esc(x.cohort)} · ${esc(x.task)} ${esc(x.column)} · ${esc(x.condition)} · ${esc(x.verdict || x.state)} · ${esc(x.short_id)}</option>`).join('');
      $('#ca').innerHTML = opts(a); $('#cb').innerHTML = opts(b);
      const go = () => { location.hash = '#/compare?' + qsOf({a: $('#ca').value, b: $('#cb').value}); };
      $('#ca').onchange = go; $('#cb').onchange = go;
      $('#cswap').onclick = () => { location.hash = '#/compare?' + qsOf({a: this.b, b: this.a}); };
    }
    let live = false;
    for (const [i, id] of [[0, this.a], [1, this.b]]) {
      const el = $('#c' + i);
      if (!id) { el.innerHTML = '<div class="empty">pick a run</div>'; continue; }
      const [t, d] = await getJSON('run?' + qsOf({id}));
      if (d.summary.state === 'running') live = true;
      if (this.last[i] === t) continue; this.last[i] = t;
      const s = d.summary, ctx = {id: id + ':cmp' + i, run: id, dir: s.run_dir};
      keepScroll(() => {
        el.innerHTML = `<div class="head">${verdictBadge(s)}<a class="mono small" href="${esc(runHref(id))}">${esc(s.short_id || id)}</a></div><section class="panel">${summaryKV(s, true)}</section><div class="steps" data-gal="1"></div>`;
        patchList(el.querySelector('.steps'), d.steps.map(st => ({key: 's' + st.step, html: stepBlock(st, ctx, true)})));
      });
    }
    this.live = live;
  },
};

/* ----------------------------------------------------------------- routing */
function parseHash() {
  const h = location.hash.replace(/^#/, '') || '/';
  const [path, query] = h.split('?');
  const params = {}; for (const [k, v] of new URLSearchParams(query || '')) params[k] = v;
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  return {parts, params};
}
let refreshing = false;
async function refresh() {
  if (!S.view || refreshing) return; refreshing = true;
  try { await S.view.load(); setUpdated(); } catch (e) { setUpdated(String(e.message || e)); } finally { refreshing = false; }
}
function route() {
  const {parts, params} = parseHash();
  S.last = {};
  if (parts[0] === 'cohort' && params.experiment && params.name) { S.view = Cohort; Cohort.mount(params); }
  else if (parts[0] === 'run' && parts[1]) { S.view = Run; Run.mount(parts[1], params); }
  else if (parts[0] === 'compare') { S.view = Compare; Compare.mount(params); }
  else { S.view = Overview; Overview.mount(); }
  window.scrollTo(0, 0);
  refresh();
}
window.addEventListener('hashchange', route);
setInterval(() => { if ($('#auto').checked && document.visibilityState === 'visible' && S.view && S.view.live !== false) refresh(); }, 10000);
route();
