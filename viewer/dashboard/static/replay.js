// ORA 3D replay of one run of any backend on one clock (the bundle's clock: RoboDojo native actions, the execution
// capture's simulator steps, else the receipts' steps): the robot from its URDF posed from logged joints where the
// run has them (RoboDojo's dual ARX X5, RoboTwin's aloha-agilex), else its links from a run's scene trace (LIBERO-Pro's
// Panda, RoboCasa365's PandaOmron), end-effector paths, the agent's RGB-D point cloud, robot commands, camera frames, simulator object meshes at
// their poses (RoboDojo's ground truth, a run's scene trace), and the agent's trace. Ported from Shaoxiong Duan's code-vs-tools dashboard (snapshot 2026-10-31), itself adapted from an earlier
// interactive episode viewer. Light theme only (the shared Viewer template).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import URDFLoader from 'urdf-loader';

const ARMS = ['left', 'right'];
const ARM_COLOR = { left: 0x2f6db5, right: 0xb5432f };
const ARM_LIGHT = { left: 0xb9cde6, right: 0xe8c1b9 };
const MODE_COLOR = { free_space: '#8a929c', precision: '#d08b2e', contact_approach: '#e0662a', contact: '#c0392b', gripper: '#202124',
  fast: '#8a929c', move: '#3f8fd1', fine: '#d08b2e', custom: '#7a3fc0' };  // move_path modes (v5): fast gray, move blue, fine amber, contact red
// a command without an interaction mode is coloured by its action
const ACTION_COLOR = { set_gripper: '#202124', move_base: '#2a8c82' };
// a command whose status is one of these, or that has no status and no failure code, did what it was asked
const OK_STATUS = new Set(['reached', 'executed', 'completed', 'succeeded', 'success']);
const BASE_COLOR = 0x2a8c82;
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const RUN = params.get('run');
// A multi-round session resets the world and the native clock every round: the page shows one round at a time.
const ROUND = Number(params.get('round') || 0) || 0;
const fileUrl = (p) => 'api/file?path=' + encodeURIComponent(p);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (x, d = 3) => (typeof x === 'number' ? x.toFixed(d) : (x ?? '—'));
const v3 = (p) => new THREE.Vector3(p[0], p[1], p[2]);
const lerp = (a, b, t) => a + (b - a) * t;
const fmtN = (x) => (typeof x === 'number' ? (Number.isInteger(x) ? String(x) : x.toFixed(1)) : '—');
// the clock's unit as the page prints it next to a time: "native" (RoboDojo), else e.g. "SAPIEN step"
const unitShort = () => { const u = S.data?.clock?.unit || 'step'; return u === 'native action' ? 'native' : u; };

const S = { data: null, frames: [], ticks: [], acts: [], obs: [], video: [], n: 0, maxN: 0, hz: 25, rate: 25,
  playing: false, lastTs: null, robots: {}, robotSpec: null, clouds: new Map(), obsIndex: -1, actIndex: -1, start: {},
  agent: null, entries: [], entryIndex: -2, pin: null, tab: 'agent', cams: [] };

// ---------- scene ----------
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const host = $('canvasHost');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const dark = false;
scene.background = new THREE.Color(dark ? 0x1c2026 : 0xf6f7f8);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
camera.position.set(0.0, -1.25, 1.45);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, -0.15, 0.82);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f96, 1.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(1, -1.5, 3);
scene.add(sun);
const groups = {};
for (const name of ['table', 'cloud', 'traj', 'target', 'models', 'labels', 'tcp', 'base', 'robot']) { groups[name] = new THREE.Group(); scene.add(groups[name]); }
const traceLinks = new THREE.Group();  // a scene trace's robot links: under the robot toggle, beside any URDF robot
groups.robot.add(traceLinks);

const fatMaterials = new Set();
function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h);
  for (const m of fatMaterials) m.resolution.set(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  drawStrip();
}
new ResizeObserver(resize).observe(host);

function fatMaterial(opts) {
  const m = new LineMaterial(opts);
  m.resolution.set(host.clientWidth, host.clientHeight);
  fatMaterials.add(m);
  m.addEventListener('dispose', () => fatMaterials.delete(m));
  return m;
}
function clearGroup(g) {
  for (const c of [...g.children]) {
    g.remove(c);
    c.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); } });
  }
}
function sphere(r, color, opts = {}) {
  return new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12),
    new THREE.MeshBasicMaterial({ color, transparent: opts.opacity !== undefined, opacity: opts.opacity ?? 1, wireframe: !!opts.wireframe }));
}
function label(text) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.font = '600 34px system-ui, sans-serif';
  canvas.width = Math.ceil(ctx.measureText(text).width) + 20; canvas.height = 50;
  ctx.font = '600 34px system-ui, sans-serif';
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#afb5bc'; ctx.lineWidth = 2; ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
  ctx.fillStyle = '#202124'; ctx.textBaseline = 'middle'; ctx.fillText(text, 10, 26);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  s.scale.set(0.013 * canvas.width / canvas.height, 0.013, 1);
  s.renderOrder = 10;
  return s;
}

// ---------- robot (described by the bundle: robots.py) ----------
const stlLoader = new STLLoader(), gltfLoader = new GLTFLoader(), colladaLoader = new ColladaLoader(), geomCache = new Map();
function recolor(material) {
  if (!material || !material.color || material.userData.recolored) return;
  const c = material.color;
  if (c.r + c.g + c.b < 0.1) c.setHex(0x3d4147); else if (c.r + c.g + c.b > 2.9) c.setHex(0xd9dce0);
  material.userData.recolored = true;
}
function meshLoader(skip) {
  return (path, manager, done) => {
    if (skip.some((s) => path.endsWith('/' + s))) { done(null); return; }  // too large for a browser (robots.py); never requested
    if (/\.glb$/i.test(path)) { gltfLoader.load(path, (g) => done(g.scene), undefined, (e) => done(null, e)); return; }
    if (/\.dae$/i.test(path)) {
      // URDF meshes are in their link's frame; the loader would turn a Z_UP Collada file to Y-up, so undo that
      colladaLoader.load(path, (d) => { d.scene.quaternion.identity(); done(d.scene); }, undefined, (e) => done(null, e));
      return;
    }
    if (!geomCache.has(path)) geomCache.set(path, new Promise((res, rej) => stlLoader.load(path, res, undefined, rej)));
    geomCache.get(path).then((g) => { const mesh = new THREE.Mesh(g); done(mesh); recolor(mesh.material); }, (e) => done(null, e));
  };
}
function loadRobot(spec) {
  return new Promise((resolve, reject) => {
    const loader = new URDFLoader();
    loader.loadMeshCb = meshLoader(spec.skip || []);
    loader.load(spec.urdf, resolve, undefined, reject);
  });
}
function setArm(robot, a, arm) {
  const spec = S.robotSpec;
  if (!robot || !a || !spec) return;
  if (Array.isArray(a.q)) a.q.forEach((v, i) => { const j = spec.joints?.[arm]?.[i]; if (j) robot.setJointValue(j, v); });
  if (Array.isArray(a.f)) a.f.forEach((v, i) => { const j = spec.fingers?.[arm]?.[i]; if (j) robot.setJointValue(j, v); });
}
function tcpOf(pose) {  // logged pose -> TCP; RoboDojo logs ee_link (= link6), the other backends the TCP itself
  const off = S.data?.tcp_offset;
  if (!off) return v3(pose);
  const q = new THREE.Quaternion(pose[3], pose[4], pose[5], pose[6]);
  return new THREE.Vector3(off[0], off[1], off[2]).applyQuaternion(q).add(v3(pose));
}

// ---------- sampling on the native-action clock ----------
function bisect(rows, n, key = (r) => r.n) {
  let lo = 0, hi = rows.length - 1;
  if (!rows.length) return -1;
  if (n < key(rows[0])) return -1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (key(rows[mid]) <= n + 1e-6) lo = mid; else hi = mid - 1; }
  return lo;
}
function sampleArms(n) {
  const fr = S.frames;
  if (!fr.length) return null;
  const i = Math.max(bisect(fr, n), 0), a = fr[i], b = fr[Math.min(i + 1, fr.length - 1)];
  const span = b.n - a.n, u = span > 0 ? Math.min(Math.max((n - a.n) / span, 0), 1) : 0;
  const arms = {};
  for (const arm of ARMS) {
    const A = a.arms[arm], B = b.arms[arm] || A;
    if (!A) continue;
    const out = { pose: A.pose, logged: A };
    if (A.q) out.q = A.q.map((x, k) => lerp(x, B.q?.[k] ?? x, u));
    if (A.f) out.f = A.f.map((x, k) => lerp(x, B.f?.[k] ?? x, u));
    if (A.pose && B.pose && u > 0) out.tcp = tcpOf(A.pose).lerp(tcpOf(B.pose), u);
    arms[arm] = out;
  }
  return arms;
}
function sampleBase(n) {  // RoboCasa365's mobile base: [x, y, yaw], interpolated
  const fr = S.baseFrames;
  if (!fr?.length) return null;
  const i = Math.max(bisect(fr, n), 0), a = fr[i], b = fr[Math.min(i + 1, fr.length - 1)];
  const span = b.n - a.n, u = span > 0 ? Math.min(Math.max((n - a.n) / span, 0), 1) : 0;
  let dy = b.base[2] - a.base[2]; dy -= 2 * Math.PI * Math.round(dy / (2 * Math.PI));
  return [lerp(a.base[0], b.base[0], u), lerp(a.base[1], b.base[1], u), a.base[2] + dy * u];
}
function samplePoses(n) {
  const rows = S.ticks;
  if (!rows.length) return null;
  const i = Math.max(bisect(rows, n), 0), a = rows[i], b = rows[Math.min(i + 1, rows.length - 1)];
  const span = b.n - a.n, u = span > 0 ? Math.min(Math.max((n - a.n) / span, 0), 1) : 0;
  const out = {};
  for (const [k, p] of Object.entries(a.poses)) {
    const q = b.poses[k] || p;
    const qa = new THREE.Quaternion(p[3], p[4], p[5], p[6]), qb = new THREE.Quaternion(q[3], q[4], q[5], q[6]);
    out[k] = { pos: new THREE.Vector3(lerp(p[0], q[0], u), lerp(p[1], q[1], u), lerp(p[2], q[2], u)), quat: qa.slerp(qb, u) };
  }
  return { n: a.n, poses: out };
}

// ---------- static scene ----------
function buildTable() {
  clearGroup(groups.table);
  if (typeof S.data.table_height !== 'number') return;  // only RoboDojo's table is fixed and known
  const h = S.data.table_height;
  const box = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.1, 0.04), new THREE.MeshStandardMaterial({ color: dark ? 0x4a4f57 : 0xe4e1da, roughness: 0.9 }));
  box.position.set(0, 0, h - 0.0205);
  groups.table.add(box);
}
const trajLines = {};
function buildTrajectories() {
  clearGroup(groups.traj);
  for (const k of Object.keys(trajLines)) delete trajLines[k];
  for (const arm of ARMS) {
    const rows = S.frames.filter((f) => f.arms[arm]?.pose);
    const pts = rows.map((f) => tcpOf(f.arms[arm].pose));
    if (pts.length < 2) { trajLines[arm] = null; continue; }
    const g = new LineGeometry().setPositions(pts.flatMap((p) => [p.x, p.y, p.z]));
    g.setColors(new Array(pts.length * 3).fill(1));
    const line = new Line2(g, fatMaterial({ vertexColors: true, linewidth: 3 }));
    line.userData.ns = rows.map((f) => f.n);
    groups.traj.add(line);
    trajLines[arm] = line;
  }
}
function colorTrajectories(n) {
  for (const arm of ARMS) {
    const line = trajLines[arm];
    if (!line) continue;
    const past = new THREE.Color(ARM_COLOR[arm]), future = new THREE.Color(ARM_LIGHT[arm]), colors = [];
    for (const k of line.userData.ns) { const c = k <= n ? past : future; colors.push(c.r, c.g, c.b); }
    line.geometry.setColors(colors);
  }
}
const tcpMarkers = {};
function buildTcpMarkers() {
  clearGroup(groups.tcp);
  for (const arm of ARMS) { const m = sphere(0.006, ARM_COLOR[arm]); groups.tcp.add(m); tcpMarkers[arm] = m; }
}
let baseMarker = null;
function buildBase() {  // the base's path on the floor and an arrow at its pose (RoboCasa365)
  clearGroup(groups.base);
  baseMarker = null;
  const fr = S.baseFrames;
  if (!fr.length) return;
  const pts = fr.map((f) => new THREE.Vector3(f.base[0], f.base[1], 0.005));
  if (pts.length >= 2) {
    const g = new LineGeometry().setPositions(pts.flatMap((p) => [p.x, p.y, p.z]));
    groups.base.add(new Line2(g, fatMaterial({ color: BASE_COLOR, linewidth: 2 })));
  }
  const shape = new THREE.Shape([new THREE.Vector2(0.22, 0), new THREE.Vector2(-0.12, 0.12), new THREE.Vector2(-0.06, 0), new THREE.Vector2(-0.12, -0.12)]);
  baseMarker = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: BASE_COLOR, side: THREE.DoubleSide, transparent: true, opacity: 0.8 }));
  groups.base.add(baseMarker);
}

// ---------- object meshes ----------
const meshCache = new Map(), modelObjects = {};
function loadMeshGeometry(usd) {
  const url = 'api/mesh?' + new URLSearchParams({ usd, v: S.data?.mesh_version ?? 1 });
  if (!meshCache.has(url)) {
    meshCache.set(url, fetch(url).then((r) => { if (!r.ok) throw new Error(`mesh ${usd}: HTTP ${r.status}`); return r.arrayBuffer(); }).then((buf) => {
      const head = new DataView(buf), n = head.getUint32(0, true), f = head.getUint32(4, true);
      const pos = new Float32Array(buf, 8, n * 3), col = new Uint8Array(buf, 8 + n * 12, n * 3);
      const idxOffset = 8 + n * 15;
      const idx = idxOffset % 4 === 0 ? new Uint32Array(buf, idxOffset, f * 3) : new Uint32Array(buf.slice(idxOffset, idxOffset + f * 12));
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3, true));
      g.setIndex(new THREE.BufferAttribute(idx, 1));
      g.computeVertexNormals();
      return g;
    }));
  }
  return meshCache.get(url);
}
// A static instance (kind "geometry": RoboDojo removes its rigid body, so it is a fixed collider) cannot move or turn
// in the simulator; the capture reads its pose once per round. Its label and the Objects table say so.
const FIXED_NOTE = 'fixed scene geometry: a static collider in the simulator, it cannot move or turn (pose read once per round)';
// A scene trace (scene_trace.py) in the bundle's instances and ticks: its static bodies are fixed instances, its
// moving ones (robot links among them) a pose per row, and a trace that names the TCP gives the end-effector frames, the
// hand's path between commands. poses.bin is float32 [row, body, 7] of the whole run; the bundle names this round's rows.
const traceUrl = (t, file) => 'api/scene-trace?' + new URLSearchParams({ id: RUN, file, v: t.version });
async function applySceneTrace(data) {
  const t = data.scene_trace;
  if (!t) return;
  const r = await fetch(traceUrl(t, 'poses.bin'));
  if (!r.ok) throw new Error(`scene trace: HTTP ${r.status}`);
  const poses = new Float32Array(await r.arrayBuffer());
  const at = (row, body) => { const i = (row * t.body_count + body) * 7; return Array.from(poses.subarray(i, i + 7)); };
  const moving = [];
  data.instances = {};
  for (const b of t.bodies) {
    const pose = at(t.rows[0][0], b.index);
    if (Number.isNaN(pose[0])) continue;  // not in this round's scene
    data.instances[b.id] = { kind: b.kind, glb: traceUrl(t, b.mesh), static: b.kind === 'static' ? pose : null };
    if (b.kind !== 'static') moving.push(b);
  }
  data.ticks = t.rows.map(([row, n]) => ({ n, poses: Object.fromEntries(moving.map((b) => [b.id, at(row, b.index)])) }));
  if (t.tcp) {
    const tcp = ({ index, offset }, row) => {
      const p = at(row, index), q = new THREE.Quaternion(p[3], p[4], p[5], p[6]);
      return new THREE.Vector3(offset[0], offset[1], offset[2]).applyQuaternion(q).add(v3(p)).toArray();
    };
    const frames = t.rows.map(([row, n]) => ({ n, seg: 'trace', arms: Object.fromEntries(Object.entries(t.tcp).map(([arm, link]) => [arm, { pose: tcp(link, row) }])) }));
    data.frames = [...(data.frames || []).filter((f) => Array.isArray(f.base)).map((f) => ({ ...f, arms: {} })), ...frames];
  }
}
function loadTraceMesh(inst, info, holder) {
  gltfLoader.load(info.glb, (gltf) => {
    holder.add(gltf.scene);
    let c = null;
    gltf.scene.traverse((o) => { if (o.isMesh) c = o.geometry.getAttribute('color'); });
    if (!c || !modelObjects[inst]) return;
    const sum = [0, 0, 0];
    for (let i = 0; i < c.count; i++) { sum[0] += c.getX(i); sum[1] += c.getY(i); sum[2] += c.getZ(i); }
    modelObjects[inst].color = sum.map((x) => Math.round((255 * x) / Math.max(c.count, 1)));
  }, undefined, (e) => console.warn(e));
}
const linkHolders = {};
function buildModels() {
  clearGroup(groups.models); clearGroup(groups.labels); clearGroup(traceLinks);
  for (const k of Object.keys(modelObjects)) delete modelObjects[k];
  for (const k of Object.keys(linkHolders)) delete linkHolders[k];
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.75, metalness: 0.05 });
  const markMaterial = new THREE.MeshBasicMaterial({ color: dark ? 0xffffff : 0x111111 });
  for (const [inst, info] of Object.entries(S.data.instances || {})) {
    if (!info.mesh && !info.glb) continue;
    const holder = new THREE.Group();
    holder.visible = false;
    if (info.kind === 'robot') {  // part of the robot, not an object: no label, not in the Objects table
      traceLinks.add(holder); linkHolders[inst] = holder;
      loadTraceMesh(inst, info, holder);
      continue;
    }
    (info.glb && info.static ? groups.table : groups.models).add(holder);  // a trace's table is under the table toggle
    const l = label(info.static ? `${inst} · fixed` : inst);
    l.visible = false;
    groups.labels.add(l);
    modelObjects[inst] = { holder, label: l, info, color: null };
    if (info.static) {
      const p = info.static;
      holder.position.set(p[0], p[1], p[2]); holder.quaternion.set(p[3], p[4], p[5], p[6]).normalize();
      holder.visible = true; l.visible = true; l.position.set(p[0], p[1], p[2] + 0.075);
    }
    if (info.glb) { loadTraceMesh(inst, info, holder); continue; }
    loadMeshGeometry(info.mesh).then((g) => {
      const mesh = new THREE.Mesh(g, material);
      const s = info.scale || [1, 1, 1];
      mesh.scale.set(s[0], s[1], s[2]);
      holder.add(mesh);
      const c = g.getAttribute('color'), sum = [0, 0, 0];
      for (let i = 0; i < c.count; i++) { sum[0] += c.getX(i); sum[1] += c.getY(i); sum[2] += c.getZ(i); }
      modelObjects[inst].color = sum.map((x) => Math.round((255 * x) / Math.max(c.count, 1)));
      if (!info.static) {  // heading mark on the top face along the object's local +x, so a turn of a symmetric nut shows
        g.computeBoundingBox();
        const b = g.boundingBox, len = 0.85 * Math.max(b.max.x * s[0], 0.005);
        const bar = new THREE.Mesh(new THREE.BoxGeometry(len, 0.0025, 0.0015), markMaterial);
        bar.position.set(len / 2, 0, b.max.z * s[2] + 0.0008);
        holder.add(bar);
      }
    }).catch((e) => console.warn(e));
  }
}
// Heading of each moving object about the world vertical, unwrapped over the round (degrees from its first tick), so
// several full turns of a nut read as e.g. -297 rather than an angle folded into ±180 or the nut's 60° symmetry.
function buildTurns() {
  S.turn = {};
  const names = new Set(S.ticks.flatMap((t) => Object.keys(t.poses)));
  const x = new THREE.Vector3(), q = new THREE.Quaternion();
  for (const inst of names) {
    const out = new Float32Array(S.ticks.length).fill(NaN);
    let first = null, prev = null, acc = 0;
    S.ticks.forEach((t, i) => {
      const p = t.poses[inst];
      if (!p) return;
      x.set(1, 0, 0).applyQuaternion(q.set(p[3], p[4], p[5], p[6]));
      const yaw = Math.atan2(x.y, x.x) * 180 / Math.PI;
      if (prev === null) { first = yaw; acc = yaw; } else { let d = yaw - prev; d -= 360 * Math.round(d / 360); acc += d; }
      prev = yaw; out[i] = acc - first;
    });
    S.turn[inst] = out;
  }
}
function turnAt(inst, n) {
  const i = bisect(S.ticks, n), a = S.turn?.[inst];
  return a && i >= 0 && Number.isFinite(a[i]) ? a[i] : null;
}
function updateModels(smp) {
  for (const [inst, holder] of Object.entries(linkHolders)) {
    const p = smp?.poses?.[inst];
    holder.visible = !!p;
    if (p) { holder.position.copy(p.pos); holder.quaternion.copy(p.quat); }
  }
  for (const [inst, m] of Object.entries(modelObjects)) {
    if (m.info.static) continue;
    const p = smp?.poses?.[inst];
    m.holder.visible = m.label.visible = !!p;
    if (!p) continue;
    m.holder.position.copy(p.pos); m.holder.quaternion.copy(p.quat);
    m.label.position.set(p.pos.x, p.pos.y, p.pos.z + 0.045);
  }
}

// ---------- point cloud ----------
const SRGB_TO_LINEAR = new Float32Array(256).map((_, i) => { const c = i / 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); });
function getCloud(obs) {
  if (!S.clouds.has(obs.dir)) {
    S.clouds.set(obs.dir, fetch('api/cloud?' + new URLSearchParams({ id: RUN, obs: obs.dir })).then(async (r) => {
      if (!r.ok) throw new Error(`cloud ${obs.dir}: HTTP ${r.status}`);
      const buf = await r.arrayBuffer(), n = new DataView(buf).getUint32(0, true);
      const pos = new Float32Array(buf, 4, n * 3), rgb = new Uint8Array(buf, 4 + n * 12, n * 3), col = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) col[i] = SRGB_TO_LINEAR[rgb[i]];
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      return new THREE.Points(g, new THREE.PointsMaterial({ size: 0.006, vertexColors: true }));
    }));
  }
  return S.clouds.get(obs.dir);
}
async function showCloud(obs) {
  if (!obs || !Object.values(obs.cams).some((c) => c.depth) || !$('showCloud').checked) { groups.cloud.clear(); return; }
  try {
    const pts = await getCloud(obs);
    if (S.obs[S.obsIndex] !== obs) return;
    groups.cloud.clear();
    groups.cloud.add(pts);
  } catch (e) { console.warn(e); }
}

// ---------- command target ----------
function drawTarget(act) {
  clearGroup(groups.target);
  if (!act) return;
  if (act.base_target) {  // a base command's target pose on the floor
    const [x, y, yaw] = act.base_target;
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.19, 32), new THREE.MeshBasicMaterial({ color: BASE_COLOR, side: THREE.DoubleSide }));
    ring.position.set(x, y, 0.004);
    groups.target.add(ring);
    const tick = new THREE.Mesh(new THREE.PlaneGeometry(0.2, 0.025), new THREE.MeshBasicMaterial({ color: BASE_COLOR, side: THREE.DoubleSide }));
    tick.position.set(x + 0.1 * Math.cos(yaw), y + 0.1 * Math.sin(yaw), 0.004); tick.rotation.set(0, 0, yaw);
    groups.target.add(tick);
  }
  for (const [arm, a] of Object.entries(act.arms || {})) {
    if (!a.target || a.target.length < 3) continue;
    const p = v3(a.target);
    const m = sphere(0.009, ARM_COLOR[arm], { wireframe: true });
    m.position.copy(p);
    groups.target.add(m);
    if (a.target.length === 7) {
      const ax = new THREE.AxesHelper(0.05);
      ax.position.copy(p); ax.quaternion.set(a.target[3], a.target[4], a.target[5], a.target[6]).normalize();
      groups.target.add(ax);
      const tcp = tcpOf(a.target), s = sphere(0.005, ARM_COLOR[arm]);
      s.position.copy(tcp);
      groups.target.add(s);
    }
  }
}

// ---------- video ----------
const frameCache = new Map();
function setupVideo() {
  const v = S.data.video;
  const cams = new Set();
  for (const [, imgs] of v?.frames || []) for (const c of Object.keys(imgs)) cams.add(c);
  if (!cams.size) for (const o of S.obs) for (const c of Object.keys(o.cams)) cams.add(c);
  S.cams = [...cams].sort();
  if (v?.frames?.length) {
    S.video = v.frames.map(([n, imgs]) => ({ n, imgs }));
    $('videoLabel').textContent = `recorded execution, ${v.fps ?? '?'} fps plus every command boundary (${v.frames.length} frames)`;
  } else {
    S.video = S.obs.filter((o) => o.n !== null).map((o) => ({ n: o.n, imgs: Object.fromEntries(Object.entries(o.cams).map(([c, x]) => [c, x.rgb])) }));
    $('videoLabel').textContent = 'no execution capture: each observation\'s frames held from its time';
  }
  $('video').innerHTML = S.cams.map((c) => `<figure><img id="vid-${esc(c)}" alt="${esc(c)}"><figcaption>${esc(c)}</figcaption></figure>`).join('');
}
function cachedImage(url) {
  let img = frameCache.get(url);
  if (!img) { img = new Image(); img.src = url; frameCache.set(url, img); if (frameCache.size > 600) frameCache.delete(frameCache.keys().next().value); }
  return img;
}
function updateVideo(n) {
  const i = bisect(S.video, n);
  if (i < 0) return;
  for (let k = i + 1; k <= Math.min(i + 8, S.video.length - 1); k++) for (const c of S.cams) if (S.video[k].imgs[c]) cachedImage(fileUrl(S.video[k].imgs[c]));
  for (const c of S.cams) {
    const el = $(`vid-${c}`), p = S.video[i].imgs[c];
    if (!el || !p) continue;
    const url = fileUrl(p);
    if (el.dataset.want === url) continue;
    el.dataset.want = url;
    const img = cachedImage(url), show = () => { if (el.dataset.want === url) el.src = url; };
    if (img.complete && img.naturalWidth) show(); else img.addEventListener('load', show, { once: true });
  }
}

// ---------- command strip ----------
const actColor = (a) => {
  if (a.action === 'set_gripper') return MODE_COLOR.gripper;
  const modes = Object.values(a.arms || {}).map((x) => x.interaction).filter(Boolean);
  const flat = modes.flatMap((m) => String(m).split('+'));
  for (const m of ['contact', 'contact_approach', 'fine', 'precision', 'custom', 'move', 'fast', 'free_space']) if (flat.includes(m)) return MODE_COLOR[m];
  return ACTION_COLOR[a.action] || '#5d6670';
};
// a stalled gripper is a normal grasp; a command without status or failure code (LIBERO-Pro's set_gripper) did not fail
const actFailed = (a) => (a.failure_code ? a.failure_code !== 'gripper_stalled_before_target' : !!a.status && !OK_STATUS.has(a.status));
function drawStrip() {
  const cv = $('strip');
  if (!cv || !S.data) return;
  const w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const x = (n) => (S.maxN > 0 ? (n / S.maxN) * (w - 2) + 1 : 1);
  for (const a of S.acts) {
    if (a.n0 === null || a.n1 === null) continue;
    const x0 = x(a.n0), x1 = Math.max(x(a.n1), x0 + 2);
    ctx.fillStyle = actColor(a);
    ctx.fillRect(x0, 8, x1 - x0, h - 18);
    if (actFailed(a)) {
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1;
      for (let xx = x0; xx < x1; xx += 4) { ctx.beginPath(); ctx.moveTo(xx, h - 10); ctx.lineTo(xx + 4, 8); ctx.stroke(); }
    }
  }
  for (const e of S.entries) {  // agent tool calls; several at one native time stack into a taller tick
    ctx.fillStyle = e.error ? '#c0392b' : (e.imgs?.length ? '#1f6fd1' : (dark ? '#8b95a1' : '#6a7380'));
    ctx.fillRect(x(e.n0) - 0.5, h - 8, 1.5, 8);
  }
  ctx.fillStyle = dark ? '#dde1e6' : '#1d2127';
  for (const o of S.obs) if (o.n !== null) ctx.fillRect(x(o.n) - 0.5, 0, 1, 7);
  ctx.fillStyle = '#1f6fd1';
  ctx.fillRect(x(S.n) - 1, 0, 2, h);
}
$('strip').addEventListener('click', (e) => {
  const r = e.target.getBoundingClientRect();
  stop(); seek(((e.clientX - r.left) / r.width) * S.maxN);
});

// ---------- agent trace (Agent tab) and the agent's files (files tab) ----------
const modeChip = (c) => {
  const col = c.kind === 'set_gripper' ? MODE_COLOR.gripper : (MODE_COLOR[['contact', 'contact_approach', 'precision', 'free_space'].find((m) => (c.modes || []).includes(m))] || ACTION_COLOR[c.kind] || '#5d6670');
  const bad = actFailed({ status: c.status, failure_code: c.fc });
  return `<span class="ag-chip" style="background:${col}" title="${esc(c.id)}${c.refusal ? ' — ' + esc(c.refusal) : ''}">${esc(c.kind || 'cmd')}${c.modes?.length ? ' ' + esc(c.modes.join('+')) : ''}${bad ? ' ✗ ' + esc(c.fc || c.status) : ''}${c.refusal ? ' refused' : ''}</span>`;
};
function entryHtml(e) {
  const head = `<div class="ag-h"><a data-seek="${e.i}">${e.call !== undefined && e.call !== null ? 'call ' + e.call : '#' + (e.i + 1)} · ${esc(unitShort())} ${e.n0 === e.n1 ? fmtN(e.n0) : fmtN(e.n0) + '–' + fmtN(e.n1)}</a>
    <span class="muted">${esc(e.tool || 'reply')}${e.dur !== null && e.dur !== undefined ? ' · ' + e.dur + ' s' : ''}${e.exit !== undefined && e.exit !== null && e.exit !== 0 ? ' · exit ' + e.exit : ''}${e.tokens?.[0] ? ' · ' + Math.round(e.tokens[0] / 1000) + 'k in' : ''}</span>
    ${(e.cmds || []).map(modeChip).join('')}${e.finished_episode ? '<span class="ag-chip" style="background:#202124">finished episode</span>' : ''}</div>`;
  const think = e.think ? `<details class="ag-think"><summary>thinking · ${esc(e.think.slice(0, 140).replace(/\s+/g, ' '))}…</summary><pre>${esc(e.think)}</pre></details>` : '';
  const say = e.say?.trim() ? `<div class="ag-say">${esc(e.say.trim())}</div>` : '';
  const code = e.code ? `<pre class="ag-code">${esc(e.code)}</pre>` : '';
  const out = e.out ? `<details class="ag-out" open><summary>returned${e.out_chars ? ' · ' + e.out_chars + ' chars' : ''}${e.imgs?.length ? ' · ' + e.imgs.length + ' image(s)' : ''}</summary><pre>${esc(e.out)}</pre></details>` : '';
  const imgs = e.imgs?.length ? `<div class="ag-imgs">${e.imgs.map((m) => `<figure><a href="${esc(m.url)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(m.url)}" alt=""></a><figcaption title="${esc(m.label)}">${esc(m.label)}</figcaption></figure>`).join('')}</div>` : '';
  const ch = e.changes, wrote = ch ? [...(ch.added || []).map((p) => '+' + p), ...(ch.modified || []).map((p) => '~' + p)].filter((p) => !/outputs\//.test(p)) : [];
  const files = wrote.length ? `<div class="ag-files">wrote ${wrote.map((p) => `<a data-file="${esc(p.slice(1))}">${esc(p)}</a>`).join(' ')}</div>` : '';
  return `<div class="ag-e${e.error ? ' err' : ''}" id="ag-${e.i}" data-i="${e.i}">${head}${think}${say}${code}${out}${imgs}${files}</div>`;
}
function renderAgent() {
  const pane = $('pane-agent');
  if (!S.agent) { pane.innerHTML = '<p class="muted">no agent trace</p>'; return; }
  pane.innerHTML = S.entries.length ? S.entries.map(entryHtml).join('') : '<p class="muted">no tool calls yet</p>';
  S.entryIndex = -2;
}
function renderTools() {
  const files = S.agent?.files || [];
  $('toolsCount').textContent = files.length ? `(${files.length})` : '';
  if (!files.length) { $('pane-tools').innerHTML = `<p class="muted">${S.agent?.code_mode ? 'The agent wrote no files.' : 'Tool-call mode: the agent writes no files.'}</p>`; return; }
  $('pane-tools').innerHTML = `<p class="small muted">Files the agent wrote in its workspace, newest version's outline (module docstring and functions, as the agent wrote them).
    Click a version to see its source as of that call; "used by" lists later calls whose code names the module.</p>` + files.map((f) => {
    const o = f.outline;
    const defs = o?.defs?.length ? `<ul>${o.defs.map((d) => `<li><code>${esc(d.kind === 'class' ? 'class ' + d.name : d.name + '(' + d.sig + ')')}</code>${d.doc ? ' — ' + esc(d.doc) : ''}</li>`).join('')}</ul>` : '';
    const vers = f.versions.map((v, k) => `<a data-blob="${esc(v.sha || '')}" data-entry="${v.entry ?? ''}" title="call ${esc(v.call)}, ${esc(unitShort())} ${v.n ?? '?'}">v${k + 1}@call ${esc(v.call)}</a>`).join(' ');
    const used = f.used_by?.length ? `<div class="small">used by ${f.used_by.slice(0, 40).map((i) => `<a data-seek="${i}">call ${S.entries[i]?.call ?? '#' + (i + 1)}</a>`).join(' ')}${f.used_by.length > 40 ? ' …' : ''}</div>` : '<div class="small muted">not referenced by later calls</div>';
    return `<div class="tl-f" data-path="${esc(f.path)}"><h4>${esc(f.path)} <span class="muted small">${f.size ?? '?'} B${f.removed !== null && f.removed !== undefined ? ' · removed at call ' + f.removed : ''}</span></h4>
      ${o?.doc ? `<div class="tl-doc">${esc(o.doc)}</div>` : ''}${defs}<div class="small">versions: ${vers}</div>${used}<pre class="ag-code tl-src" hidden></pre></div>`;
  }).join('');
}
async function showBlob(a) {
  const box = a.closest('.tl-f').querySelector('.tl-src');
  if (!a.dataset.blob) { box.hidden = false; box.textContent = '(version not recorded)'; return; }
  if (box.dataset.sha === a.dataset.blob && !box.hidden) { box.hidden = true; return; }
  box.hidden = false; box.textContent = 'loading…'; box.dataset.sha = a.dataset.blob;
  const r = await fetch('api/blob?' + new URLSearchParams({ id: RUN, sha: a.dataset.blob }));
  box.textContent = r.ok ? await r.text() : `HTTP ${r.status}`;
}
function seekEntry(i) {
  const e = S.entries[i];
  if (!e) return;
  stop(); S.pin = i; S.n = Math.min(Math.max(e.n0, 0), S.maxN); update();
}
function jumpCall(delta) {
  if (!S.entries.length) return;
  const cur = S.entryIndex >= 0 ? S.entryIndex : -1;
  seekEntry(Math.min(Math.max(cur + delta, 0), S.entries.length - 1));
}
function currentEntry(n) {
  if (S.pin !== null && S.entries[S.pin] && Math.abs(S.entries[S.pin].n0 - n) < 1e-6) return S.pin;
  S.pin = null;
  let k = -1;
  for (const e of S.entries) { if (e.n0 <= n + 1e-6) k = e.i; else break; }
  return k;
}
function updateAgent(n) {
  const k = currentEntry(n);
  if (k === S.entryIndex) return;
  S.entryIndex = k;
  const pane = $('pane-agent');
  for (const el of pane.querySelectorAll('.ag-e')) {
    const i = Number(el.dataset.i);
    el.classList.toggle('on', i === k);
    el.classList.toggle('future', i > k);
  }
  const el = $(`ag-${k}`);
  if (el && $('follow').checked && !pane.hidden) pane.scrollTop = el.offsetTop - pane.offsetTop - 6;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem('ora.replay.' + k); return v === null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('ora.replay.' + k, String(v)); } catch (e) { /* storage blocked: not remembered */ } },
};
function dragSplit(el, onMove, onEnd) {
  el.addEventListener('pointerdown', (e) => {
    e.preventDefault(); el.setPointerCapture(e.pointerId); el.classList.add('drag');
    const move = (ev) => onMove(ev);
    const up = () => { el.classList.remove('drag'); el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); onEnd(); };
    el.addEventListener('pointermove', move); el.addEventListener('pointerup', up);
  });
}
const layoutEl = document.querySelector('.rp-layout');
const sideVar = () => 'side-w';
dragSplit($('rpSplit'), (ev) => {
  const r = layoutEl.getBoundingClientRect(), w = Math.min(Math.max(r.right - ev.clientX - 3, 320), r.width - 320);
  layoutEl.style.setProperty('--' + sideVar(), w + 'px');
}, () => { store.set(sideVar(), layoutEl.style.getPropertyValue('--' + sideVar())); resize(); });
{ const v = store.get('side-w', ''); if (v) layoutEl.style.setProperty('--side-w', v); }

function setTab(tab) {
  S.tab = tab;
  for (const b of document.querySelectorAll('.rp-tabs button')) b.classList.toggle('on', b.dataset.tab === tab);
  $('pane-agent').hidden = tab !== 'agent'; $('side').hidden = tab !== 'state'; $('pane-tools').hidden = tab !== 'tools';
  if (tab === 'agent') { S.entryIndex = -2; updateAgent(S.n); }
  resize();
}
for (const b of document.querySelectorAll('.rp-tabs button')) b.addEventListener('click', () => setTab(b.dataset.tab));
document.querySelector('.rp-side').addEventListener('click', (ev) => {
  const a = ev.target.closest('a');
  if (!a) return;
  if (a.dataset.seek !== undefined) { ev.preventDefault(); setTab('agent'); seekEntry(Number(a.dataset.seek)); }
  else if (a.dataset.file !== undefined) {
    ev.preventDefault(); setTab('tools');
    const box = [...document.querySelectorAll('.tl-f')].find((x) => x.dataset.path === a.dataset.file);
    if (box) { $('pane-tools').scrollTop = box.offsetTop - $('pane-tools').offsetTop; box.querySelector('a[data-blob]:last-of-type')?.click(); }
  } else if (a.dataset.blob !== undefined) { ev.preventDefault(); showBlob(a); }
});
$('prevCall').addEventListener('click', () => jumpCall(-1));
$('nextCall').addEventListener('click', () => jumpCall(1));

// ---------- side panel ----------
function eulerDeg(q) {
  const e = new THREE.Euler().setFromQuaternion(q, 'ZYX');
  return [e.x, e.y, e.z].map((r) => (r * 180 / Math.PI).toFixed(1)).join(', ');
}
function renderSide(smp) {
  const act = S.acts[S.actIndex], obs = S.obs[S.obsIndex];
  let h = '';
  if (act) {
    const arms = Object.entries(act.arms || {}).map(([arm, a]) => `<tr><td>${arm}</td><td>${esc(a.interaction ?? a.gripper ?? '')}</td>
      <td>${a.target ? a.target.slice(0, 3).map((v) => fmt(v, 3)).join(', ') : '—'}</td><td>${esc(a.outcome ?? '')}</td></tr>`).join('');
    const bt = act.base_target;
    h += `<h3>Command ${act.index + 1}/${S.acts.length} <span class="rp-chip" style="background:${actColor(act)};color:#fff">${esc(act.action)}</span></h3>
      <div class="small">${esc(unitShort())} ${fmtN(act.n0)}–${fmtN(act.n1)}${act.native_actions !== undefined && act.native_actions !== null ? ` (${act.native_actions} native)` : ''} · ${esc(act.status ?? 'no status')}${act.failure_code ? ` · <span class="rp-warn">${esc(act.failure_code)}</span>` : ''}${act.terminal ? ' · <span class="rp-warn">terminal</span>' : ''}</div>
      ${arms ? `<table><thead><tr><th>arm</th><th>mode</th><th>target xyz (${esc(S.data.pose_point || 'TCP')})</th><th>outcome</th></tr></thead><tbody>${arms}</tbody></table>` : ''}
      ${bt ? `<div class="small">base target x ${fmt(bt[0], 3)}, y ${fmt(bt[1], 3)}, yaw ${fmt(bt[2] * 180 / Math.PI, 1)}°</div>` : ''}
      ${Object.entries(act.arms || {}).filter(([, a]) => a.legs).map(([arm, a]) => `<table><thead><tr><th>${esc(arm)} leg</th><th>mode</th><th>status</th><th>ticks</th><th>moved mm</th><th>peak m/s</th><th>held</th></tr></thead><tbody>${a.legs.map((l, i) => `<tr${l.status !== 'reached' && l.status !== 'skipped' ? ' class="rp-warn"' : ''} title="${esc(l.why || '')}"><td>${i}</td><td>${esc(l.mode)}</td><td>${esc(l.status)}${l.stop ? ' ' + esc(l.stop) : ''}</td><td>${l.ticks ?? ''}</td><td>${l.moved_mm ?? ''}</td><td>${l.peak_speed_mps ?? ''}</td><td>${l.held ?? ''}</td></tr>`).join('')}</tbody></table>`).join('')}
      <details><summary class="small">arguments</summary><pre>${esc(act.arguments)}</pre></details>`;
  } else {
    h += '<h3>No robot command yet</h3>';
  }
  if (obs) {
    const imgs = Object.entries(obs.cams).map(([c, x]) => `<figure><a href="${fileUrl(x.rgb)}" target="_blank" rel="noopener"><img src="${fileUrl(x.rgb)}" alt="${c}"></a><figcaption>${c}</figcaption></figure>`).join('');
    h += `<h3>Observation ${obs.seq} <span class="small muted">at ${esc(unitShort())} ${fmtN(obs.n)} · what the agent saw</span></h3><div class="rp-obs">${imgs}</div>`;
  }
  const rows = [], fixed = [];
  const swatch = (m) => (m.color ? `<span title="mesh colour" style="display:inline-block;width:9px;height:9px;margin-right:4px;border:1px solid #888;background:rgb(${m.color.join(',')})"></span>` : '');
  for (const [inst, m] of Object.entries(modelObjects)) {
    if (m.info.static) {
      const p = m.info.static;
      fixed.push(`<tr title="${esc(FIXED_NOTE)}"><td>${swatch(m)}${esc(inst)}</td><td>${p.slice(0, 3).map((v) => (v * 1000).toFixed(0)).join(' ')}</td>
        <td>${eulerDeg(new THREE.Quaternion(p[3], p[4], p[5], p[6]))}</td><td colspan="3" class="muted">fixed (static collider)</td></tr>`);
      continue;
    }
    if (!smp?.poses?.[inst]) continue;
    const p = smp.poses[inst], s0 = S.start[inst];
    let rise = '', tilt = '';
    if (s0) {
      rise = ((p.pos.z - s0.pos.z) * 1000).toFixed(1);
      const up = new THREE.Vector3(0, 0, 1);
      tilt = (up.clone().applyQuaternion(p.quat).angleTo(up.clone().applyQuaternion(s0.quat)) * 180 / Math.PI).toFixed(1);
    }
    const turn = turnAt(inst, S.n);
    const warn = Math.abs(Number(rise)) > 5 || Number(tilt) > 15;
    rows.push(`<tr${warn ? ' class="rp-warn"' : ''}><td>${swatch(m)}${esc(inst)}</td><td>${[p.pos.x, p.pos.y, p.pos.z].map((v) => (v * 1000).toFixed(0)).join(' ')}</td>
      <td>${eulerDeg(p.quat)}</td><td>${turn === null ? '' : turn.toFixed(0)}</td><td>${rise}</td><td>${tilt}</td></tr>`);
  }
  if (rows.length || fixed.length) {
    h += `<h3>Objects (ground truth, developer only)</h3><table><thead><tr><th>instance</th><th>x y z mm</th><th>roll, pitch, yaw °</th>
      <th title="turn about the vertical since the round start, unwrapped (several turns add up); the short bar on the object's top marks its heading">turned °</th>
      <th title="z now minus z at the start">rise mm</th><th title="angle of the object's up axis from its start">tilt °</th></tr></thead><tbody>${rows.join('')}${fixed.join('')}</tbody></table>
      ${fixed.length ? `<div class="small muted">"fixed": ${esc(FIXED_NOTE)}.</div>` : ''}`;
  }
  $('side').innerHTML = h;
  $('cmdNow').innerHTML = act ? `command ${act.index + 1}/${S.acts.length}: <b>${esc(act.action)}</b> ${Object.entries(act.arms || {}).map(([arm, a]) => esc(arm + ' ' + (a.interaction ?? a.gripper ?? ''))).join(', ')} · ${esc(act.status ?? 'no status')}${act.failure_code ? ` <span class="rp-warn">${esc(act.failure_code)}</span>` : ''}${act.terminal ? ' <span class="rp-warn">terminal</span>' : ''} · ${esc(unitShort())} ${fmtN(act.n0)}–${fmtN(act.n1)}` : 'no robot command yet';
}

// ---------- update ----------
function update(force = false) {
  const n = S.n;
  $('tick').value = String(Math.round(n));
  // the command executing now, else the last one that started at or before now
  let actIndex = S.acts.findIndex((a) => a.n0 !== null && n > a.n0 && n <= a.n1 + 1e-6);
  if (actIndex < 0) S.acts.forEach((a, i) => { if (a.n0 !== null && a.n0 <= n + 1e-6) actIndex = i; });
  const obsIndex = bisect(S.obs, n);
  const unit = S.data.clock?.unit || 'step';
  $('tickLabel').textContent = `${unit} ${n.toFixed(unit === 'native action' ? 1 : 0)}/${fmtN(S.maxN)}` + (S.hz ? ` · ${(n / S.hz).toFixed(2)} s sim` : '');
  const arms = sampleArms(n);
  if (arms) {
    const errs = [], spec = S.robotSpec;
    for (const arm of ARMS) {
      const a = arms[arm], robot = S.robots[arm];
      if (!a) continue;
      if (a.pose && tcpMarkers[arm]) tcpMarkers[arm].position.copy(a.tcp || tcpOf(a.pose));
      if (!robot || !spec) continue;
      // the check: the URDF's FK of the latest logged frame's joints against that frame's logged pose
      const link = robot.links?.[spec.fk_link?.[arm]];
      if (a.pose && a.logged?.q && link) {
        setArm(robot, a.logged, arm);
        robot.updateMatrixWorld(true);
        const off = spec.fk_offset || [0, 0, 0];
        const fk = new THREE.Vector3(off[0], off[1], off[2]).applyMatrix4(link.matrixWorld);
        errs.push(`${arm[0].toUpperCase()} ${(fk.distanceTo(v3(a.pose)) * 1000).toFixed(1)}`);
      }
      setArm(robot, a, arm);
    }
    $('fkCheck').textContent = errs.length ? `${spec.fk_label || 'URDF FK vs logged pose'} (mm): ${errs.join(', ')}` : '';
  }
  const base = sampleBase(n);
  if (base && baseMarker) { baseMarker.position.set(base[0], base[1], 0.006); baseMarker.rotation.set(0, 0, base[2]); }
  colorTrajectories(n);
  const smp = samplePoses(n);
  updateModels(smp);
  updateVideo(n);
  if (force || actIndex !== S.actIndex) { S.actIndex = actIndex; drawTarget(S.acts[actIndex]); }
  if (force || obsIndex !== S.obsIndex) { S.obsIndex = obsIndex; showCloud(S.obs[obsIndex]); }
  renderSide(smp);
  updateAgent(n);
  drawStrip();
  try { history.replaceState(null, '', `?run=${encodeURIComponent(RUN)}${ROUND ? '&round=' + ROUND : ''}&n=${Math.round(n)}${S.tab && S.tab !== 'agent' ? '&tab=' + S.tab : ''}`); } catch (e) { /* ignore */ }
}

class Refused extends Error {}
async function fetchBundle() {
  const r = await fetch('api/replay?' + new URLSearchParams({ id: RUN, round: ROUND }), { cache: 'no-store' });
  const data = await r.json();
  if (r.status === 422) throw new Refused(data.error || 'this run has no 3D replay');
  if (!r.ok) throw new Error(data.error || r.status);
  return data;
}
async function load(n, data) {
  stop();
  S.data = data || await fetchBundle();
  data = S.data;
  try { await applySceneTrace(data); } catch (e) { showNotice(`The scene trace did not load (${e?.message || e}); no objects.`); }
  const rounds = data.rounds || [0];
  $('rounds').innerHTML = rounds.length > 1 ? 'round: ' + rounds.map((r) => r === ROUND ? `<b>${r + 1}</b>`
    : `<a href="?run=${encodeURIComponent(RUN)}&round=${r}${S.tab && S.tab !== 'agent' ? '&tab=' + S.tab : ''}">${r + 1}</a>`).join(' ') +
    ' <span class="muted">(each round starts from the simulator\'s reset)</span>' : '';
  // the clock: its rate when the run records one (seconds in the label), and the playback rate at 1x
  S.hz = data.clock?.hz || null;
  S.rate = S.hz || 20;
  S.frames = (data.frames || []).filter((f) => f.arms && Object.keys(f.arms).length);
  S.baseFrames = (data.frames || []).filter((f) => Array.isArray(f.base));
  S.ticks = (data.ticks || []).filter((t) => typeof t.n === 'number');
  S.acts = data.actions || [];
  S.obs = (data.observations || []).filter((o) => o.n !== null);
  S.maxN = Math.max(0, data.max_n || 0, ...S.frames.map((f) => f.n), ...S.ticks.map((t) => t.n));
  buildTurns();
  S.start = {};
  const first = samplePoses(0);
  if (first) S.start = first.poses;
  $('tick').max = String(Math.ceil(S.maxN));
  $('tick').setAttribute('aria-label', data.clock?.unit || 'step');
  $('clockNote').textContent = data.clock ? `clock: ${data.clock.unit}${data.clock.hz ? ` (${data.clock.hz} Hz)` : ', rate not recorded (1x plays 20 per second)'}; ${data.clock.source}` : '';
  buildTable(); buildTrajectories(); buildModels(); buildBase(); setupVideo();
  const instances = Object.values(data.instances || {});
  const objects = instances.some((i) => (i.mesh || i.glb) && i.kind !== 'robot');
  setLayer('showModels', objects, data.objects_note);
  setLayer('showLabels', objects, null);
  setLayer('showTable', typeof data.table_height === 'number' || instances.some((i) => i.glb && i.static), 'no fixed table in this backend');
  setLayer('showRobot', !!Object.keys(S.robots).length || instances.some((i) => i.kind === 'robot'), data.robot_note || 'no robot model');
  loadAgent();
  if (!data.ground_truth) $('side').innerHTML = '<p class="muted">No ground-truth capture for this run.</p>';
  S.n = Math.min(Math.max(n ?? 0, 0), S.maxN);
  S.actIndex = S.obsIndex = -2;
  update(true);
}

async function loadAgent() {
  try {
    const r = await fetch('api/replay/agent?' + new URLSearchParams({ id: RUN, round: ROUND }), { cache: 'no-store' });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.status);
    S.agent = data; S.entries = data.entries || [];
  } catch (e) { S.agent = null; S.entries = []; $('pane-agent').innerHTML = `<p class="rp-warn">agent trace: ${esc(String(e))}</p>`; return; }
  renderAgent(); renderTools(); updateAgent(S.n); drawStrip();
}

// ---------- controls ----------
function stop() { S.playing = false; $('play').textContent = 'Play'; }
function play() { if (S.n >= S.maxN) S.n = 0; S.playing = true; S.lastTs = null; $('play').textContent = 'Pause'; }
function seek(n) { S.n = Math.min(Math.max(n, 0), S.maxN); update(); }
function jumpAct(delta) {
  const starts = S.acts.filter((a) => a.n0 !== null).map((a) => a.n0);
  const target = delta > 0 ? starts.find((s) => s > S.n + 1e-6) : [...starts].reverse().find((s) => s < S.n - 1e-6);
  if (target !== undefined) { stop(); seek(target); }
}
$('play').addEventListener('click', () => (S.playing ? stop() : play()));
$('prevAct').addEventListener('click', () => jumpAct(-1));
$('nextAct').addEventListener('click', () => jumpAct(1));
$('tick').addEventListener('input', (e) => { stop(); seek(Number(e.target.value)); });
$('reload').addEventListener('click', () => { S.clouds.clear(); load(S.n).catch(showError); });
const toggles = { showRobot: 'robot', showCloud: 'cloud', showModels: 'models', showLabels: 'labels', showTraj: 'traj', showTarget: 'target', showTable: 'table' };
for (const [id, g] of Object.entries(toggles)) {
  const apply = () => { groups[g].visible = $(id).checked; if (id === 'showTraj') groups.base.visible = $(id).checked; };
  $(id).addEventListener('change', () => { apply(); if (id === 'showCloud') showCloud(S.obs[S.obsIndex]); });
  apply();
}
// a layer this run cannot show: its toggle disabled, with the reason beside it
function setLayer(id, available, note) {
  const box = $(id), lab = box.closest('label');
  box.disabled = !available;
  lab.classList.toggle('muted', !available);
  lab.title = available ? '' : (note || '');
  const hint = lab.querySelector('.rp-why');
  if (hint) hint.textContent = available ? '' : ` — ${note || 'not available'}`;
}
document.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) && e.target.type !== 'range') return;
  if (e.key === ' ') { e.preventDefault(); S.playing ? stop() : play(); }
  else if (e.key === 'ArrowRight' && !e.shiftKey) { e.preventDefault(); stop(); seek(Math.floor(S.n) + 1); }
  else if (e.key === 'ArrowLeft' && !e.shiftKey) { e.preventDefault(); stop(); seek(Math.ceil(S.n) - 1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); jumpAct(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); jumpAct(-1); }
  else if (e.key === ']') { e.preventDefault(); jumpCall(1); }
  else if (e.key === '[') { e.preventDefault(); jumpCall(-1); }
});
renderer.setAnimationLoop((ts) => {
  if (S.playing) {
    if (S.lastTs !== null) {
      S.n += ((ts - S.lastTs) / 1000) * S.rate * Number($('speed').value);
      if (S.n >= S.maxN) { S.n = S.maxN; stop(); }
      update();
    }
    S.lastTs = ts;
  }
  controls.update();
  renderer.render(scene, camera);
});
function showError(e) { $('side').innerHTML = $('pane-agent').innerHTML = `<p class="rp-warn">${esc(String(e))}</p>`; console.error(e); }

function showNotice(text) {
  const el = $('notice');
  el.textContent = text; el.hidden = false;
}
function place(obj, pose) { obj.position.set(...pose.pos); obj.quaternion.set(...pose.quat).normalize(); }
async function boot() {
  if (!RUN) { showError('no run given (?run=<run_id>)'); return; }
  resize();
  let data;
  try { data = await fetchBundle(); } catch (e) {
    if (e instanceof Refused) {   // a backend without a 3D replay (Synthetic): nothing to draw
      showNotice(String(e.message));
      document.querySelector('.rp-layout').hidden = true;
      for (const el of document.querySelectorAll('.rp-bar, .rp-strip')) el.hidden = true;
      return;
    }
    throw e;
  }
  document.title = `ORA 3D replay · ${data.backend_name || data.backend || ''}`;
  if (data.view) { camera.position.set(...data.view.eye); controls.target.set(...data.view.target); controls.update(); }
  buildTcpMarkers();
  const spec = data.robot;
  if (spec) {
    try {
      if (spec.kind === 'per_arm') {   // RoboDojo: one URDF per arm at its mount
        const [left, right] = await Promise.all([loadRobot(spec), loadRobot(spec)]);
        place(left, spec.mounts.left); place(right, spec.mounts.right);
        S.robots = { left, right };
        groups.robot.add(left, right);
      } else {                         // RoboTwin: one URDF holding both arms at the embodiment's root pose
        const robot = await loadRobot(spec);
        place(robot, spec.root);
        S.robots = { left: robot, right: robot };
        groups.robot.add(robot);
      }
      S.robotSpec = spec;
    } catch (e) { showNotice('The robot model did not load (' + String(e?.message || e) + '); TCP markers and paths only.'); }
  } else if (data.robot_note && data.robot_note_level === 'notice') {
    showNotice(data.robot_note);
  }
  if (['agent', 'state', 'tools'].includes(params.get('tab'))) setTab(params.get('tab'));
  await load(Number(params.get('n') || 0), data);
}
boot().catch(showError);
