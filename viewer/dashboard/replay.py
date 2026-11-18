"""3D replay data for one ORA run of any backend: the robot, the agent's RGB-D, the robot commands, the camera
frames and the agent's trace on one clock.

Ported from Shaoxiong Duan's code-vs-tools dashboard (snapshot 2026-10-31), which replayed RoboDojo runs, and extended
to give every backend the same bundle. Nothing here is shown to the agent, and nothing is written into a run: point
clouds and meshes go to the configured cache.

The clock, per backend (``clock`` in the bundle names its unit and rate):
  RoboDojo          native actions since the round reset (one native action = 10 physics steps = 0.04 s), from
                    native/controller/ and native/observation-clocks.jsonl
  RoboTwin,         the execution capture's simulator steps since the round reset (RoboTwin: SAPIEN steps at 250 Hz,
  LIBERO-Pro        LIBERO-Pro: MuJoCo steps at 500 Hz), each command spanning its action_start..action_end frames
  RoboCasa365,      each receipt's steps (steps_used, else native_actions, else the controller's physics steps)
  RoboLab           accumulated in action order within a round: native actions, at a rate the run does not record

Sources (all written by ORA during the run):
  native/receipts/action-NNNN.json     each robot command: action, arguments, status, failure code, and (all but
                                       RoboDojo) the measured end-effector pose before and after it
  policy-observations/*/               the policy's RGB-D frames, calibration and proprioception at each observation;
                                       observation.json names the receipt it follows (previous_execution_receipt)
  native/execution-capture/            original camera frames at 10 fps plus every action boundary (RoboDojo,
                                       RoboTwin, LIBERO-Pro)
  native/controller/action-NNNN.json   RoboDojo: measured arm and finger joints and ee_link pose per native action
  native/controller-replay/*.npz       RoboTwin: measured arm joints and TCP pose per SAPIEN step (``global__``
                                       prefixed arrays first when a command ran a global then a local phase), and
                                       the commanded gripper per step of a set_gripper
  native/observation-clocks.jsonl      RoboDojo: the physics clock of each observation
  ../../ground-truth/<run>/            RoboDojo, developer-only: object poses after every native action
                                       (ticks.jsonl) and the scene's instances (physics/round-0000.json), present only
                                       for runs launched with ``runtime.ground_truth_capture``
  ../../scene-trace/<run>/             RoboTwin, LIBERO-Pro, RoboCasa365, RoboLab and RoboDojo without ground truth,
                                       developer-only: every body's mesh and pose, written by ``ora replay RUN --scene``
                                       (``scene_trace.py``; ``--scene-traces`` moves it)

This module builds RoboDojo's bundle and holds the readers, point clouds and meshes; ``replay_receipts.py`` builds the
other backends' bundle in the same shape. The robots the page poses are described in ``robots.py`` (RoboDojo's X5,
RoboTwin's aloha-agilex). Point clouds and the RoboTwin controller replay need numpy (and Pillow for the clouds), mesh
conversion needs usd-core (``meshes.py``, run in a subprocess); all are imported lazily.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import struct
import subprocess
import threading

from viewer.dashboard import robots
from viewer.dashboard.config import Config
from viewer.dashboard.runs import BACKEND_NAMES, plan_info
from viewer.dashboard.store import memo, stat_sig

PHYSICS_STEPS_PER_NATIVE = 10
NATIVE_HZ = 25.0
TABLE_HEIGHT = 0.765
WORKSPACE = ([-0.7, -0.6, 0.735], [0.7, 0.5, 1.465])
#: RoboDojo's logged pose is ee_link (= link6); its TCP, the centre of the inner finger pads, sits here in link6.
TCP_IN_LINK6 = [0.15007, 0.0, -0.00144]
ROBODOJO_VIEW = {"eye": [0.0, -1.25, 1.45], "target": [0.0, -0.15, 0.82]}
CLOUD_MAX_POINTS = 40000
# v2: the crop box is part of the cache key (every backend but RoboDojo crops to its own end-effector paths)
CLOUD_VERSION = 2
# v2: positions in the asset root's frame without the root's own xform (factory_bolt sat 26.25 mm too low), colour
# from the material's constant diffuse colour (nuts and bolts were all grey); see meshes.py
MESH_VERSION = 2
SUPPORTED_BACKENDS = ("robodojo", "robotwin", "libero_pro", "robocasa", "robolab")
HERE = os.path.dirname(os.path.abspath(__file__))

_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}


def _key_lock(key: str) -> threading.Lock:
    with _lock:
        return _key_locks.setdefault(key, threading.Lock())


def _json(path: str):
    try:
        with open(path, "rb") as f:
            return json.loads(f.read())
    except (OSError, ValueError):
        return None


def _jsonl(path: str) -> list:
    rows = []
    try:
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        break           # partially written last line of a live run
    except OSError:
        pass
    return rows


def _r(xs, d=5):
    return [round(float(x), d) for x in xs] if isinstance(xs, (list, tuple)) else None


def ground_truth_dir(run_dir: str) -> str:
    run_dir = run_dir.rstrip("/")
    return os.path.join(os.path.dirname(os.path.dirname(run_dir)), "ground-truth", os.path.basename(run_dir))


def _arm_state(ee):
    """One arm's measured state -> {q, f, pose} (pose = ee_link = link6, xyz + quat xyzw)."""
    if not isinstance(ee, dict):
        return None
    return {"q": _r(ee.get("arm_joint_qpos")), "f": _r(ee.get("gripper_finger_qpos")),
            "pose": _r(ee.get("pose_xyz_xyzw")), "g": ee.get("gripper_command_target")}


def _controller_frames(path: str) -> dict:
    d = _json(path) or {}
    out = []
    for s in d.get("samples") or []:
        m = s.get("measured") or {}
        out.append({"n": s.get("native_action"), "seg": s.get("segment"),
                    "arms": {a: _arm_state(m.get(a)) for a in ("left", "right") if m.get(a)}})
    return {"action_id": d.get("action_id"), "frames": out}


def _cameras(obs_dir: str) -> list[str]:
    try:
        return sorted(c for c in os.listdir(obs_dir) if os.path.isfile(os.path.join(obs_dir, c, "rgb.png")))
    except OSError:
        return []


def _observations(run_dir: str, start_clock):
    clocks = {}
    for row in _jsonl(os.path.join(run_dir, "native", "observation-clocks.jsonl")):
        if isinstance(row.get("observation_sequence"), int):
            clocks[row["observation_sequence"]] = row.get("after_physics_steps")
    out = []
    for d in glob.glob(os.path.join(glob.escape(run_dir), "policy-observations", "*")):
        m = re.search(r"observation-(\d+)$", d)
        if not m:
            continue
        seq = int(m.group(1))
        clock = clocks.get(seq)
        n = None
        if isinstance(clock, int) and isinstance(start_clock, int):
            n = (clock - start_clock) / PHYSICS_STEPS_PER_NATIVE
        cams = {cam: {"rgb": os.path.join(d, cam, "rgb.png"),
                      "depth": os.path.isfile(os.path.join(d, cam, "depth-m.npy"))
                      and os.path.isfile(os.path.join(d, cam, "calibration.json"))}
                for cam in _cameras(d)}
        out.append({"seq": seq, "dir": os.path.basename(d), "n": n, "cams": cams})
    out.sort(key=lambda o: o["seq"])
    return out


def _capture_rows(run_dir: str) -> list:
    path = os.path.join(run_dir, "native", "execution-capture", "timeline.jsonl")
    return memo(("replay-capture", path), stat_sig(path), lambda: _jsonl(path))


def _video(run_dir: str, round_index: int = 0, per_unit: float = PHYSICS_STEPS_PER_NATIVE) -> dict:
    """Execution-capture frames of one round as [clock, {camera: path}, phase]; the clock is the capture's sim_step
    divided by ``per_unit`` (RoboDojo: 10 physics steps per native action; elsewhere the clock is the sim step)."""
    root = os.path.join(run_dir, "native", "execution-capture")
    frames = []
    for r in _capture_rows(run_dir):
        if (r.get("round_index") or 0) != round_index:
            continue
        imgs = {cam: os.path.join(root, v["path"]) for cam, v in (r.get("images") or {}).items()
                if isinstance(v, dict) and v.get("path")}
        if imgs and isinstance(r.get("sim_step"), (int, float)):
            frames.append([r["sim_step"] / per_unit, imgs, r.get("phase")])
    manifest = _json(os.path.join(root, "manifest.json")) or {}
    return {"frames": frames, "fps": (manifest.get("config") or {}).get("fps"), "status": manifest.get("status")}


def _pose7(p):
    return _r(p) if isinstance(p, (list, tuple)) and len(p) == 7 else None


def _base3(side) -> list | None:
    p = (side or {}).get("pose_xy_yaw") if isinstance(side, dict) else None
    return _r(p) if isinstance(p, (list, tuple)) and len(p) == 3 else None


def _first_int(*values) -> int | None:
    return next((int(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)), None)


def _receipt(path: str) -> dict:
    """One command of any backend. ``_poses`` (taken out before the page sees the command) holds what the bundle
    builds frames and the clock from: the measured end-effector poses before and after, the base pose (RoboCasa365),
    the command's steps and its round."""
    r = _json(path) or {}
    args = r.get("arguments") or {}
    targets = args.get("targets") if isinstance(args.get("targets"), dict) else {}
    interactions = args.get("interactions") if isinstance(args.get("interactions"), dict) else {}
    arms = {}
    for a, info in (r.get("arms") or {}).items():
        if info and info.get("targeted"):
            tgt = targets.get(a)
            if tgt is None and args.get("arm") == a:     # LIBERO-Pro set_gripper: {"arm": ..., "gripper": ...}
                tgt = args.get("gripper")
            pose = None
            if isinstance(tgt, dict) and tgt.get("position_m"):
                pose = _r(list(tgt["position_m"]) + list(tgt.get("orientation_xyzw") or []))
            mode = info.get("interaction") or interactions.get(a) or (tgt.get("interaction") if isinstance(tgt, dict) else None)
            arms[a] = {"interaction": mode, "target": pose,
                       "gripper": tgt if isinstance(tgt, (str, int, float)) else None,
                       "outcome": (info.get("motion_progress") or {}).get("outcome")}
    for path in r.get("paths") or (r.get("detail") or {}).get("paths") or []:   # move_path: one leg list per arm
        legs = [leg for leg in path.get("legs") or [] if isinstance(leg, dict)]
        if not path.get("arm") or not legs:
            continue
        ran = [leg for leg in legs if leg.get("status") != "skipped"] or legs
        last = ran[-1]
        tgt = last.get("target") if isinstance(last.get("target"), dict) else {}
        pose = _r(list(tgt["position_m"]) + list(tgt.get("orientation_xyzw") or [])) if tgt.get("position_m") else None
        modes = []
        for leg in ran:
            if leg.get("mode") and leg.get("mode") not in modes:
                modes.append(leg["mode"])
        arms[path["arm"]] = {"interaction": "+".join(modes) or None, "target": pose,
                             "gripper": (last.get("gripper") or {}).get("state") if isinstance(last.get("gripper"), dict) else None,
                             "outcome": last.get("stop") or last.get("status"),
                             "legs": [{"mode": leg.get("mode"), "status": leg.get("status"), "stop": leg.get("stop"),
                                       "ticks": leg.get("ticks"), "moved_mm": leg.get("moved_mm"),
                                       "peak_speed_mps": leg.get("peak_speed_mps"), "held": leg.get("held"),
                                       "why": leg.get("why")} for leg in legs]}
    text = json.dumps(args, ensure_ascii=False)
    base_target = args.get("target") if isinstance(args.get("target"), dict) else {}
    side = lambda key: {a: p for a, v in (r.get(key) or {}).items()   # noqa: E731
                        if isinstance(v, dict) and (p := _pose7(v.get("pose_xyz_xyzw")))}
    tracking = r.get("controller_tracking") if isinstance(r.get("controller_tracking"), dict) else {}
    return {"action_id": r.get("action_id"), "action": r.get("action"), "status": r.get("execution_status"),
            "failure_code": r.get("failure_code"), "native_actions": r.get("native_actions"),
            "terminal": r.get("terminal_after_action"), "arms": arms,
            "base_target": _r([base_target.get("x_m"), base_target.get("y_m"), base_target.get("yaw_rad")])
            if all(isinstance(base_target.get(k), (int, float)) for k in ("x_m", "y_m", "yaw_rad")) else None,
            "arguments": text if len(text) < 1500 else text[:1500] + " ...",
            "_poses": {"before": side("before"), "after": side("after"), "base_before": _base3(r.get("base_before")),
                       "base_after": _base3(r.get("base_after")), "round": _first_int(r.get("round_index")),
                       "steps": _first_int(r.get("steps_used"), r.get("native_actions"),
                                           tracking.get("physics_step_attempts")),
                       "finished": r.get("finished_unix_s")}}


def local_asset(usd_path: str, object_root: str | None) -> str | None:
    """The recorded asset path rebased onto this host's RoboDojo ``Object/`` directory, if it exists there."""
    if not object_root or not usd_path.endswith("/object.usdz") or "/Object/" not in usd_path or ".." in usd_path:
        return None
    local = os.path.join(object_root, usd_path.split("/Object/", 1)[1])
    return local if os.path.isfile(local) else None


def _instances(gt_dir: str, round_index: int, object_root: str | None) -> dict:
    phys = _json(os.path.join(gt_dir, "physics", f"round-{round_index:04d}.json")) or {}
    out = {}
    for inst, info in (phys.get("instances") or {}).items():
        asset = info.get("asset") or {}
        static = info.get("static_world_pose") or None
        out[inst] = {"kind": info.get("kind"), "category": asset.get("category"),
                     "mesh": local_asset(asset.get("usd_path") or "", object_root),
                     "scale": asset.get("scale"), "size": (info.get("configured_physics") or {}).get("size"),
                     "static": None if not static else _r(list(static["position_m"]) + _wxyz_to_xyzw(static["orientation_wxyz"]))}
    return out


def _wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]


def bundle(run_dir: str, cfg: Config, round_index: int = 0) -> dict:
    """Everything the 3D page needs for one round of the run (a multi-round session resets the world and the clock
    every round, so rounds are shown one at a time), rebuilt piecewise when its source files change. The server
    refuses a backend outside ``SUPPORTED_BACKENDS`` before it calls this."""
    backend = plan_info(run_dir, cfg).get("backend")
    if backend == "robodojo":
        return _robodojo_bundle(run_dir, cfg, round_index)
    from viewer.dashboard import replay_receipts    # it builds on this module's readers
    return replay_receipts.bundle(run_dir, cfg, round_index, backend)


def run_workspace(run_dir: str, cfg: Config) -> tuple | None:
    """The point clouds' crop box: RoboDojo's fixed workspace, elsewhere ``replay_receipts.workspace``."""
    if plan_info(run_dir, cfg).get("backend") == "robodojo":
        return WORKSPACE
    from viewer.dashboard import replay_receipts
    return replay_receipts.workspace(run_dir)


def _trace_rows(run_dir: str, trace: dict) -> list:
    """A scene trace's rounds and actions as ground-truth rows without poses: a run with no capture learns from them
    which round each action belongs to, its span and each round's physics clock at reset."""
    rows = []
    for r in trace["rounds"]:
        name = "reset.json" if r == 0 else os.path.join("round-resets", f"round-{r:04d}.json")
        reset = _json(os.path.join(run_dir, "native", name)) or {}
        rows.append({"n": 0, "clock": reset.get("physics_steps_before_policy"), "seg": "round_start", "action_id": None,
                     "poses": {}, "round": r})
    for span in trace["spans"]:
        first, last = span["first_step"] // PHYSICS_STEPS_PER_NATIVE, span["last_step"] // PHYSICS_STEPS_PER_NATIVE
        rows += [{"n": n, "clock": None, "seg": None, "action_id": span["action_id"], "poses": {}, "round": span["round"]}
                 for n in ((first + 1, last) if last > first else ())]
    return rows


def _robodojo_bundle(run_dir: str, cfg: Config, round_index: int) -> dict:
    from viewer.dashboard import scene_trace    # it builds on this module's readers

    gt = ground_truth_dir(run_dir)
    ticks_path = os.path.join(gt, "ticks.jsonl")

    def build_ticks():
        rows = []
        for t in _jsonl(ticks_path):
            poses = {k: _r(list(v["position_m"]) + _wxyz_to_xyzw(v["orientation_wxyz"]))
                     for k, v in (t.get("objects") or {}).items() if v.get("position_m") and v.get("orientation_wxyz")}
            rows.append({"n": t.get("native_action"), "clock": t.get("physics_clock"), "seg": t.get("segment"),
                         "action_id": t.get("action_id"), "poses": poses, "round": t.get("round_index") or 0})
        return rows
    every = memo(("replay-ticks", ticks_path), stat_sig(ticks_path), build_ticks)
    robot = robots.x5(cfg)
    trace = None if every else scene_trace.bundle(run_dir, cfg, round_index, robot=robot is None,
                                                  per_unit=PHYSICS_STEPS_PER_NATIVE)
    if trace:
        every = _trace_rows(run_dir, trace)
    rounds = sorted({t["round"] for t in every}) or [0]
    ticks = [t for t in every if t["round"] == round_index]
    starts = {r: next((t["clock"] for t in every if t["round"] == r and t["seg"] == "round_start"), None) for r in rounds}
    start_clock = starts.get(round_index)
    if start_clock is None and (trace or not every):   # no ground truth: the first observation's clock is the round start
        clocks = [row.get("after_physics_steps") for row in _jsonl(os.path.join(run_dir, "native", "observation-clocks.jsonl"))]
        start_clock = next((c for c in clocks if isinstance(c, int)), None)
    later = [c for r, c in starts.items() if r > round_index and isinstance(c, int)]
    end_clock = min(later) if later else None
    in_round = {t["action_id"] for t in ticks if t["action_id"]}
    every_action = {t["action_id"] for t in every if t["action_id"]}

    frames = []
    obs = [o for o in _observations(run_dir, start_clock) if o["n"] is not None and o["n"] >= 0
           and (end_clock is None or start_clock is None or o["n"] < (end_clock - start_clock) / PHYSICS_STEPS_PER_NATIVE)]
    if obs:
        p = os.path.join(run_dir, "policy-observations", obs[0]["dir"], "proprioception.json")
        first = memo(("replay-prop", p), stat_sig(p), lambda: _json(p) or {})
        ee = first.get("end_effectors") or {}
        frames.append({"n": obs[0]["n"] or 0, "seg": "observation",
                       "arms": {a: _arm_state(ee.get(a)) for a in ("left", "right") if ee.get(a)}})
    controller_span = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(run_dir), "native", "controller", "action-*.json"))):
        c = memo(("replay-ctl", path), stat_sig(path), lambda path=path: _controller_frames(path))
        if c["action_id"] in in_round or (len(rounds) == 1 and c["action_id"] not in every_action):
            frames.extend(c["frames"])
            ns = [f["n"] for f in c["frames"] if isinstance(f.get("n"), int)]
            if ns and c["action_id"]:
                controller_span[c["action_id"]] = (min(ns), max(ns))
    # A receipt belongs to the round its ticks are in; one that never moved (refused) to the round of the receipt
    # before it.
    actions = []
    owner = rounds[0]
    for path in sorted(glob.glob(os.path.join(glob.escape(run_dir), "native", "receipts", "action-*.json"))):
        a = memo(("replay-rcpt", path), stat_sig(path), lambda path=path: _receipt(path))
        a = dict(a)
        a.pop("_poses", None)
        owner = next((t["round"] for t in every if t["action_id"] == a.get("action_id")), owner)
        if owner == round_index:
            actions.append(a)
    span = {}
    for t in ticks:
        if t["action_id"] and isinstance(t["n"], int):
            lo, hi = span.get(t["action_id"], (t["n"], t["n"]))
            span[t["action_id"]] = (min(lo, t["n"]), max(hi, t["n"]))
    if not span:            # no ground-truth ticks: each command's span from the controller's samples
        span = controller_span
    last = 0
    for i, a in enumerate(actions):
        a["index"] = i
        lo_hi = span.get(a["action_id"])
        if lo_hi:
            a["n0"], a["n1"] = lo_hi[0] - 1, lo_hi[1]
            last = lo_hi[1]
        else:               # refused or failed before moving: a zero-length mark where the robot stood
            a["n0"] = a["n1"] = last

    frames = [f for f in frames if isinstance(f.get("n"), (int, float))]
    frames.sort(key=lambda f: f["n"])
    physics = os.path.join(gt, "physics", f"round-{round_index:04d}.json")
    instances = memo(("replay-inst", gt, round_index, cfg.object_root), stat_sig(physics),
                     lambda: _instances(gt, round_index, cfg.object_root))
    video = _video(run_dir, round_index)
    ground_truth = os.path.isfile(ticks_path)
    robot_links = trace and any(b["kind"] == scene_trace.ROBOT for b in trace["bodies"])
    return {
        "backend": "robodojo", "backend_name": BACKEND_NAMES["robodojo"],
        "clock": {"unit": "native action", "hz": NATIVE_HZ,
                  "source": "native actions since the round reset (10 physics steps each)"},
        "table_height": TABLE_HEIGHT, "workspace": WORKSPACE, "view": ROBODOJO_VIEW, "tcp_offset": TCP_IN_LINK6,
        "pose_point": "ee_link",
        "frames": frames, "ticks": [{"n": t["n"], "poses": t["poses"]} for t in ticks],
        "instances": instances, "observations": obs, "video": video, "actions": actions,
        "ground_truth": ground_truth, "round": round_index, "rounds": rounds,
        "max_n": _max_n(frames, [{"n": t["n"]} for t in ticks], obs, actions, video),
        "robot": robot, "robot_note": None if robot or robot_links else (
            "No robot model: start the dashboard with %s for the X5 URDF; TCP markers and paths only." % robots.FLAGS["x5"]),
        "robot_note_level": "notice",
        "objects_note": None if ground_truth or trace else scene_trace.missing_note(run_dir, " --resource-profile SLOT"),
        "scene_trace": trace,
        "mesh_version": MESH_VERSION,   # in the mesh URL, so a browser never reuses a mesh cached from an older conversion
    }


def _max_n(frames: list, ticks: list, obs: list, actions: list, video: dict) -> float:
    ns = [f["n"] for f in frames + ticks if isinstance(f.get("n"), (int, float))]
    ns += [o["n"] for o in obs if isinstance(o.get("n"), (int, float))]
    ns += [a["n1"] for a in actions if isinstance(a.get("n1"), (int, float))]
    ns += [f[0] for f in (video or {}).get("frames") or [] if isinstance(f[0], (int, float))]
    return max(ns + [0])


def _cam2world(calib) -> list | None:
    """A camera's pose in the episode frame (OpenCV axes), 4x4: ``cam2world_cv``, else the inverse of the
    world-to-camera ``extrinsic_cv`` (RoboTwin records only that and ``cam2world_gl``)."""
    if not isinstance(calib, dict):
        return None
    T = calib.get("cam2world_cv")
    if isinstance(T, list) and len(T) >= 3 and all(isinstance(r, list) and len(r) == 4 for r in T[:3]):
        return [list(map(float, r)) for r in T[:3]] + [[0.0, 0.0, 0.0, 1.0]]
    E = calib.get("extrinsic_cv")
    if not (isinstance(E, list) and len(E) >= 3 and all(isinstance(r, list) and len(r) == 4 for r in E[:3])):
        return None
    R = [[float(E[i][j]) for j in range(3)] for i in range(3)]       # rigid: the inverse is [R^T, -R^T t]
    t = [float(E[i][3]) for i in range(3)]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    return [Rt[i] + [-sum(Rt[i][k] * t[k] for k in range(3))] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


#: Without a crop box (a run that records no end-effector pose), points farther than this from their camera go.
CLOUD_MAX_DEPTH_M = 4.0


def cloud(run_dir: str, obs_dir: str, cfg: Config) -> str | None:
    """Binary: uint32 n, then n*3 float32 xyz, then n*3 uint8 rgb. All cameras of one observation, back-projected
    into the episode frame, cropped to the run's workspace (``run_workspace``) and voxel-downsampled (cached under
    ``<cache>/clouds/``)."""
    if not re.match(r"^[A-Za-z0-9:._-]+$", obs_dir or ""):
        return None
    src = os.path.join(run_dir, "policy-observations", obs_dir)
    if not os.path.isdir(src):
        return None
    box = run_workspace(run_dir, cfg)
    tag = hashlib.sha256((os.path.realpath(src) + json.dumps(box)).encode()).hexdigest()[:16]
    out = os.path.join(cfg.cache, "clouds", "v%d_%s.bin" % (CLOUD_VERSION, tag))
    with _key_lock(out):
        if os.path.isfile(out):
            return out
        import numpy as np
        from PIL import Image
        pts, cols = [], []
        lo, hi = (np.asarray(box[0]), np.asarray(box[1])) if box else (None, None)
        for cam in _cameras(src):
            d = os.path.join(src, cam)
            calib = _json(os.path.join(d, "calibration.json"))
            T = _cam2world(calib)
            if T is None or not isinstance(calib.get("intrinsic_cv"), list) or not os.path.isfile(os.path.join(d, "depth-m.npy")):
                continue
            depth = np.load(os.path.join(d, "depth-m.npy")).astype(np.float32)
            rgb = np.asarray(Image.open(os.path.join(d, "rgb.png")).convert("RGB"))
            K = np.asarray(calib["intrinsic_cv"], float)
            T = np.asarray(T, float)
            h, w = depth.shape
            v, u = np.mgrid[0:h, 0:w]
            valid = np.isfinite(depth) & (depth > 0)
            if box is None:
                valid &= depth <= CLOUD_MAX_DEPTH_M
            z = depth[valid]
            x = (u[valid] - K[0, 2]) * z / K[0, 0]
            y = (v[valid] - K[1, 2]) * z / K[1, 1]
            world = np.stack([x, y, z], 1) @ T[:3, :3].T + T[:3, 3]
            keep = np.all((world >= lo) & (world <= hi), axis=1) if box else np.ones(len(world), bool)
            pts.append(world[keep])
            cols.append(rgb[valid][keep])
        if pts:
            P, C = np.concatenate(pts), np.concatenate(cols).astype(np.float32)
        else:
            P, C = np.zeros((0, 3)), np.zeros((0, 3))
        voxel = 0.003
        if lo is None and len(P):
            lo = P.min(axis=0)
        while len(P):
            cells = np.floor((P - lo) / voxel).astype(np.int64)
            keys = (cells[:, 0] * 4096 + cells[:, 1]) * 4096 + cells[:, 2]
            _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
            inv = inv.reshape(-1)
            if len(counts) <= CLOUD_MAX_POINTS:
                break
            voxel *= 1.25
        if len(P):
            n = len(counts)
            mp = np.zeros((n, 3))
            np.add.at(mp, inv, P)
            mp /= counts[:, None]
            mc = np.zeros((n, 3))
            np.add.at(mc, inv, C)
            mc /= counts[:, None]
        else:
            mp, mc = P, C
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "wb") as f:
            f.write(struct.pack("<I", len(mp)) + mp.astype("<f4").tobytes() + np.clip(mc, 0, 255).astype(np.uint8).tobytes())
        os.replace(tmp, out)
        return out


# ----------------------------------------------------------------------------- object meshes

def mesh(usd_path: str | None, cfg: Config) -> str | None:
    """A simplified, coloured mesh of a RoboDojo object asset under the configured ``Object/`` directory
    (``meshes.py`` under ``cfg.mesh_python``; converted once, cached under ``<cache>/meshes/``)."""
    root = cfg.object_root
    if not root or not usd_path or ".." in usd_path or not usd_path.endswith("/object.usdz") \
            or not os.path.realpath(usd_path).startswith(root.rstrip(os.sep) + os.sep) or not os.path.isfile(usd_path):
        return None
    asset = os.path.dirname(usd_path)
    rel = os.path.relpath(asset, root)
    out = os.path.join(cfg.cache, "meshes", "v%d_%s.bin" % (MESH_VERSION, re.sub(r"[^A-Za-z0-9_]", "_", rel)))
    with _key_lock(out):
        if not os.path.isfile(out):
            done = subprocess.run([cfg.mesh_python, "-I", os.path.join(HERE, "meshes.py"), asset, out],
                                  capture_output=True, text=True, timeout=300)
            if done.returncode or not os.path.isfile(out):
                raise RuntimeError("mesh conversion failed: " + (done.stderr or done.stdout)[-400:])
    return out
