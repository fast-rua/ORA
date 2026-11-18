"""The 3D replay bundle for RoboTwin, LIBERO-Pro, RoboCasa365 and RoboLab runs: the same shape as RoboDojo's
(``replay.py``), built from what every backend writes (receipts, observations, the agent's trace) and what some add
(execution capture, RoboTwin's controller replay). ``replay.bundle`` dispatches here; the module docstring of
``replay.py`` lists the sources and each backend's clock.

- Clock: the execution capture's simulator steps since the round reset, each command spanning its action_start to
  action_end frames; without a capture, each receipt's steps accumulated in action order within its round.
- Commands: the receipts, coloured by interaction mode where they have one (``_receipt``).
- Poses: RoboTwin's controller replay per SAPIEN step (arm joints, TCP, the commanded gripper), else each command's
  measured end-effector (and base) pose before and after it and each observation's proprioception.
- Observations: at the end of the command their observation.json names (``previous_execution_receipt``), else at
  their round's start.
- Objects: the run's scene trace when one exists (``scene_trace.py``, on the same simulator-step clock); so is the
  robot on a backend without a URDF here, and the hand's path where the trace names the TCP.
"""

from __future__ import annotations

import glob
import os
import re

from viewer.dashboard import robots, scene_trace
from viewer.dashboard.config import Config
from viewer.dashboard.replay import (
    MESH_VERSION, _base3, _cam2world, _cameras, _capture_rows, _json, _jsonl, _max_n, _pose7, _r, _receipt, _video,
)
from viewer.dashboard.runs import BACKEND_NAMES
from viewer.dashboard.store import memo, read_json_proj, stat_sig

#: Execution-capture step units, as the page names them.
STEP_UNITS = {"sapien_step_since_round_reset": "SAPIEN step", "mujoco_step_since_round_reset": "MuJoCo step"}
#: Every Nth SAPIEN step of RoboTwin's controller replay becomes a frame (250 Hz -> 50 Hz), plus each command's last.
REPLAY_STRIDE = 5
#: Point clouds are cropped to the end-effector positions' box grown by this much (x, y, below, above), metres.
CLOUD_PAD = (0.6, 0.6, 0.6, 0.4)
OBJECTS_NOTE = "needs ground truth or a scene trace, phase 2"
#: The backends ``ora replay RUN --scene`` writes a trace for.
SCENE_BACKENDS = ("robotwin", "libero_pro", "robocasa", "robolab")
#: A trace's steps per unit of the page's clock, where the two differ: both clocks here are native actions, 25 MuJoCo
#: steps each on RoboCasa365 and 8 PhysX steps on RoboLab.
TRACE_STEPS_PER_UNIT = {"robocasa": 25, "robolab": 8}
#: A run sealed before these backends recorded their controls has no replay program, so no trace can ever be exported.
NO_CONTROLS_NOTE = "no scene trace: this run records no per-tick controls, so it cannot be replayed"
NO_JOINTS = {
    "libero_pro": "No robot model: LIBERO-Pro's controller replay records commands only, no measured joints over time; "
                  "a scene trace draws it (`ora replay RUN --scene`).",
    "robocasa": "No robot model: RoboCasa365 records the end effector and the base, no joints over time; "
                "a scene trace draws it (`ora replay RUN --scene`).",
    "robolab": "No robot model: RoboLab records the end effector only, no joints over time; a scene trace draws it "
               "(`ora replay RUN --scene`).",
    "robotwin": "No robot model: this run has no measured joints over time (native/controller-replay).",
}


def _plan_runtime(p: dict) -> dict:
    rt = p.get("runtime") if isinstance(p.get("runtime"), dict) else {}
    ctx = rt.get("public_environment_context") if isinstance(rt.get("public_environment_context"), dict) else {}
    return {"embodiment_id": ctx.get("embodiment_id")}


def _receipts(run_dir: str) -> list[dict]:
    """Every receipt of the run in action order (a copy of each, with its file name under ``_file``)."""
    out = []
    for path in sorted(glob.glob(os.path.join(glob.escape(run_dir), "native", "receipts", "action-*.json"))):
        a = dict(memo(("replay-rcpt", path), stat_sig(path), lambda path=path: _receipt(path)))
        a["_file"] = os.path.basename(path)
        out.append(a)
    return out


def _capture_clock(rows: list) -> dict | None:
    """The execution capture's step unit and rate, and each command's span in its steps:
    action_id -> (round, first action_start, last action_end); None when it marks no command."""
    spans: dict = {}
    unit, rates = None, []
    for r in rows:
        step = r.get("sim_step")
        if not isinstance(step, (int, float)):
            continue
        unit = unit or r.get("step_unit")
        t = r.get("sim_time_s")
        if isinstance(t, (int, float)) and t > 0 and step > 0:
            rates.append(step / t)
        aid = r.get("action_id")
        if not aid or r.get("phase") not in ("action_start", "action_end"):
            continue
        rnd, lo, hi = spans.get(aid, (r.get("round_index") or 0, None, None))
        if r["phase"] == "action_start" and lo is None:
            lo = step
        elif r["phase"] == "action_end":
            hi = step
        spans[aid] = (rnd, lo, hi)
    if not spans:
        return None
    rates.sort()
    return {"unit": STEP_UNITS.get(unit, unit or "simulator step"),
            "hz": round(rates[len(rates) // 2], 1) if rates else None, "spans": spans}


def _replay_files(run_dir: str) -> dict:
    """RoboTwin's controller replay: action_id -> npz path relative to native/, from its events journal."""
    path = os.path.join(run_dir, "native", "controller-replay", "events.jsonl")

    def build() -> dict:
        out = {}
        for row in _jsonl(path):
            e = row.get("entry") if isinstance(row.get("entry"), dict) else {}
            rel = (e.get("artifact") or {}).get("path")
            if e.get("action_id") and isinstance(rel, str) and re.match(r"^controller-replay/[A-Za-z0-9._-]+\.npz$", rel):
                out[e["action_id"]] = rel
        return out
    return memo(("replay-ctlrep", path), stat_sig(path), build)


def _replay_series(path: str) -> dict | None:
    """One RoboTwin command's measured arm joints and TCP poses per SAPIEN step and, for a set_gripper, the
    commanded gripper (0 closed .. 1 open) per step: {arm: {"q", "pose", "g"}}; a command that ran a global then a
    local phase stores the global phase's arrays under ``global__``, first."""
    try:
        import numpy as np
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception:       # numpy missing or an unreadable file: the receipts' poses still draw the paths
        return None
    out = {}
    for arm in ("left", "right"):
        def series(name: str):
            parts = [arrays[k] for k in ("global__" + name, name) if k in arrays and arrays[k].ndim == 2 and len(arrays[k])]
            return np.concatenate(parts).astype(float) if parts else None
        q, pose = series(f"measured_{arm}_arm__position"), series(f"measured_{arm}_tcp__pose_xyzw")
        if q is None or pose is None or len(q) != len(pose) or pose.shape[1] != 7:
            continue
        g = arrays.get(f"{arm}_gripper__result")
        out[arm] = {"q": q.round(5).tolist(), "pose": pose.round(5).tolist(),
                    "g": g.astype(float).round(4).tolist() if g is not None and g.ndim == 1 and len(g) else None}
    return out or None


def _proj_observation(o: dict) -> dict:
    prev = o.get("previous_execution_receipt") if isinstance(o.get("previous_execution_receipt"), dict) else {}
    return {"after": os.path.basename(prev["path"]) if isinstance(prev.get("path"), str) else None,
            "captured": o.get("captured_unix_s"), "main_camera": o.get("main_camera")}


def _proj_proprio(p: dict) -> dict:
    """End-effector poses (and, RoboTwin, arm joints and the measured finger joint) and the base pose."""
    joints = p.get("joints") if isinstance(p.get("joints"), dict) else {}
    arms = {}
    for a, e in (p.get("end_effectors") or {}).items():
        pose = _pose7(e.get("pose_xyz_xyzw")) if isinstance(e, dict) else None
        if not pose:
            continue
        st: dict = {"pose": pose}
        q = joints.get(f"{a}_arm")
        if isinstance(q, list) and q:
            st["q"] = _r(q)
        gs = e.get("gripper_state") if isinstance(e.get("gripper_state"), dict) else {}
        if isinstance(gs.get("measured_primary_joint_position"), (int, float)):
            st["finger"] = round(float(gs["measured_primary_joint_position"]), 5)
        arms[a] = st
    return {"arms": arms, "base": _base3(p.get("base"))}


def _proprio(run_dir: str, obs_dir: str) -> dict:
    path = os.path.join(run_dir, "policy-observations", obs_dir, "proprioception.json")
    return read_json_proj(path, _proj_proprio, kind="replay-proprio") or {"arms": {}, "base": None}


def _arm_frame(st: dict, finger) -> dict:
    out = {"pose": st["pose"]}
    if st.get("q"):
        out["q"] = st["q"]
    if finger is not None:
        out["f"] = [finger, finger]
    return out


def _replay_frames(acts: list, series: dict, obs: list, proprio: dict, scale: list | None) -> list:
    """RoboTwin frames from the controller replay: every ``REPLAY_STRIDE``-th SAPIEN step of each command, plus the
    first observation. Fingers: the measured opening at observations; during a set_gripper the commanded opening
    per step, stopped at the opening measured at the next observation when it closes (fingers stop on what they
    grasp; the replay logs the command, not the fingers); held between."""
    frames, finger = [], {}
    measured = sorted((o["n"], arm, st["finger"]) for o in obs for arm, st in proprio[o["dir"]]["arms"].items()
                      if st.get("finger") is not None)
    if obs:
        first = proprio[obs[0]["dir"]]["arms"]
        finger = {arm: st.get("finger") for arm, st in first.items()}
        frames.append({"n": obs[0]["n"], "seg": "observation",
                       "arms": {arm: _arm_frame(st, finger.get(arm)) for arm, st in first.items()}})
    for a in acts:
        sr = series.get(a["_file"])
        if not sr:
            continue
        length = len(next(iter(sr.values()))["q"])
        ramp = {}
        for arm, s in sr.items():
            g = s.get("g")
            if not g or not scale:
                continue
            v = [scale[0] + x * (scale[1] - scale[0]) for x in g]
            if g[-1] < g[0]:
                after = next((f for n, ar, f in measured if ar == arm and n >= a["n1"]), None)
                if after is not None:
                    v = [max(x, after) for x in v]
            ramp[arm] = [round(max(0.0, x), 5) for x in v]
        per = (a["n1"] - a["n0"]) / length if a["n1"] > a["n0"] else 1.0
        for i in sorted(set(range(0, length, REPLAY_STRIDE)) | {length - 1}):
            arms = {}
            for arm, s in sr.items():
                f = ramp[arm][min(i, len(ramp[arm]) - 1)] if arm in ramp else finger.get(arm)
                arms[arm] = _arm_frame({"pose": s["pose"][i], "q": s["q"][i]}, f)
            frames.append({"n": round(a["n0"] + (i + 1) * per, 3), "seg": a.get("action"), "arms": arms})
        for arm, v in ramp.items():
            finger[arm] = v[-1]
    return frames


def _receipt_frames(acts: list, obs: list, proprio: dict) -> list:
    """End-effector (and base) poses at each observation and at each command's start and end."""
    frames = []
    for o in obs:
        p = proprio[o["dir"]]
        frames.append({"n": o["n"], "seg": "observation", "arms": {a: {"pose": st["pose"]} for a, st in p["arms"].items()},
                       "base": p["base"]})
    for a in acts:
        P = a["_poses"]
        frames.append({"n": a["n0"], "seg": "before", "arms": {arm: {"pose": p} for arm, p in P["before"].items()},
                       "base": P["base_before"]})
        frames.append({"n": a["n1"], "seg": "after", "arms": {arm: {"pose": p} for arm, p in P["after"].items()},
                       "base": P["base_after"]})
    return [f for f in frames if f["arms"] or f["base"]]


def workspace(run_dir: str) -> tuple | None:
    """The point clouds' crop box: the box of every end-effector position the run's receipts and observations
    record (all rounds), grown by ``CLOUD_PAD``; None when the run records none."""
    rdir, odir = os.path.join(run_dir, "native", "receipts"), os.path.join(run_dir, "policy-observations")

    def build():
        pts = [p[:3] for a in _receipts(run_dir) for side in ("before", "after") for p in a["_poses"][side].values()]
        for d in glob.glob(os.path.join(glob.escape(odir), "*")):
            pts += [st["pose"][:3] for st in _proprio(run_dir, os.path.basename(d))["arms"].values()]
        if not pts:
            return None
        px, py, below, above = CLOUD_PAD
        lo = [min(p[k] for p in pts) for k in range(3)]
        hi = [max(p[k] for p in pts) for k in range(3)]
        return ([round(lo[0] - px, 3), round(lo[1] - py, 3), round(lo[2] - below, 3)],
                [round(hi[0] + px, 3), round(hi[1] + py, 3), round(hi[2] + above, 3)])
    return memo(("replay-ws", run_dir), (stat_sig(rdir), stat_sig(odir)), build)


#: RoboTwin's main camera sits on the robot's own mast, which then fills the view: look from the front right.
VIEW_OFFSET = {"robotwin": [1.0, 0.8, 0.7]}
OVERVIEW_EXTENT_M = 1.5


def _view(run_dir: str, obs: list, box: tuple | None, backend: str | None = None) -> dict:
    """The page's first view: from the agent's main (else first fixed) camera of the first observation, pulled back,
    towards the centre of the end-effector positions (RoboTwin: from a fixed offset to that centre; a run whose end
    effector travels more than ``OVERVIEW_EXTENT_M``, a mobile base's, from above and behind all of it)."""
    extent = 0.0
    if box:
        px, py, below, above = CLOUD_PAD
        lo, hi = box
        lo, hi = [lo[0] + px, lo[1] + py, lo[2] + below], [hi[0] - px, hi[1] - py, hi[2] - above]
        target = [round((lo[k] + hi[k]) / 2, 3) for k in range(3)]
        extent = max(hi[k] - lo[k] for k in range(3))
    else:
        target = [0.0, 0.0, 0.8]
    eye = [round(target[k] + VIEW_OFFSET[backend][k], 3) for k in range(3)] if backend in VIEW_OFFSET else None
    if extent > OVERVIEW_EXTENT_M:
        eye = [round(target[0] - 0.2 * extent, 3), round(target[1] - 0.9 * extent, 3), round(target[2] + 0.9 * extent, 3)]
    if obs and eye is None:
        o = obs[0]
        names = [o.get("_main")] + [c for c in sorted(o["cams"]) if not re.search(r"wrist|hand|left_camera|right_camera", c)]
        for cam in names:
            if cam not in o["cams"]:
                continue
            T = _cam2world(_json(os.path.join(run_dir, "policy-observations", o["dir"], cam, "calibration.json")))
            if T is not None:
                c = [T[k][3] for k in range(3)]
                eye = [round(c[k] + 0.3 * (c[k] - target[k]), 3) for k in range(3)]
                break
    if eye is None or sum((eye[k] - target[k]) ** 2 for k in range(3)) < 0.04:
        eye = [target[0], round(target[1] - 1.2, 3), round(target[2] + 0.8, 3)]
    return {"eye": eye, "target": target}


def _objects_note(run_dir: str, backend: str | None) -> str:
    """Why the page has no objects: an export away, nothing to export from, or a backend with no trace yet."""
    if backend not in SCENE_BACKENDS:
        return OBJECTS_NOTE
    if backend in ("robocasa", "robolab") and not os.path.isfile(
            os.path.join(run_dir, "native", "controller-replay", "manifest.json")):
        return NO_CONTROLS_NOTE
    return scene_trace.missing_note(run_dir)


def bundle(run_dir: str, cfg: Config, round_index: int, backend: str | None) -> dict:
    """The 3D page's bundle for one round of a RoboTwin, LIBERO-Pro, RoboCasa365 or RoboLab run, the same shape as
    RoboDojo's (``replay.bundle``)."""
    receipts = _receipts(run_dir)
    by_file = {a["_file"]: a for a in receipts}
    capture = _capture_clock(_capture_rows(run_dir))
    spans = capture["spans"] if capture else {}
    series: dict = {}
    if backend == "robotwin":
        files = _replay_files(run_dir)
        for a in receipts:
            rel = files.get(a.get("action_id")) or "controller-replay/" + a["_file"][:-5] + ".npz"
            p = os.path.join(run_dir, "native", rel)
            if os.path.isfile(p):
                sr = memo(("replay-npz", p), stat_sig(p), lambda p=p: _replay_series(p))
                if sr:
                    series[a["_file"]] = sr

    # the round of each command: its receipt's, else the execution capture's, else the round of the one before it
    rnd = 0
    for a in receipts:
        P, cap = a["_poses"], spans.get(a.get("action_id"))
        rnd = P["round"] if P["round"] is not None else (cap[0] if cap else rnd)
        a["_round"] = rnd
    rounds = sorted({a["_round"] for a in receipts} | {s[0] for s in spans.values()}) or [0]

    # each command's span on the clock
    if capture:
        clock = {"unit": capture["unit"], "hz": capture["hz"],
                 "source": "the execution capture's simulator steps since the round reset"}
    else:
        clock = {"unit": "SAPIEN step" if backend == "robotwin" else "native action", "hz": None,
                 "source": "each command's steps from its receipt, accumulated in action order"}
    cursor: dict = {}
    for a in receipts:
        last = cursor.get(a["_round"], 0)
        if capture:
            cap = spans.get(a.get("action_id"))
            if cap and cap[1] is not None:      # a command that never moved has no span: a mark where the robot stood
                a["n0"], a["n1"] = cap[1], cap[2] if cap[2] is not None else cap[1]
            else:
                a["n0"] = a["n1"] = last
        else:
            steps = a["_poses"]["steps"]
            if a["_file"] in series:
                steps = len(next(iter(series[a["_file"]].values()))["q"])
            a["n0"], a["n1"] = last, last + (steps or 0)
        cursor[a["_round"]] = max(last, a["n1"])

    # each observation at the end of the command it follows (observation.json names that receipt), else at its
    # round's start
    obs = []
    for d in glob.glob(os.path.join(glob.escape(run_dir), "policy-observations", "*")):
        m = re.search(r"observation-(\d+)$", d)
        if not m:
            continue
        name = os.path.basename(d)
        info = read_json_proj(os.path.join(d, "observation.json"), _proj_observation, kind="replay-obs") or {}
        named = re.search(r":round-(\d+):", name)
        prev = by_file.get(info.get("after"))
        if prev and (not named or int(named.group(1)) == prev["_round"]):
            o_round, n = prev["_round"], prev["n1"]
        else:
            t = info.get("captured")
            nxt = next((a for a in receipts if isinstance(t, (int, float)) and isinstance(a["_poses"]["finished"], (int, float))
                        and a["_poses"]["finished"] > t), None)
            o_round = int(named.group(1)) if named else (nxt["_round"] if nxt else rounds[0])
            n = 0
        if o_round != round_index:
            continue
        cams = {cam: {"rgb": os.path.join(d, cam, "rgb.png"),
                      "depth": os.path.isfile(os.path.join(d, cam, "depth-m.npy"))
                      and os.path.isfile(os.path.join(d, cam, "calibration.json"))}
                for cam in _cameras(d)}
        obs.append({"seq": int(m.group(1)), "dir": name, "n": n, "cams": cams, "_main": info.get("main_camera")})
    obs.sort(key=lambda o: o["seq"])

    acts = [a for a in receipts if a["_round"] == round_index]
    proprio = {o["dir"]: _proprio(run_dir, o["dir"]) for o in obs}
    if any(a["_file"] in series for a in acts):
        conf = robots.aloha_config(cfg)
        frames = _replay_frames(acts, series, obs, proprio, (conf or {}).get("gripper_scale"))
    else:
        frames = _receipt_frames(acts, obs, proprio)
    frames.sort(key=lambda f: f["n"])

    robot, note, level = None, NO_JOINTS.get(backend or ""), "info"
    if backend == "robotwin":
        plan = read_json_proj(os.path.join(run_dir, "plan.json"), _plan_runtime, kind="replay-plan-runtime") or {}
        embodiment = plan.get("embodiment_id")
        if embodiment not in (None, "aloha_agilex"):
            note = "No robot model for embodiment %s; TCP markers and paths only." % embodiment
        elif any(st.get("q") for f in frames for st in f["arms"].values()):
            robot = robots.aloha(cfg)
            note = None if robot else ("No robot model: start the dashboard with %s for the aloha-agilex URDF; TCP "
                                       "markers and paths only." % robots.FLAGS["aloha"])
            level = "notice"
    actions = []
    for i, a in enumerate(acts):
        out = {k: v for k, v in a.items() if not k.startswith("_")}
        out["index"] = i
        actions.append(out)
    box = workspace(run_dir)
    video = _video(run_dir, round_index, per_unit=1) if capture else {"frames": [], "fps": None, "status": None}
    observations = [{k: v for k, v in o.items() if not k.startswith("_")} for o in obs]
    trace = scene_trace.bundle(run_dir, cfg, round_index, robot=robot is None,
                               per_unit=TRACE_STEPS_PER_UNIT.get(backend or "", 1))
    if trace and any(b["kind"] == scene_trace.ROBOT for b in trace["bodies"]):
        note = None
    return {
        "backend": backend, "backend_name": BACKEND_NAMES.get(backend, backend), "clock": clock,
        "table_height": None, "workspace": box, "view": _view(run_dir, obs, box, backend), "tcp_offset": None,
        "pose_point": "TCP", "frames": frames, "ticks": [], "instances": {}, "observations": observations,
        "video": video, "actions": actions, "ground_truth": False, "round": round_index, "rounds": rounds,
        "max_n": _max_n(frames, [], observations, actions, video),
        "robot": robot, "robot_note": note, "robot_note_level": level,
        "objects_note": None if trace else _objects_note(run_dir, backend),
        "scene_trace": trace, "mesh_version": MESH_VERSION,
    }


# ----------------------------------------------------------------------------- point clouds
