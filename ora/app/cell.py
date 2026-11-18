"""``ora cell``: set up, watch and recover a real robot cell (``ora/cell``, docs/usage/real-cells.md).

    ora cell init [--example PATH]          validate the cell config against its robot, list devices
    ora cell check                          buses, other controllers, cameras and their depth
    ora cell read | record-rest             the arms, read without enabling anything
    ora cell grippers [--orient]            each gripper opens and closes once to find its limits;
                                            then (or with --orient alone, nothing moving) you say which way each is
    ora cell calibrate --method arm-poses|board-poses|board-relative|charuco|wrists [--square-mm N] [--pattern CxR] [--role R]
    ora cell calibrate --method hand-eye [--square-mm N] [--pattern CxR] [--placements N] [--arms A,B]
                       [--records DIR ...] [--ask terminal|broker]
    ora cell setup [--status] [--redo STEP ...] [--only STEP ...]
                                            a new cell, step by step: devices, rest, grippers, fixed cameras,
                                            hand-eye calibration, table, verify (what is not done yet)
    ora cell board [--port 8765]            the operator's page: state, live cloud, verbs
    ora cell depth-service [...]            the depth completion service (ora.cell.depth.service)
    ora cell home | hold | float | release | estop | diagnose | stop | kill | cameras | can
    ora cell pending | approve [--id ID] [--value V] | decline [--id ID] [--reason TEXT]
    ora cell auto-approve [on|off|status]   motions of the running play worker answer themselves
    ora cell mode [ask|safe|auto|status]    the permission mode: every motion asked, small ones not, or none
    ora cell stop-motion                    the running motion ends where the arms are; they stay held there
    ora cell sim init PATH | status | place X Y [--yaw DEG] | clear | operator [--seconds S]
                                            a simulated cell (``robot.sim``): write one, compare its
                                            calibration with the injected truth, move its board, answer
                                            its questions as the person at the cell would

Every command reads the cell config (``--cell``, else ``$ORA_CELL``, else
``~/.config/ora/cell.yaml``). The operator verbs (home ... can) act at once, also while a
play session moves the arms; ``hold`` and ``float`` keep the arms held and so need a
running cell board, which they reach over HTTP; with a board running every verb goes
through it. ``--json`` prints ``{"ok": true, "result": ...}`` or ``{"ok": false,
"error": ...}`` (exit 1); ``--yes`` skips the confirmation of verbs that make arms limp
or move them by hand.

``calibrate --method hand-eye`` is the guided calibration of a new cell
(``ora/cell/operator/handeye.py``): the person places a checkerboard a few times, the arms
look at it, and one fit measures the fixed cameras, the wrist cameras' mounts, the arms'
bases and their joint zeros; it asks every step (place the board, each motion, the
measured square, writing the result) on this terminal, or through the approval broker
when it has none (``--ask broker``) and prints its progress on stderr (JSON lines with
``--json``). ``--records`` fits earlier captures again without moving anything.

``pending``, ``approve`` and ``decline`` answer the questions a robot's worker asks the
operator (enable the arms, run a motion, release arms that could not get home): they
read and write only the cell's approval files (``$ORA_CELL_APPROVALS``, else beside the
cell lock), never the lock or the hardware. An answer names the question's ``--id``
(``pending`` lists them); without one, on a terminal, the one waiting question is shown
and answered after a yes.

``auto-approve on`` (the operator's "YOLO" switch) makes the play worker that drives the
arms now (``rig_mode on``) answer its own motion questions; enabling the arms and releasing
them are still asked. ``off`` turns it off; ``status`` (the default) says whether it is on.
It too touches only the approval files: the switch lives for that worker only, and a new
worker clears it unless its run was started with ``ora play ... --auto-approve``.

A simulated cell (``robot.sim: true``, docs/usage/real-cells.md) runs every command above
with no hardware: the robot's simulated adapter renders its cameras and moves kinematic
arms. Its lock, approvals and alerts live in its state directory unless
``ORA_CELL_LOCK``, ``ORA_CELL_APPROVALS`` or ``ORA_CELL_ALERTS`` name others, so it never
shares them with a real cell. ``ora cell sim init PATH`` writes such a config;
``sim status`` compares the calibration with the errors the simulation injected (its
``world.json``), ``sim place`` and ``sim clear`` move the checkerboard, and ``sim operator``
is the person at the cell: it answers the broker's questions (board placements, motions,
square size, which way a gripper opens, writing) until stopped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

VERBS = ("home", "hold", "float", "release", "estop", "diagnose", "stop", "kill", "cameras", "can")
SETUP = ("init", "check", "read", "record-rest", "grippers", "calibrate", "setup", "board", "depth-service")
ANSWERS = ("pending", "approve", "decline", "auto-approve", "mode", "stop-motion")
SIM = ("sim",)
CONFIRM = {
    "float": "Float the arms (gravity compensation)? Put a hand on each arm first.",
    "release": "Release the arms? They go limp and fall unless supported or at rest.",
    "estop": "Disable every motor now? Arms that are up will fall.",
    "kill": "End the program holding the cell? Enabled arms fall within the motor timeout.",
    "grippers": "Each gripper will open and close once. Keep hands clear of the jaws.",
    "calibrate": "The arms will move through a few poses near rest. Stand clear, keep the board open.",
}
EXAMPLE = """schema: ora.cell-config.v1
cell_id: my-cell-01
robot:
  environment: yam                      # the robot family's backend
  model: yam_station_linear_4310_d405   # the robot model: geometry and policy composition
  python: ~/envs/yam/bin/python         # the interpreter with the robot's drivers (i2rt, pyrealsense2, ...)
  buses: {left: can_left, right: can_right}
cameras:                                # role: the robot model's public camera; serial: the device
  overhead:    {serial: "000000000000", mount: fixed_world}
  front:       {serial: "000000000000", mount: fixed_world}
  wrist_left:  {serial: "000000000000", mount: end_effector}
  wrist_right: {serial: "000000000000", mount: end_effector}
calibration: ~/.config/ora/cells/my-cell-01.calibration.json   # written by `ora cell` tools only
safety:
  operator_present_required: true
  foreign_controllers: {ignore: []}     # command-line fragments of programs you vouch for
perception:
  depth_completion: null                # e.g. http://localhost:8770 (`ora cell depth-service`)
policy:
  provider_profile: deepseek_flash_openai_low_16k_v1
"""
SIM_EXAMPLE = """schema: ora.cell-config.v1
cell_id: sim-yam-01
robot:
  environment: yam
  model: yam_station_linear_4310_d405
  sim: true                             # no hardware: state, truth and lock in {state}
  python: {python}
  buses: {{left: sim_left, right: sim_right}}
cameras:
  overhead:    {{serial: "sim-overhead", mount: fixed_world}}
  front:       {{serial: "sim-front", mount: fixed_world}}
  wrist_left:  {{serial: "sim-wrist-left", mount: end_effector}}
  wrist_right: {{serial: "sim-wrist-right", mount: end_effector}}
calibration: {calibration}
safety:
  operator_present_required: true
perception:
  depth_completion: null
policy:
  provider_profile: deepseek_flash_openai_low_16k_v1
"""


def register(sub: Any) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cell", type=Path, help="the cell config (default: $ORA_CELL, else ~/.config/ora/cell.yaml)")
    common.add_argument("--json", action="store_true", help="print one JSON object")
    common.add_argument("--yes", action="store_true", help="skip confirmations")
    common.add_argument("--board-port", type=int, default=int(os.environ.get("ORA_CELL_BOARD_PORT", "8765")),
                        help="the port a running cell board serves on")
    parser = sub.add_parser("cell", help="set up, watch and recover a real robot cell", description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="action", required=True, metavar="ACTION")
    helps = {"init": "validate the cell config, list devices", "check": "buses, other controllers, cameras",
             "read": "the arms, read without enabling", "record-rest": "record the rest pose",
             "grippers": "find each gripper's limits (opens and closes once)",
             "calibrate": "place the fixed cameras", "setup": "a new cell, step by step",
             "board": "the operator's page",
             "depth-service": "the depth completion service",
             "home": "arms back to rest, then released", "hold": "hold the arms where they are (needs a board)",
             "float": "float the arms for hand guiding (needs a board)", "release": "the arms go limp",
             "estop": "disable every motor, whoever holds the cell", "diagnose": "who holds the cell, faults, rest",
             "stop": "ask the holder to home and release", "kill": "end the holder",
             "cameras": "hardware-reset the cameras", "can": "re-initialise the buses (sudo)",
             "pending": "the worker's questions waiting for an answer",
             "approve": "approve one waiting question (by --id)", "decline": "decline one waiting question (by --id)",
             "auto-approve": "the running worker's motions answer themselves (on|off|status)",
             "mode": "the permission mode (ask|safe|auto|status)",
             "stop-motion": "end the running motion where the arms are, and hold there",
             "sim": "a simulated cell: init, status, place, clear, operator"}
    for name in SETUP + VERBS + ANSWERS + SIM:
        if name == "depth-service":
            action = actions.add_parser(name, help=helps[name], add_help=False)
            action.add_argument("rest", nargs=argparse.REMAINDER, help="arguments of ora.cell.depth.service")
            continue
        action = actions.add_parser(name, help=helps[name], parents=[common])
        if name == "board":
            action.add_argument("--port", type=int, help="the port to serve on (default 8765)")
            action.add_argument("--host", default="127.0.0.1", help="the address to serve on")
            action.add_argument("--depth-service", help="a depth service URL (default: the cell config's)")
        if name == "calibrate":
            action.add_argument("--method", choices=("arm-poses", "board-poses", "board-relative", "charuco", "wrists",
                                                     "hand-eye"), required=True)
            action.add_argument("--reference", help="--method board-relative: a placed fixed camera that also sees the board")
            action.add_argument("--pattern", help="--method board-poses|board-relative|hand-eye: inner corners COLSxROWS of a plain checkerboard")
            action.add_argument("--role", help="--method wrists: the fixed camera to place")
            action.add_argument("--square-mm", type=float, help="--method charuco|board-*|hand-eye: one square, measured")
            action.add_argument("--placements", type=int, help="--method hand-eye: how many board placements (default 5)")
            action.add_argument("--arms", help="--method hand-eye: the arms that look, comma-separated (default: every arm)")
            action.add_argument("--records", nargs="+", type=Path,
                                help="--method hand-eye: fit these earlier placements again; nothing moves")
        if name == "setup":
            action.add_argument("--status", action="store_true", help="only say what is done and what comes next")
            action.add_argument("--redo", nargs="+", default=[], metavar="STEP", help="do these steps again")
            action.add_argument("--only", nargs="+", default=[], metavar="STEP", help="do only these steps")
        if name in {"calibrate", "grippers", "setup"}:
            action.add_argument("--ask", choices=("terminal", "broker"),
                                help="where to ask the person (default: the terminal when there is one)")
        if name == "grippers":
            action.add_argument("--orient", action="store_true",
                                help="only check which way each gripper opens (nothing moves)")
        if name in {"calibrate", "grippers"}:
            action.add_argument("--no-save", action="store_true", help="report without saving")
        if name == "init":
            action.add_argument("--example", type=Path, help="write an example cell config here and stop")
        if name in {"approve", "decline"}:
            action.add_argument("--id", help="the question's id (`ora cell pending`); required with --json or no terminal")
            action.add_argument("--by", default="ora cell", help="who answers, kept with the answer")
        if name == "approve":
            action.add_argument("--value", help="the answer to a value question (a measured length, ...)")
        if name == "decline":
            action.add_argument("--reason", help="why not; a declined motion's reason reaches the agent")
        if name == "auto-approve":
            action.add_argument("state", nargs="?", choices=("on", "off", "status"), default="status",
                                help="on: motions run without asking (enabling and releasing are still asked); "
                                     "off; status (default)")
            action.add_argument("--by", default="ora cell", help="who turns it on, kept with every auto-approval")
        if name == "stop-motion":
            action.add_argument("--by", default="ora cell", help="who stops it, kept with the stop")
            action.add_argument("--withdraw", action="store_true", help="take back a stop nobody has read yet")
        if name == "sim":
            action.add_argument("what", choices=("init", "status", "place", "clear", "operator"),
                                help="init PATH: write a simulated cell config; status: the truth against the "
                                     "calibration; place X Y: the board's middle at (X, Y) m on the table; clear: "
                                     "the board off the table; operator: answer the cell's questions until stopped")
            action.add_argument("values", nargs="*", help="init: the config's path; place: X Y in metres")
            action.add_argument("--yaw", type=float, default=0.0, help="place: the board's turn, degrees")
            action.add_argument("--seconds", type=float, help="operator: stop after this long (default: Ctrl-C)")
            action.add_argument("--python", help="init: the interpreter with MuJoCo and i2rt (default: this one)")
        if name == "mode":
            action.add_argument("state", nargs="?", choices=("ask", "safe", "auto", "status"), default="status",
                                help="ask: every motion waits for a person; safe: a motion inside the safe limits "
                                     "runs without asking; auto: every motion of the running worker does; "
                                     "status (default). Enabling and releasing the arms are always asked")
            action.add_argument("--by", default="ora cell", help="who sets it, kept with every self-answered motion")


def _board(port: int, cell_id: str) -> str | None:
    """The URL of a cell board serving this cell on this machine, or None."""

    url = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(url + "/api/status", timeout=2.0) as reply:
            status = json.loads(reply.read())
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return url if status.get("cell_id") == cell_id else None


TAKE_OVER = {"home", "hold", "float"}
BOARD_START_S = 30.0


def _start_board(port: int, cell: Any) -> str:
    """A cell board started in the background (it then holds the arms for as long as needed)."""

    import subprocess
    import time

    from ora.cell.errors import CellError

    root = Path(__file__).resolve().parents[2]
    log = Path("~/.cache/ora/cell-board.log").expanduser()
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}
    with open(log, "ab") as out:
        subprocess.Popen([sys.executable, "-m", "ora.app", "cell", "board", "--cell", str(cell.config.path),
                          "--port", str(port)], stdout=out, stderr=out, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env, cwd=root)
    end = time.monotonic() + BOARD_START_S
    while time.monotonic() < end:
        url = _board(port, cell.config.cell_id)
        if url is not None:
            return url
        time.sleep(0.5)
    raise CellError(f"the cell board did not start on port {port} (log: {log})")


def _via_board(url: str, verb: str) -> Any:
    request = urllib.request.Request(f"{url}/api/reset/{verb}", method="POST", data=b"")
    try:
        with urllib.request.urlopen(request, timeout=120.0) as reply:
            return json.loads(reply.read())["result"]
    except urllib.error.HTTPError as exc:
        from ora.cell.errors import CellError

        raise CellError(json.loads(exc.read() or b"{}").get("error", str(exc))) from None


def _confirm(action: str, args: argparse.Namespace) -> bool:
    if args.yes or action not in CONFIRM:
        return True
    if action == "calibrate" and getattr(args, "method", None) in {"charuco", "wrists", "board-relative"}:
        return True                                # these only look; nothing moves
    if action == "calibrate" and getattr(args, "method", None) == "hand-eye":
        return True                                # it asks before every motion itself
    if action == "grippers" and getattr(args, "orient", False):
        return True                                # nothing moves
    if action == "setup":
        return True                                # it asks before every step that moves
    from ora.cell.operator.actions import ask_terminal

    return ask_terminal(CONFIRM[action])


def _setup(action: str, args: argparse.Namespace, cell: Any) -> Any:
    from ora.cell.errors import CellError
    from ora.cell.operator import motion, tools

    save = not getattr(args, "no_save", False)
    if action == "init":
        connected = cell.camera_driver.connected()
        return {"cell_id": cell.config.cell_id, "robot": cell.adapter.model, "config": str(cell.config.path),
                "calibration": str(cell.config.calibration_path),
                "buses": {arm: cell.bus.interface(ch).get("state", "missing") for arm, ch in cell.adapter.buses.items()},
                "cameras": {role: ("connected" if c.serial in connected else "NOT CONNECTED")
                            for role, c in cell.config.cameras.items()}}
    if action == "check":
        from ora.cell.lock import CellLock
        from ora.cell.occupancy import check_occupancy

        with CellLock("ora cell check"):
            occupancy = check_occupancy(mode="read", channels=cell.channels, serials=cell.config.serials.values(),
                                        ignore_processes=cell.config.ignore_processes,
                                        markers=cell.adapter.foreign_controller_markers,
                                        bus=cell.bus, busy=cell.camera_driver.busy)
            frames = motion.capture(cell, list(cell.config.cameras), frames=1)
        return {"occupancy": occupancy, "depth_valid": {r: round(f["depth_valid_fraction"], 3) for r, f in frames.items()}}
    if action == "read":
        return tools.read_state(cell)
    if action == "record-rest":
        return tools.record_rest(cell)
    if action == "setup":
        from ora.cell.operator import setup

        if args.status:
            return setup.status(cell)

        def event(value: dict[str, Any]) -> None:
            if args.json:
                print(json.dumps(dict(value), ensure_ascii=False), file=sys.stderr, flush=True)
            else:
                print(f"setup: {value['step']} {value['state']}" + (f": {value['why']}" if value.get("why") else ""),
                      file=sys.stderr, flush=True)

        return setup.run(cell, _asker(args, "ora cell setup"), redo=tuple(args.redo), only=tuple(args.only),
                         progress=_progress(args), event=event)
    if action == "grippers":
        from ora.cell.operator.tools import orient_grippers

        if args.orient:
            return orient_grippers(cell, _asker(args, "ora cell grippers --orient"))
        found = motion.grippers(cell, save=save)
        if save and found["gripper_limits"] and not found["stopped"]:
            found["orientation"] = orient_grippers(cell, _asker(args, "ora cell grippers"))
        return found
    if action == "calibrate":
        if args.method == "arm-poses":
            return motion.calibrate_cameras(cell, save=save)
        if args.method == "board-poses":
            if not args.square_mm or not args.pattern:
                raise CellError("--square-mm and --pattern COLSxROWS (inner corners) are required")
            columns, rows = (int(v) for v in args.pattern.lower().split("x"))
            return motion.calibrate_with_board(cell, square_mm=args.square_mm, pattern=(columns, rows), save=save)
        if args.method == "board-relative":
            if not (args.square_mm and args.pattern and args.role and args.reference):
                raise CellError("--role, --reference, --square-mm and --pattern COLSxROWS are required")
            columns, rows = (int(v) for v in args.pattern.lower().split("x"))
            return tools.board_relative(cell, args.role, args.reference, square_mm=args.square_mm,
                                        pattern=(columns, rows), save=save)
        if args.method == "charuco":
            if not args.square_mm:
                raise CellError("--square-mm is required: measure one black square of the board as shown")
            return tools.charuco(cell, square_mm=args.square_mm, save=save)
        if args.method == "hand-eye":
            return _hand_eye(args, cell, save)
        if args.method == "wrists":
            if not args.role:
                raise CellError("--role is required (a fixed camera)")
            return tools.fixed_camera_from_wrists(cell, args.role, save=save)
        raise CellError("--method arm-poses|board-poses|board-relative|charuco|wrists is required")
    raise CellError(f"unknown action {action}")


def _progress(args: argparse.Namespace) -> Callable[[str], None]:
    """Progress on stderr: one ``{"progress": ...}`` JSON line each with ``--json``."""

    def progress(text: str) -> None:
        print(json.dumps({"progress": text}, ensure_ascii=False) if args.json else f"{args.action}: {text}",
              file=sys.stderr, flush=True)

    return progress


def _asker(args: argparse.Namespace, label: str) -> Any:
    """Where a setup step asks the person: ``--ask``, else this terminal when there is one."""

    from ora.cell.operator import asking

    where = args.ask or ("terminal" if _interactive() and not args.json else "broker")
    return asking.TerminalAsker() if where == "terminal" else asking.BrokerAsker(label)


def _hand_eye(args: argparse.Namespace, cell: Any, save: bool) -> Any:
    from ora.cell.errors import CellError
    from ora.cell.operator import handeye

    pattern = None
    if args.pattern:
        pattern = handeye.parse_pattern(args.pattern)
        if pattern is None:
            raise CellError(f"--pattern {args.pattern}: give the inner corners as COLSxROWS, e.g. 11x8")
    asker = _asker(args, "ora cell calibrate --method hand-eye")
    progress = _progress(args)

    return handeye.calibrate(cell, asker=asker, arms=args.arms.split(",") if args.arms else None,
                             placements=args.placements or handeye.PLACEMENTS, square_mm=args.square_mm,
                             pattern=pattern, records=args.records or (), save=save, progress=progress)


def _verb(action: str, args: argparse.Namespace, cell: Any) -> Any:
    from ora.cell.errors import CellError
    from ora.cell.operator import actions

    board = _board(args.board_port, cell.config.cell_id)
    if board is None and action in TAKE_OVER:
        # Taking the arms over needs a holder that outlives this command: a blocked drive
        # home holds where the arms are, and a process that exited would let them fall.
        board = _start_board(args.board_port, cell)
    if board is not None:
        return _via_board(board, action)
    yes: Callable[[str], bool] = lambda _q: True  # noqa: E731 - confirmed above
    verbs = {"release": lambda: actions.release_all(cell, yes),
             "estop": lambda: actions.estop(cell), "diagnose": lambda: actions.diagnose(cell), "stop": actions.stop,
             "kill": lambda: actions.kill(yes), "cameras": lambda: actions.reset_cameras(cell),
             "can": lambda: actions.reset_buses(cell)}
    return verbs[action]()


def _interactive() -> bool:
    return sys.stdin.isatty()


def _describe(request: dict[str, Any]) -> str:
    import time

    kinds = {"enable": "enable the arms", "motion": "motion", "release": "release in place",
             "task": "do by hand, then approve", "value": "answer with --value", "write": "save the result"}
    left = max(0.0, float(request.get("expires_unix_s") or 0.0) - time.time())
    head = f"[{request['id']}] {kinds.get(request.get('kind'), request.get('kind'))} · {request.get('title', '')}"
    lines = [f"{head} · {left:.0f} s left"]
    lines += ["  " + line for line in str(request.get("text") or "").splitlines()]
    return "\n".join(lines)


def _describe_auto(state: dict[str, Any]) -> str:
    if state["on"]:
        return (f"auto-approve is on: {' and '.join(state['kinds'])} questions of pid {state['pid']} answer "
                f"themselves (set by {state['by']}, for {state['run']}); enabling and releasing the arms are "
                "still asked")
    stale = f" ({state['stale']})" if state.get("stale") else ""
    return f"auto-approve is off{stale}: every question waits for an answer"


def _describe_mode(state: dict[str, Any]) -> str:
    limits = state["safe_limits"]
    words = {"ask": "ask: every motion waits for a person",
             "safe": (f"safe: a motion of at most {limits['max_duration_s']:g} s and "
                      f"{limits['max_joint_travel_rad']:g} rad per joint, at least {limits['min_table_distance_m'] * 100:g} cm "
                      f"above the table and {limits['min_inter_arm_distance_m'] * 100:g} cm from the other arm runs "
                      "without asking; any other waits"),
             "auto": f"auto: every motion of pid {state['pid']} runs without asking (back to "
                     f"{state.get('falls_back_to')} when it ends)"}
    stale = f" ({state['stale']})" if state.get("stale") else ""
    return f"permission mode {words[state['mode']]}; enabling and releasing the arms are always asked{stale}"


def _answer(args: argparse.Namespace) -> Any:
    """``pending``, ``approve``, ``decline`` and ``auto-approve``: the approval files only (no lock taken, no
    board, no robot)."""

    from ora.cell.errors import CellError
    from ora.cell.operator import approval

    if args.action == "auto-approve":
        state = (approval.auto_approve_state() if args.state == "status"
                 else approval.switch_auto_approve(args.state == "on", by=args.by))
        return state if args.json else _describe_auto(state)
    if args.action == "stop-motion" and args.withdraw:
        withdrawn = approval.withdraw_stop_request()
        return {"withdrawn": withdrawn} if args.json else ("stop withdrawn" if withdrawn else "no stop was pending")
    if args.action == "stop-motion":
        record = approval.request_stop_motion(by=args.by)
        return record if args.json else f"stop sent to pid {record['pid']}: its motion ends where the arms are"
    if args.action == "mode":
        state = approval.mode_state() if args.state == "status" else approval.switch_mode(args.state, by=args.by)
        return state if args.json else _describe_mode(state)
    waiting = approval.pending()
    if args.action == "pending":
        if args.json:
            return waiting
        return "\n\n".join(_describe(request) for request in waiting) or "nothing is waiting for an answer"
    approved = args.action == "approve"
    request_id = args.id
    if request_id is None:
        if args.json or not _interactive():
            raise CellError("--id is required here (`ora cell pending` lists the questions)")
        if not waiting:
            raise CellError("nothing is waiting for an answer")
        if len(waiting) > 1:
            raise CellError("more than one question is waiting: pass --id\n"
                            + "\n".join(_describe(request).splitlines()[0] for request in waiting))
        print(_describe(waiting[0]))
        from ora.cell.operator.actions import ask_terminal

        if not ask_terminal(f"{'Approve' if approved else 'Decline'} {waiting[0]['id']}?"):
            raise CellError("not answered: nothing sent")
        request_id = waiting[0]["id"]
    return approval.answer(request_id, approved, getattr(args, "reason", None), by=args.by,
                           value=getattr(args, "value", None))


def run(args: argparse.Namespace) -> int:
    if args.action == "init" and getattr(args, "example", None):
        args.example.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.example.expanduser().write_text(EXAMPLE)
        print(f"wrote an example cell config to {args.example}: fill in the serials, then `ora cell init --cell {args.example}`")
        return 0
    if args.action == "depth-service":
        from ora.cell.depth.service import main as depth_main

        return depth_main(args.rest)
    if args.action == "sim" and args.what == "init":
        from ora.cell.errors import CellError

        try:
            return _print(args, _sim_init(args))
        except CellError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}) if args.json else f"refused: {exc}", file=sys.stderr)
            return 1
    if args.action not in ANSWERS:
        _reexec_in_robot_interpreter(args)
    _isolate_simulated(args)
    if args.action in INTERRUPTIBLE:
        _take_interrupts(say=args.json)
    from ora.cell.errors import CellError
    from ora.cell.operator.context import Cell

    try:
        if args.action in ANSWERS:
            return _print(args, _answer(args))
        cell = Cell.load(args.cell)
        if args.action == "board":
            from ora.cell.operator.board import DEFAULT_PORT, serve

            return serve(cell, host=args.host, port=args.port or DEFAULT_PORT, depth_service=args.depth_service)
        if not _confirm(args.action, args):
            raise CellError("not confirmed: nothing done")
        # With --json, stdout carries the one result object: what the drivers print goes to stderr.
        with contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext():
            if args.action in SIM:
                result = _sim(args, cell)
            else:
                result = _setup(args.action, args, cell) if args.action in SETUP else _verb(args.action, args, cell)
    except CellError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"refused: {exc}", file=sys.stderr)
        return 1
    return _print(args, result)


def _print(args: argparse.Namespace, result: Any) -> int:
    if args.json:
        print(json.dumps({"ok": True, "result": result}, default=_plain))
    elif isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, default=_plain, ensure_ascii=False))
    return 0


def _isolate_simulated(args: argparse.Namespace) -> None:
    """A simulated cell keeps its lock, approvals and alerts in its state directory (``config.isolate``)."""

    from ora.cell import config as cell_config
    from ora.cell.errors import CellError

    try:
        config = cell_config.load(args.cell)
    except CellError:
        return                                   # the command reports the config problem itself
    chosen = cell_config.isolate(config)
    if chosen and not args.json:
        print(f"simulated cell {config.cell_id}: lock and approvals in {config.sim_state}", file=sys.stderr)


def _sim_init(args: argparse.Namespace) -> Any:
    from ora.cell.errors import CellError

    if len(args.values) != 1:
        raise CellError("ora cell sim init PATH: where to write the simulated cell's config")
    path = Path(args.values[0]).expanduser().resolve()
    if path.exists():
        raise CellError(f"{path} exists: nothing written")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SIM_EXAMPLE.format(state=path.with_suffix(".sim"), python=args.python or sys.executable,
                                       calibration=path.with_suffix(".calibration.json")))
    return (f"wrote a simulated cell config to {path}; its truth is written to {path.with_suffix('.sim')}/world.json "
            f"by the first command. Next: ora cell setup --cell {path}")


def _sim(args: argparse.Namespace, cell: Any) -> Any:
    import threading

    import numpy as np

    from ora.cell.errors import CellError

    sim = getattr(cell.adapter, "sim", None)
    if sim is None:
        raise CellError(f"{cell.config.path} is not a simulated cell (robot.sim)")
    if args.what == "status":
        return sim.status(cell.calibration())
    if args.what == "place":
        if len(args.values) != 2:
            raise CellError("ora cell sim place X Y [--yaw DEG]: the board's middle on the table, in metres")
        pose = sim.state.place_board([float(v) for v in args.values], args.yaw, "ora cell sim place")
        return {"board2world": np.round(pose, 4).tolist()} if args.json else f"board placed: {np.round(pose[:3, 3], 3).tolist()}"
    if args.what == "clear":
        sim.state.clear_board()
        return "board off the table"
    stop = threading.Event()
    if args.seconds:
        timer = threading.Timer(args.seconds, stop.set)
        timer.daemon = True
        timer.start()
    try:
        answered = sim.operate(stop, say=lambda m: print(m, file=sys.stderr, flush=True))
    except KeyboardInterrupt:
        stop.set()
        return "simulated operator stopped"
    return f"simulated operator answered {answered} questions"


def _reexec_in_robot_interpreter(args: argparse.Namespace) -> None:
    """Run the rest of this command in the interpreter that has the robot's drivers.

    The cell config names it (``robot.python``); ``ora`` itself usually runs in the
    project's own environment, which has none of them."""

    if os.environ.get("ORA_CELL_REEXEC"):
        return
    from ora.cell import config as cell_config
    from ora.cell.errors import CellError

    try:
        python = cell_config.load(args.cell).python
    except CellError:
        return                                   # the command reports the config problem itself
    # Virtual environments link the same base interpreter: compare environments, not binaries.
    if python is None or not python.exists() or python.parent.parent.resolve() == Path(sys.prefix).resolve():
        return
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "ORA_CELL_REEXEC": "1",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}
    os.execve(str(python), [str(python), "-m", "ora.app", "cell", *sys.argv[sys.argv.index("cell") + 1:]], env)


#: Long steps a front end stops with SIGINT (the board's 中止, ``ora play --setup``).
INTERRUPTIBLE = frozenset({"calibrate", "setup", "grippers"})


def _take_interrupts(say: bool = False) -> None:
    """SIGINT raises KeyboardInterrupt again. A process started in the background by a shell
    (``&``, ``nohup``: the board, a supervisor) inherits SIGINT ignored, and so would ignore
    the stop its front end sends between motions.

    ``say`` (``--json``) then prints ``{"interruptible": true}`` on stderr: a SIGINT sent before
    it may be lost (ignored, or caught by the parent's handler before exec), so the board holds a
    stop asked earlier until it reads this line."""

    import signal

    signal.signal(signal.SIGINT, signal.default_int_handler)
    if say:
        print(json.dumps({"interruptible": True}), file=sys.stderr, flush=True)


def _plain(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "__dict__"):
        return value.__dict__
    return np.asarray(value).tolist()
