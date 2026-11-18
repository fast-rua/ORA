"""The YAM cell's start-up self-check: every check before the arms are enabled, with its fix.

``RealRig`` runs it on ``read`` and ``on`` from what it has measured at start (the passive motor
reads, the joints against the model, the table seen through the wrists, one frame of every camera)
and records the result in its start-up report (``rig-startup.json``). On ``on`` a failed check
refuses the start before anything is enabled, and the refusal lists every failed check with what
to do, not only the first. A check that cannot be made says why (``passed`` None) and does not
refuse. ``read`` answers the same checks with nothing enabled, so a session there is the
dry run of a real start.

The checks came from one evening (2026-11-14): a gripper motor zero moved over a power cycle, the
table stood 10 cm higher than calibrated, a wrist camera streamed no frames, joints read just past
their limits at rest. Each was found by a motion that failed, or by a broken fingertip.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from ora.cell import calibration as cell_calibration

from .health import TEMPERATURE_WARN_C

#: A calibration older than this is named (it does not refuse): the cell may have moved since.
CALIBRATION_AGE_WARN_DAYS = 14.0

ARMS_CN = {"left": "左臂", "right": "右臂"}


def _check(check_id: str, title: str, passed: bool | None, detail: str, fix: str | None = None) -> dict[str, Any]:
    return {"id": check_id, "title": title, "passed": passed, "detail": detail, "fix": None if passed else fix}


def startup_checks(*, readings: Mapping[str, Mapping[str, Any]], joints: Mapping[str, Any] | None,
                   wrist_table: Mapping[str, Mapping[str, Any]] | None,
                   calibration: Mapping[str, Any], cameras: Mapping[str, Mapping[str, Any]] | None,
                   now: float | None = None) -> list[dict[str, Any]]:
    """Every start-up check, in the order a person would fix them."""

    now = time.time() if now is None else now
    out = []
    errors = {arm: dict(reading.get("errors") or {}) for arm, reading in readings.items() if reading.get("errors")}
    out.append(_check("motors_ok", "电机没有故障", not errors,
                      "所有电机正常" if not errors else f"电机报错 {errors}",
                      "断电重启这条手臂，再跑 `ora cell read` 确认故障已清除"))
    temperatures = {arm: float(reading.get("max_temperature_c", 0.0)) for arm, reading in readings.items()}
    hot = {arm: value for arm, value in temperatures.items() if value >= TEMPERATURE_WARN_C}
    out.append(_check("motors_cool", "电机温度正常", not hot,
                      "最高 " + "，".join(f"{ARMS_CN.get(a, a)} {v:.0f}°C" for a, v in temperatures.items()),
                      f"等电机冷却到 {TEMPERATURE_WARN_C:.0f}°C 以下再启动"))
    if joints is None:
        out.append(_check("joints_in_limits", "关节在限位内", None, "没有读到关节"))
    else:
        outside = joints.get("outside_limits") or {}
        out.append(_check("joints_in_limits", "关节在限位内", not outside,
                          "都在限位内" if not outside else f"超出限位 {outside}",
                          "编码器零点可能变了：断电重启后再读一次；仍然超出就重做手眼标定（`ora cell calibrate --method hand-eye`）"))
    limits = calibration.get("gripper_limits") or {}
    missing = [arm for arm in readings if arm not in limits]
    if missing:
        out.append(_check("grippers_calibrated", "夹爪测过限位", False,
                          "没有记录 " + "、".join(ARMS_CN.get(a, a) for a in missing) + " 的夹爪限位",
                          "跑 `ora cell grippers`"))
    else:
        off = cell_calibration.gripper_readings_off(
            limits, {arm: reading.get("gripper_raw_rad") for arm, reading in readings.items()})
        out.append(_check("grippers_calibrated", "夹爪读数在记录的限位里", not off,
                          "读数都在记录的限位里" if not off else "；".join(
                              f"{ARMS_CN.get(a, a)}读数 {v['reading_rad']:.2f} 不在 [{v['low_rad']:.2f}, {v['high_rad']:.2f}] 里"
                              for a, v in off.items()),
                          "夹爪电机的零点变了（常见于断电重启后）：跑 `ora cell grippers` 重新测"))
    if not wrist_table:
        out.append(_check("table_seen", "腕部相机看到的桌面和标定一致", None, "没有核对"))
    else:
        failed = {arm: check.get("problems") for arm, check in wrist_table.items() if check.get("passed") is False}
        unknown = [arm for arm, check in wrist_table.items() if check.get("passed") is None]
        passed = None if (not failed and unknown) else not failed
        detail = ("一致" if passed else "；".join(f"{ARMS_CN.get(a, a)}: {', '.join(p or [])}" for a, p in failed.items())
                  if failed or passed else "深度不够，没能拟合桌面：" + "、".join(ARMS_CN.get(a, a) for a in unknown))
        out.append(_check("table_seen", "腕部相机看到的桌面和标定一致", passed, detail,
                          "桌子、手臂或相机可能动过：先看手腕正下方有没有东西；桌子换过就跑 `ora cell setup --redo table --only table verify`"))
    if cameras is None:
        out.append(_check("cameras_depth", "每个相机都有画面和深度", None, "没有拍"))
    else:
        low = {role: frame for role, frame in cameras.items() if frame.get("error") or frame.get("depth_below_floor")}
        out.append(_check("cameras_depth", "每个相机都有画面和深度", not low,
                          "，".join(f"{role} {frame['depth_valid_fraction']:.2f}" for role, frame in cameras.items()
                                   if "depth_valid_fraction" in frame) if not low else
                          "；".join(f"{role}: {frame.get('error') or '有效深度 %.2f 太少' % frame['depth_valid_fraction']}"
                                   for role, frame in low.items()),
                          "看相机有没有被挡住、线有没有松；`ora cell check` 会逐个相机重试"))
    measured = (calibration.get("verified") or {}).get("at") or calibration.get("updated_unix_s")
    if measured:
        days = (now - float(measured)) / 86400.0
        out.append(_check("calibration_recent", "标定不太旧", True if days <= CALIBRATION_AGE_WARN_DAYS else None,
                          f"{days:.0f} 天前验证" + ("" if days <= CALIBRATION_AGE_WARN_DAYS else
                                                  f"，超过 {CALIBRATION_AGE_WARN_DAYS:.0f} 天：建议跑 `ora cell setup --status`")))
    return out


def refusal(checks: Sequence[Mapping[str, Any]]) -> str | None:
    """The start's refusal: every failed check with its fix; None when nothing failed."""

    failed = [check for check in checks if check["passed"] is False]
    if not failed:
        return None
    return "the start-up self-check failed, nothing was enabled:\n" + "\n".join(
        f"- {check['title']}：{check['detail']}。怎么办：{check['fix']}" for check in failed)
