"""The YAM worker's words for the person at the cell: the questions the approval broker shows on the cell board
(``ora/cell/operator/approval.py``), in Chinese, each a (title, text) pair, and the plan behind a motion question.
``real_rig.py`` asks them; nothing here decides anything."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np

from .health import ARM_CN


def plan_text(summary: Mapping[str, Any]) -> str:
    fields = {key: summary[key] for key in ("arms", "duration_s", "min_inter_arm_distance_m", "min_table_distance_m",
                                            "max_joint_travel_rad", "gripper", "joint_space") if key in summary}
    return json.dumps(fields, default=lambda v: np.round(np.asarray(v, float), 3).tolist())


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _arms_cn(arms: Any) -> str:
    names = [ARM_CN.get(arm, str(arm)) for arm in arms]
    return "两臂" if len(names) == 2 else "、".join(names) or "手臂"


def motion(summary: Mapping[str, Any]) -> tuple[str, str]:
    """A motion plan in plain words for the operator (title, text), then the plan itself."""

    if "gripper" in summary:
        goals = sorted(dict(summary["gripper"]).items())
        title = "夹爪动作等待批准"
        line = "，".join(f"{ARM_CN.get(arm, arm)}夹爪{'闭合（夹住）' if goal == 'close' else '张开'}"
                        for arm, goal in goals) + "；手臂本身不动。"
    else:
        arms = list(summary.get("arms") or [])
        who = _arms_cn(arms)
        title = f"{who}动作等待批准"
        how = "沿关节直线回到之前的安全姿态" if summary.get("joint_space") else "移动到智能体选的位置"
        duration = _finite(summary.get("duration_s"))
        facts = []
        table = summary.get("min_table_distance_m")
        for arm in arms:
            distance = _finite(table.get(arm)) if isinstance(table, Mapping) else None
            if distance is not None:
                facts.append(f"{ARM_CN.get(arm, arm)}离桌面最近 {distance * 100:.0f} 厘米")
        between = _finite(summary.get("min_inter_arm_distance_m"))
        if between is not None:
            facts.append(f"两臂之间最近 {between * 100:.0f} 厘米")
        travel = [_finite(v) for v in dict(summary.get("max_joint_travel_rad") or {}).values()]
        travel = [v for v in travel if v is not None]
        if travel:
            facts.append(f"关节最多转 {math.degrees(max(travel)):.0f}°")
        line = (f"{who}{how}" + (f"，约 {duration:.1f} 秒" if duration is not None else "")
                + ("；" + "，".join(facts) if facts else "") + "。")
    text = "\n".join([line, "批准 = 执行这个动作；拒绝 = 不动，你写的原因会转告智能体。", "计划：" + plan_text(summary)])
    return title, text


def enable(*, at_rest: bool | None, grippers_known: bool) -> tuple[str, str]:
    lines = ["通电后两臂锁在现在的姿态；之后智能体的每个动作还会单独请你批准。"]
    if not at_rest:
        lines.append("注意：手臂不在记录的静止位（或还没记录静止位）。")
    if not grippers_known:
        lines.append("注意：夹爪限位还没标定，通电时会把两个夹爪各开合一次来测量，手别放在夹爪之间。")
    lines.append("批准 = 通电并保持；拒绝 = 不通电，本次运行停止。")
    return "要给两条手臂通电吗？", "\n".join(lines)


def release(*, blocked: list[str], path_blocked: bool, attempt: int, attempts: int, timeout_s: float,
            home_limit_s: float, last: bool) -> tuple[str, str]:
    if path_blocked:
        why = "回起始位的直线路径会碰到桌面或另一条手臂，所以没有动"
    elif blocked:
        why = f"{_arms_cn(blocked)}跟不上指令（可能被挡住或卡住）"
    else:
        why = f"{home_limit_s:g} 秒内没回到起始位"
    lines = [f"本次运行结束，收尾时手臂回起始位没成功：{why}。手臂现在通电停在原地。",
             "现在释放，手臂会软掉落下。批准 = 就地释放（先用手扶住两臂）。"]
    if last:
        lines.append(f"这是最后一次询问：拒绝或 {timeout_s:.0f} 秒内不答复，仍会就地释放"
                     "（运行结束后没有程序控制电机，手臂反正会软掉）。")
    else:
        lines.append(f"拒绝，或 {timeout_s:.0f} 秒内不答复 = 继续保持，再试一次回起始位"
                     f"（第 {attempt}/{attempts} 次）。")
    lines.append("想让手臂一直保持：点看板的“回到起始位”（或 /home、ora cell home），约 20 秒后看板接管手臂并一直保持。")
    return "手臂没能回到起始位", "\n".join(lines)
