"""Record fresh sensor diagnostics; tools consume the original published bytes."""

from .live_boundary import verified
from .selection import read_local
from .sensor_metrics import measure
from ora.agent.toolkit.interface.types import (
    CorruptRecord,
    Divergence,
    canonical,
    decode,
    digest,
)


def prepare(session, source, output, boundary):
    """Physical checks gate authority; re-rendered pixels are diagnostic only.

    Frozen candidates and tool inputs bind the recorded publication. Fresh
    captures never replace those bytes, including when they happen to match.
    """
    expected = boundary["expected"]
    original_ref = boundary["snapshot"]["available_at_boundary"]["observation"][
        "manifest"
    ]
    original = decode(verified(source, original_ref))
    original_root = (source / original_ref["path"]).parent
    capture_root = session.output_root.parent / "observations"
    report = dict(
        schema="ora.toolkit.live-sensors.v2",
        blocking=False,
        gate="physical_only",
        passed=False,
        sensor_byte_equal=False,
        differences=[],
        files=[],
        tool_observation_source="recorded_publication",
        tool_observation=original_ref,
        historical_identity_retained=True,
        fresh_capture_used_for_tools=False,
        fresh_manifest=None,
        warning="未逐字节复现的传感器",
        render_history=boundary.get("render_history", {}),
    )
    try:
        response = session.diagnostic("prepare", destination=str(capture_root))
        actual = response["observation"]
        oid = actual["observation_id"]
        fresh_root = capture_root / oid
        if not fresh_root.resolve().is_relative_to(capture_root.resolve()):
            raise CorruptRecord("fresh observation escapes capture root")
        report["fresh_manifest"] = str(
            (fresh_root / "observation.json").relative_to(output)
        )
        for key in ("observation_id", "sequence", "round_index", "proprioception"):
            if original.get(key) != actual.get(key):
                report["differences"].append(key)
        if set(original["cameras"]) != set(actual["cameras"]):
            report["differences"].append("cameras")
        for camera in sorted(set(original["cameras"]) | set(actual["cameras"])):
            old_files = original["cameras"].get(camera, {}).get("files", {})
            new_files = actual["cameras"].get(camera, {}).get("files", {})
            for role in sorted(set(old_files) | set(new_files)):
                item = dict(
                    camera=camera, role=role, byte_equal=False, content_equal=False
                )
                try:
                    left = verified(original_root, old_files[role])
                    right = verified(fresh_root, new_files[role])
                    item.update(
                        original_sha256=digest(left),
                        fresh_sha256=digest(right),
                        byte_equal=left == right,
                        content_equal=left == right,
                        metrics=measure(role, left, right),
                    )
                except Exception as error:
                    item["error"] = dict(type=type(error).__name__, message=str(error))
                if not item["byte_equal"]:
                    report["differences"].append(f"cameras.{camera}.{role}")
                report["files"].append(item)
        report["sensor_byte_equal"] = (
            bool(report["files"]) and not report["differences"]
        )
        report["passed"] = report["sensor_byte_equal"]
        report["status"] = "byte_equal" if report["passed"] else "not_byte_equal"
        if report["passed"]:
            report["warning"] = None
    except Exception as error:
        report.update(
            status="unavailable",
            error=dict(type=type(error).__name__, message=str(error)),
        )
    history_path = session.output_root / "render-history-replay.json"
    report["render_history_replay"] = (
        decode(history_path.read_bytes()) if history_path.is_file() else None
    )
    (output / "sensor-verification.json").write_bytes(canonical(report) + b"\n")
    # A failed capture or a pixel difference never relaxes the physical guard.
    after = session.diagnostic("verify_state")
    if any(
        after.get(key) != expected[key]
        for key in (
            "round_index",
            "physics_clock",
            "sim_steps",
            "native_actions",
            "proprioception",
        )
    ):
        raise Divergence(
            "observation verification changed the restored physical boundary"
        )
    session.diagnostic("authorize")
    return report


def copy_publication(source, output, ref):
    """Copy only the manifest-declared public sensor members, preserving bytes."""
    manifest_bytes = verified(source, ref)
    manifest = decode(manifest_bytes)
    root = (source / ref["path"]).parent
    oid = manifest["observation_id"]
    if "/" in oid or "\\" in oid or oid in {".", ".."}:
        raise CorruptRecord("invalid observation identity")
    target = output / "policy-observations" / oid
    target.mkdir(parents=True, exist_ok=True)
    (target / "observation.json").write_bytes(manifest_bytes)
    (target / "proprioception.json").write_bytes(
        read_local(root, "proprioception.json")
    )
    for camera in manifest["cameras"].values():
        for item in camera["files"].values():
            data = verified(root, item)
            path = target / item["path"]
            if not path.resolve().is_relative_to(target.resolve()):
                raise CorruptRecord("camera member escapes publication")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return target, manifest
