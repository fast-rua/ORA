"""The code-mode interpreter held to its lock (``ora/agent/code_mode/sandbox-requirements.txt``).

The lock pins every package ``ORA_CODE_MODE_PYTHON`` holds, with hashes; preflight (``sandbox_status``) imports each in
the real sandbox, refuses an interpreter that differs, naming the lock, the package, both versions and the command that
builds a conforming one, and records the interpreter it checked. Pillow, scikit-image and Shapely work inside the
sandbox. The tests that run the sandbox are skipped where it is unavailable, an interpreter that differs from the lock
included (the skip reason is the refusal).
"""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from ora.agent.code_mode import sandbox
from ora.agent.code_mode.sandbox import LOCK, lock_refusal, locked_packages, sandbox_status

STATUS = sandbox_status()
sandboxed = pytest.mark.skipif(not STATUS["ok"], reason=f"code-mode sandbox unavailable: {STATUS.get('reason')}")
LIMITS = {"wall_s": 60, "cpu_s": 30, "memory_mib": 2048}


def test_the_lock_pins_every_package_with_hashes_and_keeps_numpy_and_scipy() -> None:
    python, packages = locked_packages(LOCK)
    assert python == "3.11"
    assert packages["numpy"] == "1.26.4" and packages["scipy"] == "1.13.1", "tests pin their floats to 1e-12"
    text = LOCK.read_text(encoding="utf-8")
    hashed = re.findall(r"^([\w.-]+)==\S+ \\\n(?:    --hash=sha256:[0-9a-f]{64}(?: \\)?\n)+", text, re.M)
    assert sorted(hashed) == sorted(packages), "every package has its hashes"
    direct = (LOCK.with_suffix(".in")).read_text(encoding="utf-8")
    wanted = {line.split("==")[0] for line in direct.splitlines() if line and not line.startswith("#")}
    assert wanted == {"numpy", "scipy", "pillow", "scikit-image", "shapely"} and wanted <= set(packages)
    assert "uv pip compile sandbox-requirements.in" in text.splitlines()[1], "the header says how it is regenerated"


def test_an_interpreter_that_differs_from_the_lock_is_refused_with_what_differs_and_how_to_build_one() -> None:
    python, packages = locked_packages(LOCK)
    matching = {"executable": "/env/bin/python", "python": "3.11.12",
                "packages": {name: {"version": version, "error": None} for name, version in packages.items()}}
    assert lock_refusal(matching, LOCK) is None
    found = {**matching, "python": "3.12.1", "packages": {
        **matching["packages"], "numpy": {"version": "1.26.3", "error": None}, "pillow": {"version": None, "error": None},
        "shapely": {"version": packages["shapely"], "error": "OSError: libgeos_c.so: cannot open shared object file"}}}
    refusal = lock_refusal(found, LOCK)
    assert refusal.startswith(f"the code-mode interpreter /env/bin/python does not match its lock {LOCK}: ")
    assert "Python: the lock says 3.11, the interpreter has 3.12.1" in refusal
    assert "numpy: the lock says 1.26.4, the interpreter has 1.26.3" in refusal
    assert f"pillow: the lock says {packages['pillow']}, the interpreter has none" in refusal
    assert f"shapely {packages['shapely']} does not import: OSError: libgeos_c.so" in refusal
    assert refusal.endswith(f"Build one that does: uv venv --python 3.11 DIR && uv pip install --python DIR/bin/python "
                            f"--require-hashes --no-deps --only-binary :all: --compile-bytecode -r {LOCK}, then set "
                            "ORA_CODE_MODE_PYTHON=DIR/bin/python")
    assert "scipy" not in refusal and "scikit-image" not in refusal, "only what differs is named"


@sandboxed
def test_preflight_records_the_interpreter_it_checked() -> None:
    from ora.orchestrator.preflight.gates import _code_mode_sandbox

    _, packages = locked_packages(LOCK)
    check = _code_mode_sandbox({"policy": {"code_mode": {"enabled": True}}})
    assert check["check"] == "code_mode_sandbox" and check["passed"] is True
    assert check["detail"]["interpreter"] == STATUS["interpreter"] == {
        "executable": sandbox.python_path(), "python": STATUS["interpreter"]["python"],
        "lock": str(LOCK), "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(), "packages": packages}
    assert STATUS["interpreter"]["python"].startswith("3.11.")


@sandboxed
def test_the_real_sandbox_refuses_an_interpreter_its_lock_does_not_name(tmp_path, monkeypatch) -> None:
    lock = tmp_path / "sandbox-requirements.txt"
    lock.write_text(LOCK.read_text(encoding="utf-8").replace("\nnumpy==1.26.4 ", "\nnumpy==1.26.3 "), encoding="utf-8")
    monkeypatch.setattr(sandbox, "LOCK", lock)
    status = sandbox_status()
    assert status["ok"] is False and status["interpreter"]["packages"]["numpy"] == "1.26.4"
    assert f"does not match its lock {lock}: numpy: the lock says 1.26.3, the interpreter has 1.26.4. " in status["reason"]
    assert status["reason"].endswith(f"-r {lock}, then set ORA_CODE_MODE_PYTHON=DIR/bin/python")


@sandboxed
def test_pillow_scikit_image_and_shapely_work_inside_the_sandbox(tmp_path) -> None:
    """Open and crop an image, measure the regions of a mask, and fit the minimum-area rectangle around points."""
    from ora.agent.code_mode.shell import run_command

    code = ("import json\nimport numpy as np\nfrom PIL import Image\nfrom skimage.measure import label, regionprops\n"
            "from shapely import MultiPoint, oriented_envelope\n"
            "pixels = (np.arange(48 * 64 * 3).reshape(48, 64, 3) % 251).astype(np.uint8)\n"
            "Image.fromarray(pixels).save('/tmp/frame.png')\n"
            "crop = Image.open('/tmp/frame.png').crop((10, 5, 30, 25))\n"
            "mask = np.zeros((40, 60), bool)\nmask[5:15, 10:30] = True\nmask[25:35, 40:45] = True\n"
            "regions = regionprops(label(mask))\n"
            "turn = np.deg2rad(30.0)\nR = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])\n"
            "corners = np.array([[0.1, 0.05], [-0.1, 0.05], [-0.1, -0.05], [0.1, -0.05]]) @ R.T + [0.4, -0.1]\n"
            "inside = np.array([[0.02, 0.01], [-0.05, -0.03], [0.07, 0.0]]) @ R.T + [0.4, -0.1]\n"
            "box = oriented_envelope(MultiPoint(np.vstack([corners, inside]).tolist()))\n"
            "ring = np.array(box.exterior.coords)\n"
            "print(json.dumps({'crop': [crop.size, crop.mode, bool(np.array_equal(np.asarray(crop), pixels[5:25, 10:30]))],\n"
            " 'regions': [[float(r.area), [int(v) for v in r.bbox], [float(v) for v in r.centroid]] for r in regions],\n"
            " 'box': [box.area, sorted(np.linalg.norm(np.diff(ring, axis=0), axis=1).tolist()),\n"
            "         sum(sorted(ring[:-1].round(9).tolist()), [])],\n"
            " 'corners': sum(sorted(corners.round(9).tolist()), [])}))")
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "program.py").write_text(code)
    outcome = run_command("python program.py", handle_call=lambda *_: {"ok": False}, workspace=workspace, **LIMITS)
    assert outcome.status == "completed", outcome.output
    found = json.loads(outcome.output)
    assert found["crop"] == [[20, 20], "RGB", True]
    assert found["regions"] == [[200.0, [5, 10, 15, 30], [9.5, 19.5]], [50.0, [25, 40, 35, 45], [29.5, 42.0]]]
    area, sides, ring = found["box"]
    assert area == pytest.approx(0.02, abs=1e-12) and sides == pytest.approx([0.1, 0.1, 0.2, 0.2], abs=1e-12)
    assert ring == pytest.approx(found["corners"], abs=1e-9)
    command = run_command("python -c 'import PIL.Image, skimage.measure, shapely; "
                          "print(PIL.__version__, skimage.__version__, shapely.__version__)'",
                          handle_call=lambda *_: {"ok": False}, workspace=workspace, **LIMITS)
    _, packages = locked_packages(LOCK)
    assert command.status == "completed", command.output
    assert command.output.split() == [packages["pillow"], packages["scikit-image"], packages["shapely"]]


@sandboxed
def test_a_program_reads_the_source_and_cannot_change_or_import_it_or_see_the_judge(tmp_path: Path) -> None:
    from ora.agent.code_mode.library import SOURCE_MOUNT
    from ora.agent.code_mode.shell import run_command

    workspace = tmp_path / "work"
    workspace.mkdir()
    toolkit = f"{SOURCE_MOUNT}/ora/agent/toolkit"
    script = (f"grep -c def {toolkit}/tools/move_pose/tool.py; "
              f"grep -rl 'cm inside ' {SOURCE_MOUNT}; "
              f"grep -l finger_1 {SOURCE_MOUNT}/ora/policy/embodiments/*/manual-public-geometry-*.md; "
              f"touch {toolkit}/lib/README.md 2>&1 || echo refused; "
              "python -c 'import ora' 2>&1 | tail -1")
    outcome = run_command(script, handle_call=lambda *_: {}, wall_s=30, cpu_s=30, memory_mib=2048, workspace=workspace,
                          extended=True, source=True)
    assert outcome.status == "completed", outcome.output
    lines = outcome.output.strip().splitlines()
    assert int(lines[0]) > 0
    assert lines[1] == f"{SOURCE_MOUNT}/ora/environment/planners/curobo_v2/worker.py", "a refusal leads to its planner"
    assert f"{SOURCE_MOUNT}/ora/policy/embodiments/arx-x5/manual-public-geometry-v5.md" in lines
    assert "No such file" in "".join(run_command(f"ls {toolkit}/tools/check_task 2>&1", handle_call=lambda *_: {},
                                                 wall_s=30, cpu_s=30, memory_mib=2048, workspace=workspace, extended=True,
                                                 source=True).output)
    assert "refused" in outcome.output and "ModuleNotFoundError" in outcome.output
