"""Fresh native-tick observations use detached snapshot APIs and release the worker."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from ora.app.main import main
from ora.agent.toolkit.interface.types import canonical, decode, digest
from ora.orchestrator.backends.registry import _LOADED, backend
from ora.orchestrator.episode.execution_state import ExecutionStateOwner
from ora.orchestrator.toolkit.snapshot_replay import experiment
from tests.support.agent.toolkit_native_snapshot import make_native


class FakeSession:
    def __init__(self, source, output, boundary):
        self.source, self.output, self.boundary = source, output, boundary
        self.calls = []
        self.failure = None
        self.execution_state_owner = ExecutionStateOwner()
        self.execution_state_owner.writer.update(boundary["exec_values"])

    def diagnostic(self, method, **params):
        self.calls.append(method)
        expected = self.boundary["expected"]
        if method == "prepare":
            if self.failure == "render":
                raise RuntimeError("renderer failed")
            document = decode((self.source / f"state/steps/step-{self.boundary['base_step']:04d}/snapshot.json").read_bytes())
            original = self.source / document["values"]["env.observation"]["manifest"]["path"]
            manifest = decode(original.read_bytes())
            manifest["observation_id"] = f"native-test-{expected['native_actions']}"
            manifest["sequence"] = expected["observation_sequence"]
            manifest["proprioception"] = expected["proprioception"]
            target = Path(params["destination"]) / manifest["observation_id"]
            shutil.copytree(original.parent, target)
            from PIL import Image
            import numpy as np
            ref = manifest["cameras"]["head"]["files"]["rgb_lossless_png"]
            image = target / ref["path"]
            pixels = np.array(Image.open(image))
            pixels[:] = [17, 31, 53]
            Image.fromarray(pixels).save(image)
            ref.update(sha256=digest(image.read_bytes()), bytes=image.stat().st_size)
            (target / "observation.json").write_bytes(canonical(manifest))
            (target / "proprioception.json").write_bytes(canonical(manifest["proprioception"]))
            return dict(observation=manifest)
        if method == "verify_state":
            actual = deepcopy(expected)
            if self.failure == "physics":
                actual["physics_clock"] += 1
            return actual
        if method == "authorize":
            return dict(verified=True)
        raise AssertionError(method)

    def status(self):
        assert "authorize" in self.calls
        self.calls.append("status")
        return dict(terminal=False, reason=None)


@pytest.fixture
def fake_native(tmp_path, monkeypatch):
    source = make_native(tmp_path / "source", monkeypatch)
    sessions = []
    settings = {}

    @contextmanager
    def factory(**kw):
        assert kw["execute_action"] is False
        session = FakeSession(source, kw["output"], kw["boundary"])
        session.failure = settings.get("failure")
        sessions.append(session)
        try:
            for stage in ("boot_worker", "replay_controls", "verify_state"):
                kw["progress"].start(stage, "CPU-only fake native worker")
            if session.failure == "sigterm":
                import signal
                signal.raise_signal(signal.SIGTERM)
            if session.failure == "restore":
                raise RuntimeError("restore failed")
            yield session
        finally:
            session.calls.append("close")
    monkeypatch.setitem(_LOADED, "robodojo", replace(backend("robodojo"), toolkit_live=factory))
    return source, sessions, settings


@pytest.mark.parametrize("number", [0, 1, 2, 4])
def test_native_script_reads_fresh_pixels_then_actions_and_planners_are_denied(fake_native, tmp_path, number):
    source, sessions, _ = fake_native
    script = tmp_path / "trial.py"
    script.write_text('''from ora.agent.toolkit.lib.observation import observe, load_camera

def run(api, args):
    camera = load_camera(api, observe(api).camera("head"))
    errors = []
    for operation in (lambda: api.act.execute_action(action={}, budget=1),
                      lambda: api.planner.probe_motion(request={})):
        try:
            operation()
        except Exception as error:
            errors.append(type(error).__name__)
    api.state.write("tool.trial", 42)
    return {"pixel": camera.rgb.numpy()[0, 0].tolist(), "errors": errors,
            "marks": api.state.read("tool.marks")}
''')
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result = experiment(source=source, native_action=number, resource_profile="fake", script=script)
    assert result["status"] == "ok", result
    assert result["replay_semantics"] == "native_tick_fresh_render"
    assert result["result"]["pixel"] == [17, 31, 53]
    assert result["result"]["errors"] == ["ActionNotExecuted", "SimulatorRequired"]  # an action is proposed, never run
    assert sessions[-1].calls == ["prepare", "verify_state", "authorize", "status", "close"]
    state = decode((Path(result["output"]) / "native-state.json").read_bytes())
    assert state["complete_action_boundary"] is False
    command = state["values"]["exec.command.left"]
    assert ("pose_target_xyz_xyzw" in command) == (number == 0)
    assert "gripper_directive" in command
    assert len(state["missing_state_fields"]) == (2 if number else 0)
    assert result["missing_state_fields"] == state["missing_state_fields"]
    assert "privileged.ground_truth" not in state["values"]
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", ["restore", "render", "physics"])
def test_native_failure_releases_worker_and_never_authorizes_tool(fake_native, tmp_path, failure):
    source, sessions, settings = fake_native
    settings["failure"] = failure
    script = tmp_path / "never.py"
    script.write_text('def run(api, args):\n    raise AssertionError("tool must not run")\n')
    result = experiment(source=source, native_action=1, resource_profile="fake", script=script)
    assert result["status"] == "error"
    assert result["error"]["type"] != "AssertionError"
    assert sessions[-1].calls[-1] == "close"
    assert "authorize" not in sessions[-1].calls


def test_cli_native_requires_profile_args_and_rejects_legacy_execution(fake_native, capsys):
    source, sessions, _ = fake_native
    for options in ([], ["--resource-profile", "fake"], ["--resource-profile", "fake", "--execute-action"]):
        assert main(["toolkit", "replay", str(source), "--native-action", "0", "--tool", "compute", *options, "--json"]) == 2
        capsys.readouterr()
    assert not sessions
    assert main(["toolkit", "replay", str(source), "--native-action", "0", "--tool", "compute",
                 "--args", '{"expressions":"total = 1 + 2"}', "--resource-profile", "fake", "--json"]) == 0
    assert decode(capsys.readouterr().out.encode())["status"] == "ok"


def test_native_perception_uses_fresh_pixels_and_worker_is_closed_before_script(fake_native, monkeypatch, tmp_path):
    from ora.agent.toolkit.interface.services import sam3_adapters
    from ora.orchestrator.toolkit import snapshot_replay
    from tests.support.agent.toolkit_lib import BINDINGS, Sam
    source, sessions, _ = fake_native
    sam = Sam()
    original = sam.ground
    seen = []
    def ground(**kwargs):
        seen.append(kwargs["rgb_u8"][0, 0].tolist())
        return original(**kwargs)
    sam.ground = ground
    @contextmanager
    def services(report):
        yield sam3_adapters(sam), BINDINGS
    monkeypatch.setattr(snapshot_replay, "service_session", services)
    build = snapshot_replay.build
    def detached(*args, **kwargs):
        assert sessions[-1].calls[-1] == "close"
        return build(*args, **kwargs)
    monkeypatch.setattr(snapshot_replay, "build", detached)
    example = Path(__file__).resolve().parents[4] / "ora/agent/toolkit/examples/ground_geometry.py"
    result = experiment(source=source, native_action=1, resource_profile="fake", script=example,
                        arguments={"camera": "head", "text": "cup", "max_candidates": 1})
    assert result["status"] == "ok", result
    assert seen == [[17, 31, 53]]
    assert result["comparison_count"] == 1
    assert result["result"]["comparison"] is None
    output = Path(result["output"])
    row = decode((output / result["comparisons"]).read_bytes())
    comparison = decode((output / row["reference"]["path"]).read_bytes())
    assert comparison["status"] == "unavailable"
    assert comparison["reason"] == "no_matching_snapshot_service_request"


def test_native_unsupported_tool_rejected_before_worker_start(fake_native):
    source, sessions, _ = fake_native
    result = experiment(source=source, native_action=1, resource_profile="fake",
                        tool="vision_ground", arguments={"text": "cup"})
    assert result["error"]["type"] == "NotRecorded"
    assert result["missing_requests"][0]["method"] == "tool_context"
    assert not sessions


def test_native_sigterm_unwinds_worker_and_restores_signal_handler(fake_native, tmp_path):
    import signal
    source, sessions, settings = fake_native
    settings["failure"] = "sigterm"
    previous = signal.getsignal(signal.SIGTERM)
    script = tmp_path / "never.py"
    script.write_text('def run(api, args):\n    raise AssertionError("must not run")\n')
    result = experiment(source=source, native_action=1, resource_profile="fake", script=script)
    assert result["error"]["type"] == "SystemExit"
    assert sessions[-1].calls[-1] == "close"
    assert signal.getsignal(signal.SIGTERM) is previous


@pytest.mark.parametrize("cancel", [False, True])
def test_native_external_output_through_host_adapter_and_real_worker_path_guard(
    fake_native, tmp_path, monkeypatch, cancel
):
    """CLI -> host lifecycle -> real observe guard, with only physics/transport fake."""
    from types import SimpleNamespace
    from ora.environment.worker import _dispatch
    from ora.orchestrator.backends.robodojo import toolkit_live as live
    from ora.orchestrator.paths import WORKSPACE_ROOT
    source, _, _ = fake_native
    output = tmp_path / "arbitrary-external-output"
    assert not output.resolve().is_relative_to(WORKSPACE_ROOT.resolve())
    order = []
    monkeypatch.setitem(_LOADED, "robodojo", replace(backend("robodojo"), toolkit_live=live.open_live))
    monkeypatch.setattr(live, "native_identity", lambda source: dict(commit="a" * 40, worker_kernel_matches_run=True))
    monkeypatch.setattr(live, "resolve_execution_resource", lambda *a, **kw: {})
    monkeypatch.setattr(live, "replay_runtime", lambda runtime, binding: runtime)
    monkeypatch.setattr(live.host_storage, "stage_launch", lambda *a: {})
    monkeypatch.setattr(live.host_storage, "scratch_environment", lambda *a: {})
    monkeypatch.setattr(live.host_storage, "release_run_scratch", lambda *a: order.append("scratch"))
    monkeypatch.setattr(live, "claim_resource_lease", lambda *a: dict(claim_token="test"))
    monkeypatch.setattr(live, "assign_resource_lease", lambda *a: None)
    monkeypatch.setattr(live, "release_resource_lease", lambda *a, **kw: order.append("lease"))
    monkeypatch.setattr(live, "verify_prefix", lambda *a: dict(passed=True, differences=[]))

    def worker_config(runtime, plan, root, native):
        path = root / "worker-config.json"
        path.write_text("{}")
        return SimpleNamespace(python_executable="unused", project_root=tmp_path,
                               requested_socket_root=tmp_path, startup_timeout_s=1, worker_environment={}), path
    monkeypatch.setattr(live, "_worker_config", worker_config)

    class Transport(FakeSession):
        def __init__(self, **kw):
            super().__init__(source, output, kw["worker_config"]["toolkit_boundary"])
            self.output_root = kw["output_root"]
            self.config = dict(kw["worker_config"], shared_root=str(self.output_root.parent.resolve()))
            assert self.config["output_root"] == str(self.output_root.resolve())
            self.process = SimpleNamespace(pid=1)

        def start(self):
            order.append("start")

        def diagnostic(self, method, **params):
            if method == "restore":
                if cancel:
                    import signal
                    signal.raise_signal(signal.SIGTERM)
                return {}
            if method == "prepare":
                fake = SimpleNamespace(observe=lambda destination: (
                    FakeSession.diagnostic(self, "prepare", destination=str(destination))["observation"], []))
                value, closing = _dispatch(fake, self.config, "observe", params)
                assert not closing
                order.append("render")
                return value
            return super().diagnostic(method, **params)

        def close(self):
            order.append("close")
    monkeypatch.setattr(live, "Session", Transport)
    result = experiment(source=source, native_action=0, resource_profile="fake",
                        output=output, tool="compute", arguments={"expressions": "total = 1 + 2"})
    assert order[-3:] == ["close", "lease", "scratch"]
    if cancel:
        assert result["error"]["type"] == "SystemExit"
        assert "render" not in order
    else:
        assert result["status"] == "ok", result
        assert "render" in order
        assert (output / "native-state.json").is_file()
