"""``ora setup --env``: recipes, idempotence and ``.env`` ownership, without downloads or a GPU."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from ora.app import env_file as dotenv, gpu_recipes as recipes, gpu_setup as setup
from ora.environment.planners.timing.requirements import PINS
from ora.app.main import build_parser


def _setup(argv):
    """``ora setup ARGV`` through the app's parser, straight into the GPU installer."""
    return setup.command_setup(build_parser().parse_args(["setup", *argv]))


def test_pins_equal_their_canonical_modules():
    from ora.agent.model_services import resident
    from ora.environment.backends.robotwin import vendored_install
    from ora.environment.benchmarks.libero_pro import LIBERO_PRO_CODE_REVISION, LIBERO_PRO_DATA_REVISION
    from ora.orchestrator.backends.robotwin import evaluation_seeds as robotwin_evaluation

    assert recipes.LIBERO_CODE[1] == LIBERO_PRO_CODE_REVISION
    assert recipes.LIBERO_DATA[1] == LIBERO_PRO_DATA_REVISION
    assert recipes.ROBOTWIN[1] == vendored_install.UPSTREAM_REVISION
    assert recipes.ROBOTWIN_INSTALL_REVISION == vendored_install.INSTALL_REVISION == robotwin_evaluation.REQUIRED_REVISION
    assert recipes.CAPX[1] == resident.CAPX_SOURCE_COMMIT
    assert (recipes.SAM3_SOURCE_COMMIT, recipes.SAM3_SHA256) == (resident.SAM3_SOURCE_COMMIT,
                                                                 resident.SAM3_CHECKPOINT_SHA256)
    assert (recipes.CONTACT_SOURCE_COMMIT, recipes.CONTACT_CHECKPOINT_SHA256, recipes.CONTACT_CONFIG_SHA256) == (
        resident.CONTACT_SOURCE_COMMIT, resident.CONTACT_CHECKPOINT_SHA256, resident.CONTACT_CONFIG_SHA256)


def test_environments_pull_in_only_what_they_run_on():
    assert recipes.components_for(["perception"]) == ["agent-core", "perception"]
    # The LIBERO planner runs under the RoboDojo runtime's interpreter.
    assert recipes.components_for(["libero"]) == ["agent-core", "perception", "libero", "robodojo", "motion-timing"]
    assert recipes.components_for(["robotwin", "libero"]) == [
        "agent-core", "perception", "robotwin-python", "robotwin-assets", "libero", "robodojo", "motion-timing"]
    # RoboDojo's default policy runs code mode, so its sandbox interpreter comes with it (LIBERO's planner does not).
    assert recipes.components_for(["robodojo"]) == ["agent-core", "perception", "robodojo", "motion-timing",
                                                    "code-mode"]


def _recording(tmp_path, **kwargs):
    commands = []
    created = [0]

    def runner(argv, *, cwd=None, env=None, timeout=None):
        commands.append({"argv": argv, "cwd": cwd, "env": dict(env or {})})
        if argv[:2] == ["uv", "venv"]:
            target = Path(argv[-1])
            (target / "bin").mkdir(parents=True, exist_ok=True)
            (target / "bin/python").write_text("")
            version = argv[argv.index("--python") + 1]
            config = target / "pyvenv.cfg"
            config.write_text(f"version_info = {version}.4\n")
            # A real ``uv venv`` takes far longer than one timestamp tick; without
            # this, ext4 can reuse the inode within one tick and a recreated
            # environment looks unchanged to ``_venv_identity``.
            created[0] += 1
            stamp = config.stat().st_mtime_ns + created[0] * 1_000_000_000
            os.utime(config, ns=(stamp, stamp))

    context = setup.Context(prefix=tmp_path / "prefix", runner=runner, environ={"PATH": "/usr/bin"}, **kwargs)
    return context, commands


def test_every_simulator_environment_installs_cu128_torch(tmp_path):
    context, commands = _recording(tmp_path)
    steps = [step for component in ("perception", "robotwin-python", "libero")
             for step in recipes.BUILDERS[component](context) if step.name == "torch"]
    setup.execute(context, steps, stream=io.StringIO())
    installs = [command["argv"] for command in commands]
    assert all(argv[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu128"] for argv in installs)
    assert [argv[5:7] for argv in installs] == [["torch==2.8.0", "torchvision==0.23.0"],
                                                ["torch==2.7.0", "torchvision==0.22.0"],
                                                ["torch==2.7.0", "torchvision==0.22.0"]]


def test_robotwin_requirements_drop_only_torch_and_torchvision(tmp_path):
    context, commands = _recording(tmp_path)
    upstream = context.path("sources/RoboTwin/script/requirements.txt")
    upstream.parent.mkdir(parents=True)
    upstream.write_text("torch==2.4.1\ntorchvision\nsapien==3.0.0b1\nmplib==0.2.1\ntorchaudio==2.4.1\n")
    step = next(step for step in recipes.robotwin_python(context) if step.name == "requirements")
    setup.execute(context, [step], stream=io.StringIO())
    written = context.path("manifests/ora-setup/robotwin-requirements.txt").read_text().split()
    assert written == ["sapien==3.0.0b1", "mplib==0.2.1", "torchaudio==2.4.1"]
    assert commands[-1]["argv"][-2:] == ["numpy==1.26.4", "setuptools==69.5.1"]


def test_curobo_is_built_for_this_gpu_and_rebuilt_when_the_arch_changes(tmp_path):
    cuda = tmp_path / "cuda-12.8"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/nvcc").write_text("")
    context, commands = _recording(tmp_path, cuda_home=cuda, cuda_arch="12.0")
    python = context.path("envs/robotwin/bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("")
    step = next(step for step in recipes.robotwin_python(context) if step.name == "curobo-build")
    setup.execute(context, [step], stream=io.StringIO())
    build = commands[-1]
    assert build["argv"][-3:] == ["-e", str(context.path("sources/curobo")), "--no-build-isolation"]
    assert build["env"]["TORCH_CUDA_ARCH_LIST"] == "12.0+PTX"
    assert build["env"]["CUDA_HOME"] == str(cuda)
    assert build["env"]["PATH"].startswith(f"{cuda / 'bin'}:")
    assert setup.step_done(context, step)
    hopper = setup.Context(prefix=context.prefix, runner=context.runner, cuda_home=cuda, cuda_arch="9.0")
    assert not setup.step_done(hopper, step)


def test_curobo_build_finds_a_conda_toolkit_s_headers(tmp_path):
    cuda = tmp_path / "cuda-conda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/nvcc").write_text("")
    (cuda / "targets/x86_64-linux/include").mkdir(parents=True)
    (cuda / "targets/x86_64-linux/include/cuda_runtime_api.h").write_text("")
    context, commands = _recording(tmp_path, cuda_home=cuda, cuda_arch="9.0")
    python = context.path("envs/robotwin/bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("")
    step = next(step for step in recipes.robotwin_python(context) if step.name == "curobo-build")
    setup.execute(context, [step], stream=io.StringIO())
    assert commands[-1]["env"]["CPATH"].split(os.pathsep)[0] == str(cuda / "targets/x86_64-linux/include")
    assert commands[-1]["env"]["TORCH_CUDA_ARCH_LIST"] == "9.0+PTX"


def test_missing_isaac_system_libraries_are_unpacked_without_sudo(tmp_path, monkeypatch):
    context, commands = _recording(tmp_path)
    monkeypatch.setattr(recipes, "_loadable", lambda name: name == "libvulkan.so.1")
    monkeypatch.setattr(recipes.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    step = recipes.system_libraries(context)
    assert not setup.step_done(context, step)
    (context.path("downloads/syslibs")).mkdir(parents=True)
    (context.path("downloads/syslibs") / "libglu1-mesa_9.0.2_amd64.deb").write_text("")
    setup.execute(context, [step], stream=io.StringIO())
    assert commands[0]["argv"] == ["apt-get", "download", "libglu1-mesa"]
    assert commands[1]["argv"][:2] == ["dpkg", "-x"] and commands[1]["argv"][-1] == str(context.path("syslibs"))
    (context.path(recipes.SYSTEM_LIBRARY_DIR)).mkdir(parents=True)
    (context.path(recipes.SYSTEM_LIBRARY_DIR) / "libGLU.so.1").write_text("")
    assert setup.step_done(context, step)
    environ = {}
    recipes.add_library_path(environ, context.prefix)
    assert environ["LD_LIBRARY_PATH"] == str(context.path(recipes.SYSTEM_LIBRARY_DIR))
    monkeypatch.setattr(recipes.shutil, "which", lambda tool: None)
    (context.path(recipes.SYSTEM_LIBRARY_DIR) / "libGLU.so.1").unlink()
    with pytest.raises(setup.GpuSetupError, match="libglu1-mesa"):
        setup.execute(context, [step], stream=io.StringIO())


def test_cuda_arch_comes_from_nvidia_smi_and_is_validated(tmp_path, monkeypatch):
    context = setup.Context(prefix=tmp_path)
    monkeypatch.setattr(setup.Context, "output", lambda self, argv, **kwargs: "12.0\n12.0")
    assert recipes.cuda_arch(context) == "12.0+PTX"
    monkeypatch.setattr(setup.Context, "output", lambda self, argv, **kwargs: "")
    with pytest.raises(setup.GpuSetupError, match="--cuda-arch"):
        recipes.cuda_arch(context)


def _repository(path: Path) -> str:
    path.mkdir()
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@example.com"],
                 ["git", "config", "user.name", "t"]):
        subprocess.run(argv, cwd=path, check=True)
    (path / "file.txt").write_text("pinned\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=path, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_clone_step_pins_a_revision_skips_when_done_and_never_resets(tmp_path):
    revision = _repository(tmp_path / "upstream")
    context = setup.Context(prefix=tmp_path / "prefix")
    step = recipes.clone_step("demo", "repo", "sources/repo", str(tmp_path / "upstream"), revision)
    assert not setup.step_done(context, step)
    setup.execute(context, [step], stream=io.StringIO())
    checkout = context.path("sources/repo")
    assert subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip() == revision
    output = io.StringIO()
    setup.execute(context, [step], stream=output)
    assert "already done" in output.getvalue()
    (checkout / "file.txt").write_text("edited\n")
    assert not setup.step_done(context, step)
    with pytest.raises(setup.GpuSetupError, match="not a clean checkout"):
        setup.execute(context, [step], stream=io.StringIO())
    assert (checkout / "file.txt").read_text() == "edited\n"


def test_sam3_download_is_verified_before_it_is_kept(tmp_path, monkeypatch):
    context, commands = _recording(tmp_path)

    def fake_curl(argv, **kwargs):
        commands.append(argv)
        Path(argv[argv.index("-o") + 1]).write_bytes(b"not the checkpoint")

    context.runner = fake_curl
    step = next(step for step in recipes.perception(context) if step.name == "sam3")
    with pytest.raises(setup.GpuSetupError, match="differs from the pinned"):
        setup.execute(context, [step], stream=io.StringIO())
    assert commands[0][-1] == recipes.SAM3_URL
    assert not context.path("models/sam3.pt").exists()
    assert not context.path("models/sam3.pt.partial").exists()
    monkeypatch.setattr(recipes, "SAM3_SHA256", recipes._sha256(Path(__file__)))
    monkeypatch.setattr(recipes, "_sha256", lambda path: recipes.SAM3_SHA256)
    setup.execute(context, [step], stream=io.StringIO())
    assert context.path("models/sam3.pt").is_file() and setup.step_done(context, step)


def test_robodojo_stages_always_name_the_prefix_root(tmp_path, monkeypatch):
    monkeypatch.setattr(recipes, "_loadable", lambda name: True)  # the system has libGLU and Vulkan
    monkeypatch.setattr(recipes.shutil, "which", lambda tool, path=None: f"/usr/bin/{tool}")  # and git-lfs
    context, commands = _recording(tmp_path)
    setup.execute(context, recipes.robodojo(context), stream=io.StringIO())
    stages = [command["argv"][2] for command in commands]
    assert sorted(stages[:3]) == ["assets", "sdk", "sources"]
    assert stages[3:] == ["resolve", "packages", "bind-assets", "graphics-compat", "doctor", "manifest"]
    root = str(context.path("robodojo-runtime"))
    assert all(command["argv"][-2:] == ["--root", root] for command in commands)
    # The real graphics-compat stage writes this receipt; with it the rerun only re-checks.
    receipt = context.path("robodojo-runtime/graphics-compat/installation.json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    commands.clear()
    setup.execute(context, recipes.robodojo(context), stream=io.StringIO())
    assert [command["argv"][2] for command in commands] == ["doctor", "manifest"]


def test_env_keys_cover_each_environment_and_hold_no_credential_or_host_marker(tmp_path):
    context = setup.Context(prefix=tmp_path)
    keys = recipes.env_keys(context, recipes.COMPONENTS)
    for name in ("ORA_SAM_CHECKPOINT", "ORA_GRASP_CONFIG", "ROBOTWIN_PYTHON", "CUROBO_PYTHON", "LIBERO_PRO_ROOT",
                 "ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT", "ROBODOJO_PYTHON", "ROBODOJO_RUNTIME_ROOT"):
        assert keys[name].startswith(str(tmp_path)), name
    assert keys["CUROBO_PYTHON"] == keys["ROBOTWIN_PYTHON"]
    assert keys["ROBODOJO_ROOT"] == f"{keys['ROBODOJO_RUNTIME_ROOT']}/source/RoboDojo"
    # LIBERO's cuRobo V2 is the RoboDojo runtime's interpreter and the curobo source it imports.
    assert keys["CUROBO_V2_PYTHON"] == keys["ROBODOJO_PYTHON"]
    assert keys["CUROBO_V2_ROOT"] == f"{keys['ROBODOJO_ROOT']}/third_party/curobo"
    # The timing worker has its own small interpreter, never a simulator's or planner's.
    assert keys["ORA_TIMING_PYTHON"] == str(tmp_path / "envs/motion-timing/bin/python")
    # So has the code-mode sandbox, where the sandbox looks for it when ORA_CODE_MODE_PYTHON is unset.
    assert keys["ORA_CODE_MODE_PYTHON"] == str(tmp_path / "envs/code-mode/bin/python")
    assert not [key for key in keys if "KEY" in key or "TOKEN" in key]
    assert "ORA_EXECUTION_HOST" not in keys


def test_the_timing_interpreter_is_pinned_and_installed_from_its_requirements(tmp_path):
    context = setup.Context(prefix=tmp_path)
    steps = recipes.motion_timing(context)
    assert [step.name for step in steps] == ["venv-motion-timing", "requirements"]
    assert recipes.TIMING_PINS == PINS
    pins = dict(pin.split("==") for pin in PINS)
    assert (pins["toppra"], pins["ruckig"], pins["numpy"]) == ("0.6.10", "0.19.4", "1.26.4")


def test_the_code_mode_interpreter_is_built_from_the_sandbox_lock_where_the_sandbox_finds_it(tmp_path, monkeypatch):
    from ora.agent.code_mode import sandbox

    assert setup.PROJECT_ROOT / recipes.CODE_MODE_LOCK == sandbox.LOCK
    bwrap = tmp_path / "bwrap"  # the machine has bubblewrap, so setup installs none
    bwrap.write_text("#!/bin/sh\nexit 0\n")
    bwrap.chmod(0o755)
    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(bwrap))
    context, commands = _recording(tmp_path)
    steps = recipes.code_mode(context)
    assert [step.name for step in steps] == ["bubblewrap", "venv-code-mode", "requirements"]
    setup.execute(context, steps, stream=io.StringIO())
    python = str(context.path(sandbox.DEFAULT_PYTHON))
    # sandbox.LOCK_BUILD, as setup runs it: no seed packages, hashes required, wheels only, bytecode compiled.
    assert [command["argv"] for command in commands] == [
        ["uv", "venv", "--python", "3.11", str(context.path("envs/code-mode"))],
        ["uv", "pip", "install", "--python", python, "--require-hashes", "--no-deps", "--only-binary", ":all:",
         "--compile-bytecode", "-r", str(sandbox.LOCK)]]
    assert recipes.env_keys(context, ["code-mode"]) == {"ORA_CODE_MODE_PYTHON": python}
    # The sandbox's fallback is setup's own prefix rule.
    for environ in ({"ORA_GPU_PREFIX": str(context.prefix)}, {"XDG_DATA_HOME": str(tmp_path / "data")}, {}):
        assert sandbox.default_python(environ) == setup.default_prefix(environ, {}) / sandbox.DEFAULT_PYTHON
    assert sandbox.default_python({"ORA_GPU_PREFIX": str(context.prefix)}) == Path(python)
    # A changed lock rebuilds.
    marker = json.loads(context.path("manifests/ora-setup/code-mode.requirements.json").read_text())
    assert marker["recipe"]["lock_sha256"] == hashlib.sha256(sandbox.LOCK.read_bytes()).hexdigest()


def test_setup_warns_when_code_mode_has_no_bwrap(monkeypatch, tmp_path):
    from ora.agent.code_mode import sandbox

    assert recipes.sandbox_warning(["agent-core", "perception"]) is None
    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(tmp_path / "missing-bwrap"))
    warning = recipes.sandbox_warning(["robodojo", "code-mode"])
    assert sandbox.BWRAP_REMEDY in warning and "ora doctor --env robodojo" in warning
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("#!/bin/sh\nexit 0\n")
    bwrap.chmod(0o755)
    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(bwrap))
    assert recipes.sandbox_warning(["robodojo", "code-mode"]) is None


def test_env_file_update_owns_keys_and_removes_blanking_credential_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# operator notes\nDEEPSEEK_LOW_API_KEY=\nDEEPSEEK_API_KEY=keep-me\nORA_SAM_PORT=8124\n"
                    "ROBOTWIN_ROOT=/old\n")
    os.chmod(path, 0o640)
    changed = dotenv.update(path, {"ROBOTWIN_ROOT": "/new root", "ORA_SAM_GPU": "0"},
                                  defaults={"DEEPSEEK_BASE_URL": "https://api.deepseek.com", "ORA_SAM_PORT": "8114"})
    text = path.read_text()
    assert "DEEPSEEK_LOW_API_KEY=" not in text and "DEEPSEEK_API_KEY=keep-me" in text
    assert "ORA_SAM_PORT=8124" in text and "# operator notes" in text
    assert dotenv.read(path)["ROBOTWIN_ROOT"] == "/new root"
    assert set(changed) == {"ROBOTWIN_ROOT", "ORA_SAM_GPU", "DEEPSEEK_BASE_URL"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert dotenv.update(path, {"ROBOTWIN_ROOT": "/new root", "ORA_SAM_GPU": "0"}) == {}
    with pytest.raises(ValueError, match="never writes credentials"):
        dotenv.update(path, {"DEEPSEEK_LOW_API_KEY": "secret"})
    fresh = tmp_path / "fresh.env"
    dotenv.update(fresh, {"A": "1"})
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o600


def test_env_file_update_moves_the_pre_1_0_service_names_to_the_ora_names(tmp_path):
    from ora.agent.model_services.service_env import SERVICE_FIELDS, legacy_renames, legacy_service_key

    renames = legacy_renames()
    assert len(renames) == sum(len(fields) for fields in SERVICE_FIELDS.values()) == 11
    assert all(new.startswith(("ORA_SAM_", "ORA_GRASP_")) for new in renames.values())
    root, port = legacy_service_key("SAM", "ROOT"), legacy_service_key("SAM", "PORT")
    grasp_port = legacy_service_key("GRASP", "PORT")
    path = tmp_path / ".env"
    path.write_text(f"# {root}=/kept/in/a/comment\n{root}=/old/sam\n{port}=8124\nROBOTWIN_ROOT=/r\n"
                    f"export {grasp_port}=8125\nORA_GRASP_PORT=8115\n")
    changed = dotenv.update(path, {"ORA_SAM_ROOT": "/new/sam"}, replaces=renames)
    assert path.read_text() == (f"# {root}=/kept/in/a/comment\nORA_SAM_ROOT=/new/sam\nORA_SAM_PORT=8124\n"
                                "ROBOTWIN_ROOT=/r\nORA_GRASP_PORT=8115\n")
    assert changed == {"ORA_SAM_ROOT": "/new/sam", "ORA_SAM_PORT": "8124"}
    assert dotenv.update(path, {"ORA_SAM_ROOT": "/new/sam"}, replaces=renames) == {}


def test_setup_env_drops_the_pre_1_0_names_of_the_keys_it_writes(tmp_path, monkeypatch, capsys):
    from ora.agent.model_services.service_env import legacy_service_key

    monkeypatch.setattr(recipes, "check_toolchain", lambda context, components: None)
    monkeypatch.setattr(recipes, "steps_for", lambda context, components: [])
    env_file = tmp_path / ".env"
    legacy = {legacy_service_key("SAM", "CHECKPOINT"): "/old/sam3.pt", legacy_service_key("GRASP", "GPU"): "3",
              legacy_service_key("SAM", "PORT"): "8124"}
    env_file.write_text("".join(f"{key}={value}\n" for key, value in legacy.items()))
    assert _setup(["--env", "perception", "--prefix", str(tmp_path / "gpu"), "--env-file", str(env_file)]) == 0
    values = dotenv.read(env_file)
    assert not set(values) & set(legacy)
    assert values["ORA_SAM_CHECKPOINT"] == str(tmp_path / "gpu/models/sam3.pt")
    assert (values["ORA_GRASP_GPU"], values["ORA_SAM_PORT"]) == ("0", "8124")
    assert f"replaced the earlier names {', '.join(sorted(legacy))} with" in capsys.readouterr().out


def test_env_file_load_never_overrides_or_blanks_the_environment(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_LOW_API_KEY=\nROBOTWIN_ROOT=/from-file\nMUJOCO_GL=egl\n")
    environ = {"DEEPSEEK_LOW_API_KEY": "from-shell", "MUJOCO_GL": "osmesa"}
    assert dotenv.load_into(environ, path) == ["ROBOTWIN_ROOT"]
    assert environ == {"DEEPSEEK_LOW_API_KEY": "from-shell", "MUJOCO_GL": "osmesa", "ROBOTWIN_ROOT": "/from-file"}


def test_cli_dry_run_changes_nothing_and_refuses_a_prefix_in_the_checkout(tmp_path, capsys):
    env_file = tmp_path / ".env"
    code = _setup(["--env", "perception", "--prefix", str(tmp_path / "gpu"), "--env-file", str(env_file),
                       "--dry-run", "--json"])
    assert code == 0
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["components"] == ["agent-core", "perception"]
    assert {item["status"] for item in report["steps"]} <= {"pending", "done"}
    assert report["env_keys_written"] == []
    assert not env_file.exists() and not (tmp_path / "gpu").exists()
    code = _setup(["--env", "perception", "--prefix", str(setup.PROJECT_ROOT / "gpu-inside"),
                       "--env-file", str(env_file), "--dry-run"])
    assert code == 1
    assert "inside the Harness checkout" in capsys.readouterr().err


def test_ora_setup_with_env_runs_the_gpu_installer_and_refuses_the_local_options(monkeypatch):
    from ora.app import local_setup
    from ora.app.main import main

    seen = []
    monkeypatch.setattr(setup, "command_setup", lambda args: seen.append(("gpu", args.environments, args.dry_run)) or 0)
    monkeypatch.setattr(local_setup, "command_setup", lambda args: seen.append(("local", args.root)) or 0)
    assert main(["setup", "--env", "robotwin", "--env", "perception", "--dry-run"]) == 0
    assert main(["setup", "--root", "/srv/ora"]) == 0
    assert seen == [("gpu", ["robotwin", "perception"], True), ("local", Path("/srv/ora"))]
    assert main(["setup", "--env", "robotwin", "--graphics", "egl"]) == 2  # a local option
    assert main(["setup", "--prefix", "/srv/gpu"]) == 2  # a GPU option without --env
    assert len(seen) == 2


def test_a_recreated_environment_invalidates_its_package_markers(tmp_path):
    context, commands = _recording(tmp_path)
    steps = [step for step in recipes.libero(context) if step.name in {"venv-libero", "torch"}]
    setup.execute(context, steps, stream=io.StringIO())
    assert setup.step_done(context, steps[1])
    import shutil
    shutil.rmtree(context.path("envs/libero"))
    commands.clear()
    setup.execute(context, steps, stream=io.StringIO())
    assert [command["argv"][:2] for command in commands] == [["uv", "venv"], ["uv", "pip"]]


def test_graphics_compat_is_done_by_its_receipt_and_never_replaced(tmp_path):
    context, commands = _recording(tmp_path)
    step = next(step for step in recipes.robodojo(context) if step.name == "graphics-compat")
    assert not setup.step_done(context, step)
    receipt = context.path("robodojo-runtime/graphics-compat/installation.json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    setup.execute(context, [step], stream=io.StringIO())
    assert commands == []


def test_a_failed_git_status_is_not_a_clean_checkout(tmp_path, monkeypatch):
    revision = _repository(tmp_path / "upstream")
    context = setup.Context(prefix=tmp_path / "prefix")
    step = recipes.clone_step("demo", "repo", "sources/repo", str(tmp_path / "upstream"), revision)
    setup.execute(context, [step], stream=io.StringIO())
    real = setup.Context.probe
    monkeypatch.setattr(setup.Context, "probe", lambda self, argv, **kwargs: None if "status" in argv
                        else real(self, argv, **kwargs))
    assert not setup.step_done(context, step)


def test_robotwin_assets_unpack_with_unzip_after_a_path_check(tmp_path):
    import zipfile

    context, commands = _recording(tmp_path)
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for name in recipes.ROBOTWIN_ASSET_ZIPS:
        with zipfile.ZipFile(downloads / name, "w") as archive:
            archive.writestr(f"{name[:-4]}/file.txt", "x")
    manifest = context.path("manifests/robotwin-assets.json")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"directory": str(downloads), "sha256": dict.fromkeys(recipes.ROBOTWIN_ASSET_ZIPS, "0")}))
    step = next(step for step in recipes.robotwin_assets(context) if step.name == "unpack")
    setup.execute(context, [step], stream=io.StringIO())
    assert [command["argv"][:3] for command in commands] == [["unzip", "-o", "-q"]] * 3
    with zipfile.ZipFile(downloads / "objects.zip", "w") as archive:
        archive.writestr("../escape.txt", "x")
    manifest.write_text(json.dumps({"directory": str(downloads), "sha256": {"changed": "1"}}))
    with pytest.raises(setup.GpuSetupError, match="escapes"):
        setup.execute(context, [step], stream=io.StringIO())


def test_missing_git_lfs_is_installed_under_the_prefix_and_checked(tmp_path, monkeypatch):
    context, commands = _recording(tmp_path)
    context.environ = {"PATH": str(tmp_path / "empty-bin")}
    monkeypatch.setattr(recipes, "_x86_64_linux", lambda: True)
    step = recipes.git_lfs(context)
    assert not setup.step_done(context, step)
    version, digest = recipes.GIT_LFS
    archive = context.path(f"downloads/git-lfs-linux-amd64-v{version}.tar.gz")

    def fetch_and_unpack(argv, *, cwd=None, env=None, timeout=None):
        commands.append({"argv": argv})
        if argv[0] == "curl":
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(b"not the release")
        if argv[0] == "tar":
            (Path(argv[argv.index("-C") + 1]) / "git-lfs").write_text("#!/bin/sh\n")

    context.runner = fetch_and_unpack
    with pytest.raises(setup.GpuSetupError, match="SHA-256"):
        setup.execute(context, [step], stream=io.StringIO())
    assert not archive.exists()
    monkeypatch.setattr(recipes, "_sha256", lambda path: digest)
    setup.execute(context, [step], stream=io.StringIO())
    assert (context.path(recipes.TOOLS_BIN) / "git-lfs").is_file()
    assert context.environ["PATH"].split(os.pathsep)[0] == str(context.path(recipes.TOOLS_BIN))
    (context.path(recipes.TOOLS_BIN) / "git-lfs").chmod(0o755)
    assert setup.step_done(context, step)


def test_missing_bubblewrap_is_unpacked_without_sudo_and_named_in_env(tmp_path, monkeypatch):
    context, commands = _recording(tmp_path)
    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(tmp_path / "missing-bwrap"))
    monkeypatch.setattr(recipes.shutil, "which", lambda tool, path=None: f"/usr/bin/{tool}")
    step = recipes.bubblewrap(context)
    assert not setup.step_done(context, step)
    assert recipes.sandbox_warning(["code-mode"], context.prefix) is not None
    download = context.path("downloads/bubblewrap")
    download.mkdir(parents=True)
    (download / "bubblewrap_0.9.0_amd64.deb").write_text("")

    def unpack(argv, *, cwd=None, env=None, timeout=None):
        commands.append({"argv": argv})
        if argv[:2] == ["dpkg", "-x"]:
            binary = context.path(recipes.BWRAP_IN_PREFIX)
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)

    context.runner = unpack
    setup.execute(context, [step], stream=io.StringIO())
    assert commands[0]["argv"] == ["apt-get", "download", "bubblewrap"]
    assert setup.step_done(context, step)
    assert recipes.sandbox_warning(["code-mode"], context.prefix) is None
    assert recipes.env_keys(context, ["code-mode"])["ORA_CODE_MODE_BWRAP"] == str(context.path(recipes.BWRAP_IN_PREFIX))
    monkeypatch.setattr(recipes.shutil, "which", lambda tool, path=None: None)
    context.path(recipes.BWRAP_IN_PREFIX).unlink()
    with pytest.raises(setup.GpuSetupError, match="bubblewrap"):
        setup.execute(context, [step], stream=io.StringIO())


def test_doctor_reads_the_checkout_env_for_a_simulator_as_ora_run_does(tmp_path):
    from ora.app import doctor

    env_path = tmp_path / ".env"
    prefix = tmp_path / "prefix"
    (prefix / recipes.SYSTEM_LIBRARY_DIR).mkdir(parents=True)
    env_path.write_text(f"ROBODOJO_PYTHON=/from-file\nORA_GPU_PREFIX={prefix}\nORA_CODE_MODE_BWRAP=/opt/bwrap\n")
    environ = {"ORA_CODE_MODE_BWRAP": "/from-shell"}
    assert doctor.load_checkout_env("synthetic", environ, env_path) == []
    loaded = doctor.load_checkout_env("robodojo", environ, env_path)
    assert sorted(loaded) == ["ORA_GPU_PREFIX", "ROBODOJO_PYTHON"]
    assert environ["ORA_CODE_MODE_BWRAP"] == "/from-shell"  # the shell wins
    assert environ["LD_LIBRARY_PATH"] == str(prefix / recipes.SYSTEM_LIBRARY_DIR)
