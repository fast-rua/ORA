"""`ora --version`, and the one version every release manifest declares."""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from pathlib import Path

import pytest
import yaml

import ora.version
from ora.agent.tui import follow
from ora.app import backend as ora_backend
from ora.app import main as app_main

ROOT = Path(__file__).resolve().parents[3]


def _pyproject_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def _installed(version: str, asked: list[str] | None = None):
    def lookup(name: str) -> str:
        if asked is not None:
            asked.append(name)
        return version
    return lookup


def _not_installed(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)


def test_a_checkout_reports_its_pyproject_version_over_stale_install_metadata(monkeypatch, capsys) -> None:
    # A shared environment's editable install, made before the version changed, still says 0.0.1.
    assert ora.version.CHECKOUT_PYPROJECT == ROOT / "pyproject.toml"
    monkeypatch.setattr(ora.version.importlib.metadata, "version", _installed("0.0.1"))
    assert ora.version.ora_version() == _pyproject_version() != "0.0.1"
    with pytest.raises(SystemExit) as exited:
        app_main.main(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out == f"ora {_pyproject_version()}\n"
    assert "--version" in app_main.build_parser()._option_string_actions

    monkeypatch.setattr(ora.version.importlib.metadata, "version", _not_installed)
    assert ora.version.ora_version() == _pyproject_version()


def test_without_the_checkout_project_file_the_installed_metadata_names_the_version(monkeypatch, tmp_path) -> None:
    asked: list[str] = []
    monkeypatch.setattr(ora.version.importlib.metadata, "version", _installed("9.8.7", asked))
    assert ora.version.ora_version(tmp_path / "missing.toml") == "9.8.7"   # a wheel install
    other = tmp_path / "pyproject.toml"
    other.write_text('[project]\nname = "someone-else"\nversion = "2.3.4"\n', encoding="utf-8")
    assert ora.version.ora_version(other) == "9.8.7"                       # not ORA's project file
    other.write_text('[project]\nname = "ora-agent"\n', encoding="utf-8")
    assert ora.version.ora_version(other) == "9.8.7"                       # ORA's, without a version
    other.write_text("[project\n", encoding="utf-8")
    assert ora.version.ora_version(other) == "9.8.7"
    assert asked == ["ora-agent"] * 4
    other.write_text('[project]\nname = "ora-agent"\nversion = "2.3.4"\n', encoding="utf-8")
    assert ora.version.ora_version(other) == "2.3.4"

    monkeypatch.setattr(ora.version.importlib.metadata, "version", _not_installed)
    assert ora.version.ora_version(tmp_path / "missing.toml") == "unknown"
    assert ora.version.version_label(tmp_path / "missing.toml") == "unknown"
    assert ora.version.version_label(other) == "v2.3.4"


def test_the_terminal_headers_show_the_same_version(monkeypatch) -> None:
    monkeypatch.setattr(ora.version.importlib.metadata, "version", _installed("0.0.1"))
    assert ora.version.version_label() == f"v{_pyproject_version()}"
    assert follow.version_label is ora_backend.version_label is ora.version.version_label
    assert not hasattr(follow, "_version")


def test_hints_name_the_command_the_user_types(monkeypatch) -> None:
    from ora.agent.code_mode import sandbox

    monkeypatch.delenv(ora.version.COMMAND_ENV, raising=False)
    assert ora.version.ora_command() == "./scripts/ora"            # a source checkout's wrapper
    assert sandbox.interpreter_remedy().startswith("run ./scripts/ora setup --env robodojo")
    monkeypatch.setenv(ora.version.COMMAND_ENV, "ora")             # exported by the installer's command
    assert ora.version.ora_command() == "ora"
    assert sandbox.interpreter_remedy().startswith("run ora setup --env robodojo")
    monkeypatch.setenv(ora.version.COMMAND_ENV, "  ")
    assert ora.version.ora_command() == "./scripts/ora"


def test_every_release_manifest_declares_the_pyproject_version_and_license() -> None:
    version = _pyproject_version()
    for package in ("ora-pi", "ora-pi/web"):
        manifest = json.loads((ROOT / package / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((ROOT / package / "package-lock.json").read_text(encoding="utf-8"))
        assert (manifest["version"], lock["version"], lock["packages"][""]["version"]) == (version,) * 3, package
        assert manifest["license"] == lock["packages"][""]["license"] == "Apache-2.0", package
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert str(citation["version"]) == version
    assert citation["license"] == "Apache-2.0"
