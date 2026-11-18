"""Named context schemes: discovery from their files, validation and each scheme's contract."""
import json
import shutil
from pathlib import Path

import pytest

from ora.agent.composition.baseline import BASELINE_ID, baseline_policy, load_native_baseline
from ora.agent.composition.context_scheme import (
    CONTEXT_FIELDS, SCHEMES_DIR, context_schemes, load_context_scheme,
)
from ora.agent.context.lossless.management import (
    APPEND_ONLY_SCHEMAS, EXCHANGE_AGING_SCHEMAS, LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS, validate_context_management,
)

ROOT = Path(__file__).resolve().parents[4]
SCHEMES = context_schemes(ROOT)
#: Schemes of append-only context: management v10, which ages exchanges into recoverable references and must
#: provide the artifact reader (the v8 and v9 schemes were retired on 2026-11-05).
APPEND_ONLY = ("context_keep_written", "context_keep_written_unaged_reasoning")


def _raw(scheme_id):
    return json.loads((ROOT / SCHEMES[scheme_id]).read_text(encoding="utf-8"))


def test_every_scheme_file_is_discovered_by_its_id():
    files = sorted(path.name for path in (ROOT / SCHEMES_DIR).glob("*.json"))
    assert sorted(Path(path).name for path in SCHEMES.values()) == files
    assert {"context_lossless_v6", *APPEND_ONLY} <= set(SCHEMES)
    for scheme_id, path in SCHEMES.items():
        assert _raw(scheme_id)["id"] == scheme_id and path.startswith(f"{SCHEMES_DIR}/")


@pytest.mark.parametrize("scheme_id", APPEND_ONLY)
def test_an_append_only_scheme_has_the_reader_only_where_it_ages_exchanges(scheme_id):
    policy = load_context_scheme(ROOT, scheme_id)["policy"]
    schema = validate_context_management(policy["context_management"])["schema"]
    assert schema in APPEND_ONLY_SCHEMAS
    assert policy["context_tools"] == (["artifact_read"] if schema in EXCHANGE_AGING_SCHEMAS else [])


def test_the_default_baseline_names_the_lossless_v6_scheme():
    """The default baseline's context half is the scheme ``context_lossless_v6``; ``ora local`` names it too."""
    from ora.agent.composition.ora_local import DEFAULT_LOCAL_COMPOSITION

    catalog = load_native_baseline(ROOT)
    assert catalog["id"] == BASELINE_ID and catalog["context_scheme"] == "context_lossless_v6"
    assert DEFAULT_LOCAL_COMPOSITION["context_scheme"] == catalog["context_scheme"]
    assert not set(catalog["policy"]) & set(CONTEXT_FIELDS)
    scheme = load_context_scheme(ROOT, "context_lossless_v6")
    policy = scheme["policy"]
    assert validate_context_management(policy["context_management"])["schema"] == "ora.native-context-management.v6"
    assert policy["context_management"]["schema"] in LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS
    # Without context_tools lossless retention provides the reader; no prompt of its own.
    assert "context_tools" not in policy and scheme["prompt"] is None
    assert baseline_policy(ROOT, catalog) == {**catalog["policy"], **policy}


def test_an_unregistered_scheme_is_refused_with_the_known_ids():
    with pytest.raises(ValueError, match="must be one of") as refused:
        load_context_scheme(ROOT, "context_unknown_v1")
    assert all(scheme_id in str(refused.value) for scheme_id in SCHEMES)


def _tree(tmp_path: Path) -> Path:
    """A source tree holding a copy of the scheme directory and the policy.lock that lists it."""
    root = tmp_path / "tree"
    shutil.copytree(ROOT / SCHEMES_DIR, root / SCHEMES_DIR)
    shutil.copy(ROOT / "policy.lock", root / "policy.lock")
    return root


def test_a_second_file_with_an_id_already_seen_is_refused(tmp_path):
    root = _tree(tmp_path)
    (root / SCHEMES_DIR / "zz-copy.json").write_text(json.dumps(_raw("context_lossless_v6")))
    with pytest.raises(ValueError, match="context scheme context_lossless_v6 is defined twice"):
        context_schemes(root)
    with pytest.raises(ValueError, match="defined twice"):
        load_context_scheme(root, "context_keep_written")


@pytest.mark.parametrize("body", ["not json", "[]", json.dumps({"schema": "ora.context-scheme.v0", "id": "x"}),
                                  json.dumps({"schema": "ora.context-scheme.v1", "id": 3})])
def test_a_file_that_is_not_a_scheme_is_refused(tmp_path, body):
    root = _tree(tmp_path)
    (root / SCHEMES_DIR / "draft.json").write_text(body)
    with pytest.raises(ValueError, match=f"{SCHEMES_DIR}/draft.json is not"):
        context_schemes(root)


def test_a_discovered_scheme_still_needs_its_policy_lock_entry(tmp_path):
    root = _tree(tmp_path)
    (root / SCHEMES_DIR / "lossless-v6-copy.json").write_text(
        json.dumps({**_raw("context_lossless_v6"), "id": "context_lossless_v6_copy"}))
    assert "context_lossless_v6_copy" in context_schemes(root)
    with pytest.raises(ValueError, match="policy.lock"):
        load_context_scheme(root, "context_lossless_v6_copy")


@pytest.mark.parametrize("scheme_id", sorted(SCHEMES))
def test_a_scheme_prompt_is_a_markdown_file_beside_the_scheme(scheme_id):
    prompt = load_context_scheme(ROOT, scheme_id)["prompt"]
    assert "prompt" not in _raw(scheme_id) if prompt is None else (
        Path(prompt).parent == Path(SCHEMES[scheme_id]).parent and prompt.endswith(".md")
        and (ROOT / prompt).read_text(encoding="utf-8").strip())


@pytest.mark.parametrize("prompt", ["../planner.md", "notes.txt", 3])
def test_a_scheme_prompt_outside_its_directory_is_refused(tmp_path, monkeypatch, prompt):
    import ora.agent.composition.context_scheme as module

    raw = {**_raw("context_keep_written"), "prompt": prompt}
    monkeypatch.setattr(module, "locked_asset", lambda root, path, label: type(
        "Asset", (), {"data": json.dumps(raw).encode(), "sha256": "0" * 64})())
    with pytest.raises(ValueError, match="prompt must be a Markdown file beside the scheme"):
        load_context_scheme(ROOT, "context_keep_written")
