import json

import pytest

from audiocompose.bundle_contract import load_bundle_identity
from audiocompose.errors import AudioComposeError


@pytest.mark.parametrize(
    ("format_name", "kind"), [("ssmdconvert.book", "book"), ("ssmdconvert.document", "document")]
)
def test_bundle_identity_uses_manifest_format(tmp_path, format_name, kind):
    root = tmp_path / "arbitrary.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"format": format_name, "schema_version": 1}))
    identity = load_bundle_identity(root)
    assert identity.root == root
    assert identity.kind == kind
    assert identity.manifest_path == root / "manifest.json"


def test_bundle_identity_rejects_non_root_and_unsupported_format(tmp_path):
    with pytest.raises(AudioComposeError, match="invalid SSMD bundle"):
        load_bundle_identity(tmp_path / "chapters")
    root = tmp_path / "other.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"format": "other", "schema_version": 1}))
    with pytest.raises(AudioComposeError, match="unsupported SSMD bundle format"):
        load_bundle_identity(root)


def test_bundle_identity_rejects_symlink_manifest(tmp_path):
    root = tmp_path / "linked.ssmd-bundle"
    root.mkdir()
    external = tmp_path / "manifest.json"
    external.write_text(json.dumps({"format": "ssmdconvert.book", "schema_version": 1}))
    (root / "manifest.json").symlink_to(external)
    with pytest.raises(AudioComposeError, match="invalid SSMD bundle"):
        load_bundle_identity(root)


def test_bundle_cli_parser_and_removed_flat_commands(capsys):
    from audiocompose.cli import _parser

    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["bundle", "compose", "--help"])
    assert exc.value.code == 0
    assert "--save-audiojobs" in capsys.readouterr().out
    for command in ("compose" + "-book", "compose" + "-document"):
        with pytest.raises(SystemExit):
            _parser().parse_args([command, "unused"])


@pytest.mark.parametrize(
    ("format_name", "kind"), [("ssmdconvert.book", "book"), ("ssmdconvert.document", "document")]
)
def test_compose_bundle_dispatches_by_manifest_kind(tmp_path, monkeypatch, format_name, kind):
    from types import SimpleNamespace

    import audiocompose.book as book_module
    import audiocompose.document as document_module
    from audiocompose.bundle import BundleComposeOptions, compose_bundle

    root = tmp_path / "story.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"format": format_name, "schema_version": 1}))
    captured = {}

    def fake_compose(path, **kwargs):
        captured.update(path=path, **kwargs)
        return SimpleNamespace(exit_code=0, to_dict=lambda: {"ok": True})

    monkeypatch.setattr(book_module, "compose_book", fake_compose)
    monkeypatch.setattr(document_module, "compose_document", fake_compose)
    options = BundleComposeOptions(chapters="all", save_audiojobs=tmp_path / "jobs")
    result = compose_bundle(root, variant="v", options=options)
    assert result.kind == kind
    assert result.exit_code == 0
    assert captured["path"] == root
    if kind == "document":
        assert captured["save_audiojob"] == tmp_path / "jobs" / "document-0001.audiojob"
    else:
        assert captured["save_audiojobs"] == tmp_path / "jobs"
