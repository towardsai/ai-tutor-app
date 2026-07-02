from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app import config


def _patched_bundle(tmp_path: Path) -> ExitStack:
    db_dir = tmp_path / "chroma"
    kb_dir = tmp_path / "kb"
    stack = ExitStack()
    for target, value in (
        ("VECTOR_DB_DIR", db_dir),
        ("CHROMA_SQLITE_PATH", db_dir / "chroma.sqlite3"),
        ("DOCUMENT_DICT_PATH", db_dir / "document_dict.pkl"),
        ("BM25_INDEX_PATH", db_dir / "bm25.pkl"),
        ("KB_MANIFEST_PATH", kb_dir / "generated" / "corpus_manifest.jsonl"),
        ("KB_INDEX_PATH", kb_dir / "wiki" / "index.md"),
        ("KB_AGENTS_PATH", kb_dir / "AGENTS.md"),
        ("KB_AGENTS_TEMPLATE_PATH", tmp_path / "template.md"),
    ):
        stack.enter_context(patch(f"app.config.{target}", str(value)))
    stack.enter_context(patch("app.config._BUNDLE_READY", False))
    return stack


def _write_bundle_files(tmp_path: Path, *, include_bm25: bool = True) -> None:
    db_dir = tmp_path / "chroma"
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "chroma.sqlite3").write_bytes(b"")
    (db_dir / "document_dict.pkl").write_bytes(b"")
    if include_bm25:
        (db_dir / "bm25.pkl").write_bytes(b"")
    (tmp_path / "kb" / "generated").mkdir(parents=True, exist_ok=True)
    (tmp_path / "kb" / "generated" / "corpus_manifest.jsonl").write_text(
        "", encoding="utf-8"
    )
    (tmp_path / "kb" / "wiki").mkdir(parents=True, exist_ok=True)
    (tmp_path / "kb" / "wiki" / "index.md").write_text("# index", encoding="utf-8")
    (tmp_path / "template.md").write_text("# KB rules", encoding="utf-8")


def test_missing_bm25_triggers_download(tmp_path: Path) -> None:
    # An interrupted download that left BM25 behind must be repaired, not
    # silently degrade hybrid retrieval to dense-only forever.
    _write_bundle_files(tmp_path, include_bm25=False)

    with _patched_bundle(tmp_path):
        with patch("huggingface_hub.snapshot_download") as snapshot_download:
            config.ensure_local_vector_db()

        snapshot_download.assert_called_once()
        # The mock download produced nothing, so readiness must stay false
        # (the next call retries instead of trusting a broken bundle).
        assert config._BUNDLE_READY is False


def test_complete_bundle_skips_download_and_caches_readiness(tmp_path: Path) -> None:
    _write_bundle_files(tmp_path)

    with _patched_bundle(tmp_path):
        with patch("huggingface_hub.snapshot_download") as snapshot_download:
            config.ensure_local_vector_db()
            assert snapshot_download.call_count == 0
            assert config._BUNDLE_READY is True
            assert (tmp_path / "kb" / "AGENTS.md").read_text(
                encoding="utf-8"
            ) == "# KB rules"

            # Once verified, later calls are flag-checks: no re-stat, no
            # download attempt even if files vanish mid-process.
            (tmp_path / "chroma" / "bm25.pkl").unlink()
            config.ensure_local_vector_db()
            assert snapshot_download.call_count == 0


def _repo_not_found_error():
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import RepositoryNotFoundError
    return RepositoryNotFoundError


def test_download_bundle_uses_public_when_no_token() -> None:
    with patch("huggingface_hub.get_token", return_value=None):
        with patch("app.config._snapshot_bundle") as snapshot:
            config._download_bundle()
    snapshot.assert_called_once()
    assert snapshot.call_args.args[0] == config.PUBLIC_VECTOR_DB_REPO_ID
    # token=False = anonymous; None would let huggingface_hub re-resolve and
    # send a cached token, which can 401 even against the public repo.
    assert snapshot.call_args.kwargs["token"] is False


def test_download_bundle_uses_private_when_token_present() -> None:
    with patch("huggingface_hub.get_token", return_value="tok"):
        with patch("app.config._snapshot_bundle") as snapshot:
            config._download_bundle()
    snapshot.assert_called_once()
    assert snapshot.call_args.args[0] == config.VECTOR_DB_REPO_ID


def test_download_bundle_falls_back_to_public_on_no_access() -> None:
    import requests

    repo_error = _repo_not_found_error()
    response = requests.Response()
    response.status_code = 401
    calls: list[tuple[str, object]] = []

    def side_effect(repo_id: str, *, token) -> None:
        calls.append((repo_id, token))
        if repo_id == config.VECTOR_DB_REPO_ID:
            raise repo_error("no access", response=response)

    with patch("huggingface_hub.get_token", return_value="tok"):
        with patch("app.config._snapshot_bundle", side_effect=side_effect):
            config._download_bundle()

    assert calls == [
        (config.VECTOR_DB_REPO_ID, "tok"),
        # The fallback must be anonymous (False), not token re-resolution.
        (config.PUBLIC_VECTOR_DB_REPO_ID, False),
    ]


def test_ensure_kb_agents_md_writes_once_and_atomically(tmp_path: Path) -> None:
    _write_bundle_files(tmp_path)

    with _patched_bundle(tmp_path):
        with patch("app.config.os.replace", wraps=os.replace) as replace:
            config.ensure_kb_agents_md()
            assert replace.call_count == 1
            # Unchanged content: no rewrite at all.
            config.ensure_kb_agents_md()
            assert replace.call_count == 1

            # Changed template: rewritten through the atomic path.
            (tmp_path / "template.md").write_text("# KB rules v2", encoding="utf-8")
            config.ensure_kb_agents_md()
            assert replace.call_count == 2

        agents_path = tmp_path / "kb" / "AGENTS.md"
        assert agents_path.read_text(encoding="utf-8") == "# KB rules v2"
        assert not agents_path.with_name("AGENTS.md.tmp").exists()
