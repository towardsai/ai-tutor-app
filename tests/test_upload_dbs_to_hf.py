from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from data.scraping_scripts import upload_dbs_to_hf as uploader


def _write_minimal_kb(root: Path) -> None:
    kb = root / "kb"
    (kb / "generated").mkdir(parents=True)
    (kb / "generated" / "corpus_manifest.jsonl").write_text(
        '{"doc_id":"1"}\n', encoding="utf-8"
    )
    (kb / "wiki").mkdir()
    (kb / "wiki" / "index.md").write_text("# Index\n", encoding="utf-8")


def test_create_kb_archive_is_deterministic_and_preserves_tree(
    tmp_path: Path,
) -> None:
    _write_minimal_kb(tmp_path)

    archive = uploader.create_kb_archive(str(tmp_path))
    first_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive = uploader.create_kb_archive(str(tmp_path))
    second_digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    assert first_digest == second_digest
    assert (tmp_path / "kb" / "wiki" / "index.md").is_file()
    with tarfile.open(archive, "r:gz") as tar:
        assert "kb/generated/corpus_manifest.jsonl" in tar.getnames()
        assert "kb/wiki/index.md" in tar.getnames()


def test_create_kb_archive_refuses_incomplete_kb(tmp_path: Path) -> None:
    (tmp_path / "kb").mkdir()

    with pytest.raises(FileNotFoundError, match="incomplete KB"):
        uploader.create_kb_archive(str(tmp_path))

    assert not (tmp_path / "kb.tar.gz").exists()


class _FakeApi:
    def __init__(self, remote_files: list[str]) -> None:
        self.remote_files = remote_files
        self.events: list[str] = []
        self.deleted: list[str] = []

    def upload_folder(self, **_kwargs) -> None:
        self.events.append("upload")

    def list_repo_files(self, _repo_id: str, *, repo_type: str) -> list[str]:
        assert repo_type == "dataset"
        self.events.append("list")
        return self.remote_files

    def create_commit(self, **kwargs) -> None:
        self.events.append("prune")
        self.deleted = [operation.path_in_repo for operation in kwargs["operations"]]


def test_upload_verifies_archive_then_prunes_legacy_kb(tmp_path: Path) -> None:
    (tmp_path / "chroma-db-all_sources").mkdir()
    (tmp_path / "chroma-db-all_sources" / "chroma.sqlite3").write_bytes(b"db")
    (tmp_path / "kb.tar.gz").write_bytes(b"archive")
    api = _FakeApi(
        [
            "chroma-db-all_sources/chroma.sqlite3",
            "kb.tar.gz",
            "kb/wiki/index.md",
            "kb/generated/corpus_manifest.jsonl",
        ]
    )

    with patch(
        "data.scraping_scripts.upload_dbs_to_hf.validate_hf_access", return_value=api
    ):
        uploader.upload_bundle(
            "org/private-bundle",
            folder_path=str(tmp_path),
            allow_patterns=["chroma-db-all_sources/**", "kb.tar.gz"],
            prune_allow_patterns=[
                "chroma-db-all_sources/**",
                "kb.tar.gz",
                "kb/**",
            ],
        )

    assert api.events == ["upload", "list", "prune"]
    assert api.deleted == [
        "kb/generated/corpus_manifest.jsonl",
        "kb/wiki/index.md",
    ]


def test_upload_refuses_to_prune_when_archive_is_missing_remotely(
    tmp_path: Path,
) -> None:
    (tmp_path / "kb.tar.gz").write_bytes(b"archive")
    api = _FakeApi(["kb/wiki/index.md"])

    with patch(
        "data.scraping_scripts.upload_dbs_to_hf.validate_hf_access", return_value=api
    ):
        with pytest.raises(RuntimeError, match="refusing to prune"):
            uploader.upload_bundle(
                "org/private-bundle",
                folder_path=str(tmp_path),
                allow_patterns=["kb.tar.gz"],
                prune_allow_patterns=["kb.tar.gz", "kb/**"],
            )

    assert api.events == ["upload", "list"]
    assert api.deleted == []
