"""Recover lossless judge inputs from durable eval artifacts.

``handgrade_sheet.csv`` is intentionally convenient for humans, but its
question and answer columns are display previews.  Subscription and API judges
must instead grade the complete content saved in ``bundles.jsonl``.  This module
does that join once and attaches enough integrity metadata to audit every CSV
chunk without exposing the experiment arm to a blinded grader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .common import load_jsonl
from .grade import faithfulness_evidence, index_battery


INTEGRITY_COLS = [
    "source_bundle_line",
    "question_chars",
    "answer_chars",
    "reference_chars",
    "question_sha256",
    "answer_sha256",
    "reference_sha256",
    "content_sha256",
]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_sha256(row: dict[str, str]) -> str:
    canonical = json.dumps(
        {
            "question": row.get("question") or "",
            "criterion": row.get("criterion") or "",
            "reference": row.get("reference") or "",
            "answer": row.get("answer") or "",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256_text(canonical)


def verify_integrity(row: dict[str, str]) -> None:
    """Reject a chunk row whose declared full-input metadata does not match."""

    row_id = row.get("sheet_row_id") or "<missing sheet_row_id>"
    for field in ("question", "answer", "reference"):
        value = row.get(field) or ""
        if row.get(f"{field}_chars") != str(len(value)):
            raise ValueError(f"{row_id}: {field}_chars integrity mismatch")
        if row.get(f"{field}_sha256") != sha256_text(value):
            raise ValueError(f"{row_id}: {field}_sha256 integrity mismatch")
    if row.get("content_sha256") != content_sha256(row):
        raise ValueError(f"{row_id}: content_sha256 integrity mismatch")


def _battery_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    repo_relative = Path(__file__).resolve().parents[1] / path
    if repo_relative.exists():
        return repo_relative
    run_relative = run_dir / path
    if run_relative.exists():
        return run_relative
    raise FileNotFoundError(f"Battery file not found: {raw_path!r} for {run_dir}")


def _full_reference(
    bundle: dict[str, Any], item_type: str, record: dict[str, Any]
) -> str:
    if item_type == "faithfulness":
        return faithfulness_evidence(bundle)
    battery_type = bundle.get("battery_type")
    if battery_type == "singleturn" and item_type in {
        "key_point",
        "behavior",
        "holistic",
    }:
        return record.get("reference_answer") or ""
    if battery_type == "replay" and item_type == "replay_reply":
        return record.get("reference_reply") or ""
    # Session probes carry their expected facts in criterion, while session
    # holistic/persona checks intentionally have no external reference.
    return ""


def _assert_preview_matches(
    *, field: str, preview: str, full: str, limit: int, sheet_row_id: str
) -> None:
    expected = full[:limit]
    # Accept a future lossless sheet as well as today's preview-shaped sheet.
    if preview not in {expected, full}:
        raise ValueError(
            f"{sheet_row_id}: {field} does not match bundles/battery content; "
            "refusing to grade a stale or misjoined sheet"
        )


def hydrate_full_inputs(
    run_dir: Path, sheet_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Replace sheet previews with full bundle/battery content.

    The join is deliberately strict: a stale sheet or duplicate ``run_id`` is
    an error rather than a plausible-looking but incorrectly paired judgment.
    """

    bundles = load_jsonl(run_dir / "bundles.jsonl")
    if not bundles:
        raise ValueError(f"No bundles in {run_dir}")
    by_run_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for line, bundle in enumerate(bundles, start=1):
        run_id = bundle.get("run_id")
        if not run_id:
            raise ValueError(f"{run_dir}/bundles.jsonl:{line}: missing run_id")
        if run_id in by_run_id:
            raise ValueError(f"{run_dir}: duplicate bundle run_id {run_id!r}")
        by_run_id[run_id] = (line, bundle)

    battery_path = _battery_path(run_dir, bundles[0]["battery_path"])
    records = index_battery(load_jsonl(battery_path))
    hydrated: list[dict[str, str]] = []
    for sheet in sheet_rows:
        sheet_row_id = sheet.get("sheet_row_id") or "<missing sheet_row_id>"
        joined = by_run_id.get(sheet.get("run_id") or "")
        if joined is None:
            raise ValueError(
                f"{sheet_row_id}: run_id {sheet.get('run_id')!r} has no bundle"
            )
        line, bundle = joined
        record = records.get(bundle.get("unit_id"), {})
        question = bundle.get("query") or ""
        answer = bundle.get("answer") or ""
        reference = _full_reference(bundle, sheet.get("item_type") or "", record)

        _assert_preview_matches(
            field="question",
            preview=sheet.get("question") or "",
            full=question,
            limit=600,
            sheet_row_id=sheet_row_id,
        )
        _assert_preview_matches(
            field="answer",
            preview=sheet.get("answer") or "",
            full=answer,
            limit=4_000,
            sheet_row_id=sheet_row_id,
        )
        # grade.sheet_row uses 2K for normal references and 200K for complete
        # faithfulness evidence.  Verifying the preview catches stale battery
        # files while allowing this function to restore the complete value.
        _assert_preview_matches(
            field="reference",
            preview=sheet.get("reference") or "",
            full=reference,
            limit=200_000 if sheet.get("item_type") == "faithfulness" else 2_000,
            sheet_row_id=sheet_row_id,
        )

        row = dict(sheet)
        row.update(
            {
                "question": question,
                "answer": answer,
                "reference": reference,
                "source_bundle_line": str(line),
                "question_chars": str(len(question)),
                "answer_chars": str(len(answer)),
                "reference_chars": str(len(reference)),
                "question_sha256": sha256_text(question),
                "answer_sha256": sha256_text(answer),
                "reference_sha256": sha256_text(reference),
            }
        )
        row["content_sha256"] = content_sha256(row)
        hydrated.append(row)
    return hydrated
