"""Renaming or exporting a transcript must not replace its recorded identity."""

import json
import subprocess
import sys
from pathlib import Path

from test_batch_analyze import SCRIPT, batch_analyze


def write(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return path


def test_export_keeps_runtime_identity_in_document_and_manifest(tmp_path: Path) -> None:
    source = write(
        tmp_path / "renamed.jsonl",
        [
            {
                "type": "runtime-config",
                "sessionId": "real-session",
                "entrypoint": "qoderwork",
                "version": "1.2.3",
            },
            {"type": "user", "message": {"content": "inspect project"}},
        ],
    )
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--file",
            str(source),
            "--export-atif",
            str(tmp_path / "export"),
            "--min-steps",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = json.loads(run.stdout)
    document = json.loads(Path(manifest["_atif_file"]).read_text(encoding="utf-8"))
    assert manifest["_session_id"] == document["session_id"] == "real-session"
    assert document["agent"]["name"] == "qoderwork"
    assert document["agent"]["version"] == "1.2.3"
    assert manifest["_source_file"] == str(source)


def test_claude_message_identity_survives_a_renamed_export(tmp_path: Path) -> None:
    source = write(
        tmp_path / ".claude" / "renamed.jsonl",
        [
            {
                "type": "user",
                "sessionId": "original",
                "version": "2.1.0",
                "message": {"content": "question"},
            },
            {"type": "assistant", "sessionId": "original", "message": {"content": "answer"}},
        ],
    )
    doc = batch_analyze.jsonl_to_atif(source)
    assert doc["session_id"] == "original"
    assert doc["agent"]["version"] == "2.1.0"
    assert doc["agent"]["name"] == "claude-code"
    assert [step["message"] for step in doc["steps"]] == ["question", "answer"]


def test_late_runtime_metadata_has_precedence_over_message_fallback(tmp_path: Path) -> None:
    source = write(
        tmp_path / "session.jsonl",
        [
            {
                "type": "user",
                "sessionId": "fallback",
                "version": "old",
                "message": {"content": "question"},
            },
            {
                "type": "runtime-config",
                "sessionId": "authoritative",
                "version": "new",
                "entrypoint": "qoder",
            },
        ],
    )
    doc = batch_analyze.jsonl_to_atif(source)
    assert doc["session_id"] == "authoritative"
    assert doc["agent"]["version"] == "new"
    assert doc["agent"]["name"] == "qoder"


def test_unusable_identity_keeps_filename_fallback(tmp_path: Path) -> None:
    source = write(
        tmp_path / "fallback.jsonl",
        [
            {"type": "runtime-config", "sessionId": None, "version": {}, "entrypoint": []},
            {"type": "user", "sessionId": "", "version": None, "message": {"content": "question"}},
        ],
    )
    doc = batch_analyze.jsonl_to_atif(source)
    assert doc["session_id"] == "fallback"
    assert doc["agent"] == {"name": "unknown", "version": "", "model_name": None}
