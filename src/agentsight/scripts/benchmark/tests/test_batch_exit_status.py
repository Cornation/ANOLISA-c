"""Automation must see failures after all selected sessions have been attempted."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_batch_analyze import SCRIPT


def source(path: Path, text: str = "question") -> Path:
    path.write_text(
        json.dumps({"type": "user", "message": {"content": text * 200}}) + "\n", encoding="utf-8"
    )
    return path


def run(*args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize("jobs", [1, 2])
def test_failed_spawn_exits_nonzero_after_processing_the_batch(tmp_path: Path, jobs: int) -> None:
    source(tmp_path / "first.jsonl")
    source(tmp_path / "second.jsonl")
    output = run(
        "--dir",
        tmp_path,
        "--dim",
        "perf",
        "--binary",
        tmp_path / "missing-analyzer",
        "--min-steps",
        1,
        "--jobs",
        jobs,
    )
    assert "Done: 0 analyzed, 0 skipped, 2 failed" in output.stderr
    assert output.stdout == ""
    assert output.returncode == 1


def test_failed_export_exits_nonzero_without_a_success_manifest(tmp_path: Path) -> None:
    session = source(tmp_path / "source.jsonl")
    destination = tmp_path / "not-a-directory"
    destination.write_text("preserve this file", encoding="utf-8")
    output = run("--file", session, "--export-atif", destination, "--min-steps", 1)
    assert "Done: 0 exported, 0 skipped, 1 failed" in output.stderr
    assert output.returncode == 1
    assert output.stdout == ""
    assert destination.read_text(encoding="utf-8") == "preserve this file"


def test_successful_export_and_intentional_skip_remain_successful(tmp_path: Path) -> None:
    session = source(tmp_path / "source.jsonl")
    success = run("--file", session, "--export-atif", tmp_path / "export", "--min-steps", 1)
    assert success.returncode == 0
    assert Path(json.loads(success.stdout)["_atif_file"]).is_file()
    skipped = run("--file", session, "--export-atif", tmp_path / "skipped", "--min-steps", 2)
    assert skipped.returncode == 0
    assert "1 skipped, 0 failed" in skipped.stderr
    assert skipped.stdout == ""
