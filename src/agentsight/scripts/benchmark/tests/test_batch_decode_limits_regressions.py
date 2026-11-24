"""Decoder-rejected records must not terminate valid neighboring trajectories."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_batch_analyze import batch_analyze


class BatchDecodeLimitsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def records(self) -> list[str]:
        return [
            json.dumps(record)
            for record in [
                {"type": "runtime-config", "entrypoint": "qoder"},
                {"type": "user", "message": {"content": "Inspect the project. " * 100}},
                {
                    "type": "assistant",
                    "message": {
                        "id": "reply",
                        "content": "Done.",
                        "usage": {"input_tokens": 17},
                    },
                },
            ]
        ]

    def rejected_records(self) -> list[str]:
        limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
        records = [
            '{"ignored":'
            + "[" * (sys.getrecursionlimit() + 100)
            + "0"
            + "]" * (sys.getrecursionlimit() + 100)
            + "}"
        ]
        if limit:
            records.append('{"ignored":' + "9" * (limit + 1) + "}")
        return records

    def write(self, name: str, lines: list[str]) -> Path:
        path = self.root / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_rejected_records_preserve_both_sides_and_source_accounting(self) -> None:
        path = self.write("session.jsonl", self.records())
        expected = batch_analyze.jsonl_to_atif(path)
        for rejected in self.rejected_records():
            with self.assertRaises((ValueError, RecursionError)):
                json.loads(rejected)
            for position in range(4):
                with self.subTest(prefix=rejected[:25], position=position):
                    lines = self.records()
                    lines.insert(position, rejected)
                    self.write(path.name, lines)
                    self.assertEqual(batch_analyze.jsonl_to_atif(path), expected)

    def test_an_entirely_rejected_file_is_skipped(self) -> None:
        path = self.write("invalid.jsonl", self.rejected_records())
        self.assertIsNone(batch_analyze.jsonl_to_atif(path))

    def test_existing_syntax_error_and_valid_nested_metadata_controls(self) -> None:
        path = self.write("session.jsonl", self.records())
        expected = batch_analyze.jsonl_to_atif(path)
        self.write(path.name, ["{", '{"ignored":[[[1]]]}', *self.records()])
        self.assertEqual(batch_analyze.jsonl_to_atif(path), expected)

    def cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, batch_analyze.__file__, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

    def test_real_export_continues_after_rejected_records(self) -> None:
        self.write("first.jsonl", [*self.records(), *self.rejected_records()])
        self.write("second.jsonl", self.records())
        result = self.cli(
            "--dir",
            str(self.root),
            "--min-steps",
            "1",
            "--export-atif",
            str(self.root / "exports"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Done: 2 exported, 0 skipped, 0 failed", result.stderr)
        for record in map(json.loads, result.stdout.splitlines()):
            doc = json.loads(Path(record["_atif_file"]).read_text(encoding="utf-8"))
            self.assertEqual(len(doc["steps"]), 2)
            self.assertEqual(doc["agent"]["name"], "qoder")

    @unittest.skipUnless(
        os.name == "posix", "executable analyzer fixture requires POSIX"
    )
    def test_serial_and_parallel_analysis_keep_valid_neighbors(self) -> None:
        self.write("first.jsonl", [*self.records(), *self.rejected_records()])
        self.write("second.jsonl", self.records())
        analyzer = self.root / "analyzer"
        analyzer.write_text(
            f"#!{sys.executable}\nimport json,sys\n"
            'with open(sys.argv[1], encoding="utf-8") as source: doc=json.load(source)\n'
            'print(json.dumps({"steps":len(doc["steps"])}))\n',
            encoding="utf-8",
        )
        analyzer.chmod(0o700)
        for jobs in ("1", "2"):
            with self.subTest(jobs=jobs):
                result = self.cli(
                    "--dir",
                    str(self.root),
                    "--min-steps",
                    "1",
                    "--dim",
                    "perf",
                    "--jobs",
                    jobs,
                    "--binary",
                    str(analyzer),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Done: 2 analyzed, 0 skipped, 0 failed", result.stderr)
                documents = []
                remaining = result.stdout.strip()
                while remaining:
                    document, end = json.JSONDecoder().raw_decode(remaining)
                    documents.append(document)
                    remaining = remaining[end:].strip()
                self.assertEqual([doc["steps"] for doc in documents], [2, 2])


if __name__ == "__main__":
    unittest.main()
