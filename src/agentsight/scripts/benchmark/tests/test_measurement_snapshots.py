"""One report must reconcile one CSV snapshot, with one decompression pass."""

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from test_benchmark import render_report
from test_campaign_evidence import campaign_evidence, summary, thresholds, write_recovery_artifacts


def write_metrics(path: Path, cpu: int = 70) -> None:
    content = (
        "timestamp,input_qps,process_alive,cpu_pct,rss_mb,completed,ring_buffer_dropped,channel_dropped\n"
        f"100,10,1,{cpu},50,0,0,0\n101,10,1,{cpu},60,100,0,0\n"
    )
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(content)
    else:
        path.write_text(content)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("replace_after_read", [False, True])
def test_report_reuses_one_measurement_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compressed: bool, replace_after_read: bool
) -> None:
    metrics = tmp_path / ("metrics.csv.gz" if compressed else "metrics.csv")
    write_metrics(metrics)
    original = render_report.metric_rows
    reads = []

    def observe(path: Path | None) -> list[dict[str, str]]:
        rows = original(path)
        reads.append(path)
        if replace_after_read and len(reads) == 1:
            write_metrics(metrics, cpu=20)
        return rows

    rendered: dict[str, Any] = {}
    original_render = render_report.render_markdown

    def capture(**arguments: Any) -> str:
        rendered.update(arguments)
        return original_render(**arguments)

    monkeypatch.setattr(render_report, "metric_rows", observe)
    monkeypatch.setattr(render_report, "render_markdown", capture)
    report, machine = tmp_path / "report.md", tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_report.py",
            "--protocol",
            "sse",
            "--qps",
            "10",
            "--duration",
            "2",
            "--metrics",
            str(metrics),
            "--output",
            str(report),
            "--summary-output",
            str(machine),
        ],
    )
    assert render_report.main() == 0
    assert reads == [metrics]
    assert rendered["resources"]["cpu_pct"]["avg"] == 70
    assert rendered["qps_resources"]["10"]["cpu_pct"]["avg"] == 70
    assert rendered["drops"]["completed"] == 100
    assert json.loads(machine.read_text())["resources"] == rendered["resources"]
    assert "70.00" in report.read_text()


@pytest.mark.parametrize("compressed", [False, True])
def test_recovery_reads_all_resource_gates_in_one_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compressed: bool
) -> None:
    phases = {}
    for label in campaign_evidence.RECOVERY_PHASES:
        path = tmp_path / label / "run-result.json"
        phases[label] = (
            path,
            {
                "summary": summary(),
                "started_at_unix": 0,
                "evaluation": {"verdict": "FAIL" if label == "overload" else "PASS"},
            },
        )
    recover = phases["recover"][0]
    write_recovery_artifacts(recover)
    path = recover.parent / "measurement/metrics.csv"
    if compressed:
        with gzip.open(path.with_suffix(".csv.gz"), "wb") as handle:
            handle.write(path.read_bytes())
        path.unlink()
        path = path.with_suffix(".csv.gz")
    original = gzip.open if compressed else Path.open
    opened = []

    def counted(target: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(target) == path:
            opened.append(target)
        return original(target, *args, **kwargs)

    if compressed:
        monkeypatch.setattr(gzip, "open", counted)
    else:
        monkeypatch.setattr(Path, "open", counted)
    outcome = campaign_evidence.recovery_outcome(
        phases, {"tolerance_ratio": 0.1, "recovery_window_seconds": 2}, thresholds()
    )
    assert outcome["verdict"] == "PASS"
    assert set(outcome["seconds"].values()) == {1.0}
    assert len(opened) == 1


def test_legacy_individual_summary_entrypoints_remain_compatible(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    write_metrics(path)
    before = path.read_bytes()
    assert render_report.summarize_resources(path)["rss_mb"]["max"] == 60
    assert render_report.summarize_qps_resources(path)["10"]["cpu_pct"]["avg"] == 70
    assert render_report.summarize_drops(path, 100)["completed"] == 100
    assert path.read_bytes() == before


def test_missing_metrics_keep_existing_unavailable_results(tmp_path: Path) -> None:
    path = tmp_path / "absent.csv"
    assert render_report.summarize_resources(path)["rss_mb"]["avg"] is None
    assert render_report.summarize_qps_resources(path) == {}
    assert render_report.summarize_drops(path, 0)["drop_rate"] is None


def test_explicit_empty_snapshot_does_not_fall_back_to_reopening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metrics.csv"
    write_metrics(path)

    def unexpected_read(_: Path | None) -> list[dict[str, str]]:
        pytest.fail("an explicitly supplied snapshot must not be replaced by a file read")

    monkeypatch.setattr(render_report, "metric_rows", unexpected_read)
    assert render_report.summarize_resources(path, rows=[])["rss_mb"]["avg"] is None
    assert render_report.summarize_qps_resources(path, rows=[]) == {}
    assert render_report.summarize_drops(path, 100, rows=[])["drop_rate"] is None


@pytest.mark.parametrize("compressed", [False, True])
def test_batched_recovery_preserves_each_fields_valid_samples(
    tmp_path: Path, compressed: bool
) -> None:
    run = tmp_path / "run-result.json"
    metrics = tmp_path / "measurement" / ("metrics.csv.gz" if compressed else "metrics.csv")
    metrics.parent.mkdir()
    content = (
        "timestamp,rss_mb,channel_length,connection_cache_bytes\n"
        "100,50,nan,1024\n101,,2,inf\nbad,99,99,99\n102,60,0,512\n"
    )
    if compressed:
        with gzip.open(metrics, "wt") as handle:
            handle.write(content)
    else:
        metrics.write_text(content)
    fields = ("rss_mb", "channel_length", "connection_cache_bytes")
    snapshot = campaign_evidence.resource_series(run, fields)
    assert snapshot == {
        "rss_mb": [(100, 50), (102, 60)],
        "channel_length": [(101, 2), (102, 0)],
        "connection_cache_bytes": [(100, 1024), (102, 512)],
    }
    assert all(
        campaign_evidence.resource_samples(run, field) == snapshot[field] for field in fields
    )
