from __future__ import annotations

import json
from pathlib import Path


def _create_tiff_placeholders(input_dir: Path, count: int) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for idx in range(count):
        path = input_dir / f"img_RDBS_{idx:04d}.tif"
        path.write_bytes(b"placeholder")
        created.append(path)
    return created


def test_main_cli_smoke_non_interactive_six_image_batch(monkeypatch, tmp_path: Path, capsys) -> None:
    import src.main as cli

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _create_tiff_placeholders(input_dir, count=6)

    run_calls: list[str] = []

    class _TqdmCapture:
        def __init__(self, iterable, **kwargs) -> None:
            self.items = list(iterable)
            self.kwargs = kwargs

        def __iter__(self):
            return iter(self.items)

    def _fake_run_batch(input_dir, output_dir, config):  # noqa: ANN001, ANN201
        image_path = next(iter(Path(input_dir).glob("*.tif")))
        image_id = image_path.stem
        run_calls.append(image_id)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "analisis_gold_standard.csv"
        csv_path.write_text("image_id,particle_id\n", encoding="utf-8")
        manifest_path = out_dir / "heatmap_manifest.json"
        manifest_path.write_text("[]", encoding="utf-8")
        return {
            "status": "completed",
            "images_total": 1,
            "images_succeeded": 1,
            "images_failed": 0,
            "images": [
                {
                    "image_id": image_id,
                    "status": "success",
                    "state": "exported",
                    "state_transitions": [],
                    "timings": {
                        "preprocessed": 0.1,
                        "segmented": 0.1,
                        "refined": 0.1,
                        "analyzed": 0.1,
                        "exported": 0.1,
                        "total_seconds": 0.5,
                    },
                    "error": None,
                }
            ],
            "exports": {
                "csv_path": str(csv_path),
                "heatmap_manifest_path": str(manifest_path),
            },
            "timings": {"total_seconds": 0.5},
        }

    monkeypatch.setattr(cli, "_build_segmentation_core", lambda _args: object())
    monkeypatch.setattr(cli, "run_batch", _fake_run_batch)
    monkeypatch.setattr(cli, "tqdm", _TqdmCapture)

    exit_code = cli.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--checkpoint-path",
            str(tmp_path / "fake.pt"),
        ]
    )

    assert exit_code == 0
    assert len(run_calls) == 6

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["images_total"] == 6
    assert payload["images_succeeded"] == 6
    assert payload["images_failed"] == 0
    assert payload["status"] == "completed"
    assert payload["continue_on_error"] is True


def test_main_cli_resilient_mode_continue_on_error_logs_err_events(monkeypatch, tmp_path: Path, capsys) -> None:
    import src.main as cli

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _create_tiff_placeholders(input_dir, count=3)
    log_path = tmp_path / "runtime.jsonl"

    run_calls: list[str] = []

    def _fake_run_batch(input_dir, output_dir, config):  # noqa: ANN001, ANN201
        image_path = next(iter(Path(input_dir).glob("*.tif")))
        image_id = image_path.stem
        run_calls.append(image_id)
        failed = image_id.endswith("0001")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "analisis_gold_standard.csv"
        csv_path.write_text("image_id,particle_id\n", encoding="utf-8")
        manifest_path = out_dir / "heatmap_manifest.json"
        manifest_path.write_text("[]", encoding="utf-8")
        return {
            "status": "completed_with_errors" if failed else "completed",
            "images_total": 1,
            "images_succeeded": 0 if failed else 1,
            "images_failed": 1 if failed else 0,
            "images": [
                {
                    "image_id": image_id,
                    "status": "failed" if failed else "success",
                    "state": "analyzed" if failed else "exported",
                    "state_transitions": [],
                    "timings": {
                        "preprocessed": 0.1,
                        "segmented": 0.1,
                        "refined": 0.1,
                        "analyzed": 0.1,
                        "exported": 0.1 if not failed else 0.0,
                        "total_seconds": 0.4,
                    },
                    "error": (
                        {
                            "code": "ERR_MASK_002",
                            "message": "No valid particles detected after filtering. Check thresholds or image quality.",
                            "stage": "validation",
                            "context": {"image_id": image_id},
                        }
                        if failed
                        else None
                    ),
                }
            ],
            "exports": {
                "csv_path": str(csv_path),
                "heatmap_manifest_path": str(manifest_path),
            },
            "timings": {"total_seconds": 0.4},
        }

    monkeypatch.setattr(cli, "_build_segmentation_core", lambda _args: object())
    monkeypatch.setattr(cli, "run_batch", _fake_run_batch)

    exit_code = cli.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--checkpoint-path",
            str(tmp_path / "fake.pt"),
            "--continue-on-error",
            "--log-file",
            str(log_path),
            "--no-progress",
        ]
    )

    assert exit_code == 1
    assert len(run_calls) == 3

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["images_total"] == 3
    assert payload["images_failed"] == 1
    assert payload["status"] == "completed_with_errors"

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    err_events = [event for event in events if event.get("event") == "ERR_EVENT"]
    image_events = [event for event in events if event.get("event") == "IMAGE_COMPLETED"]
    assert len(err_events) == 1
    assert err_events[0]["error"]["code"] == "ERR_MASK_002"
    assert err_events[0]["error"]["stage"] == "validation"
    assert any("segmented" in event["stage_timings"] for event in image_events)


def test_main_cli_fail_fast_stops_after_first_error(monkeypatch, tmp_path: Path, capsys) -> None:
    import src.main as cli

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _create_tiff_placeholders(input_dir, count=3)

    run_calls: list[str] = []

    def _fake_run_batch(input_dir, output_dir, config):  # noqa: ANN001, ANN201
        image_path = next(iter(Path(input_dir).glob("*.tif")))
        image_id = image_path.stem
        run_calls.append(image_id)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "analisis_gold_standard.csv"
        csv_path.write_text("image_id,particle_id\n", encoding="utf-8")
        manifest_path = out_dir / "heatmap_manifest.json"
        manifest_path.write_text("[]", encoding="utf-8")
        return {
            "status": "completed_with_errors",
            "images_total": 1,
            "images_succeeded": 0,
            "images_failed": 1,
            "images": [
                {
                    "image_id": image_id,
                    "status": "failed",
                    "state": "segmented",
                    "state_transitions": [],
                    "timings": {"segmented": 0.1, "total_seconds": 0.1},
                    "error": {
                        "code": "ERR_VRAM_001",
                        "message": "GPU memory exceeded during segmentation. Reduce image resolution or tile size.",
                        "stage": "segmentation",
                        "context": {"reason": "stress"},
                    },
                }
            ],
            "exports": {
                "csv_path": str(csv_path),
                "heatmap_manifest_path": str(manifest_path),
            },
            "timings": {"total_seconds": 0.1},
        }

    monkeypatch.setattr(cli, "_build_segmentation_core", lambda _args: object())
    monkeypatch.setattr(cli, "run_batch", _fake_run_batch)

    exit_code = cli.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--checkpoint-path",
            str(tmp_path / "fake.pt"),
            "--fail-fast",
            "--no-progress",
        ]
    )

    assert exit_code == 1
    assert run_calls == ["img_RDBS_0000"]

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["images_total"] == 1
    assert payload["images_failed"] == 1
    assert payload["continue_on_error"] is False
