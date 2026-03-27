from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

from src.pipeline import run_batch
from src.segmentation import MaskGeneratorConfig, MobileSamInferenceCore, ModelLoadConfig
from src.utils.error_codes import ERROR_CATALOG, serialize_error


def _parse_tile_grid_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("tile-grid-size must be provided as WIDTH,HEIGHT")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("tile-grid-size values must be integers") from error


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nmc811-orchestrator",
        description="Runtime CLI wrapper for NMC811 orchestration engine.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing .tif/.tiff images.")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs and logs are written.")
    parser.add_argument("--checkpoint-path", required=True, help="MobileSAM/SAM2 checkpoint path.")
    parser.add_argument("--sam2-config-path", default=None, help="SAM2 config path (required when --backend sam2).")
    parser.add_argument("--backend", choices=("mobilesam", "sam2"), default="mobilesam")
    parser.add_argument("--model-type", default="vit_t")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid-size", type=_parse_tile_grid_size, default=(8, 8))
    parser.add_argument("--vram-stress-limit-bytes", type=int, default=None)
    parser.add_argument("--start-at", default=None, help="Image stem or file name to resume from.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for checkpoint-friendly runs.")
    parser.add_argument("--log-file", default=None, help="Structured runtime JSONL log file path.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    mode.add_argument("--fail-fast", dest="continue_on_error", action="store_false")
    parser.set_defaults(continue_on_error=True)

    return parser


def _list_input_images(input_dir: Path) -> list[Path]:
    images = [*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")]
    return sorted(images, key=lambda item: item.name.lower())


def _apply_resume_window(images: list[Path], start_at: str | None, max_images: int | None) -> list[Path]:
    filtered = images
    if start_at:
        start_key = start_at.lower()
        start_idx = next(
            (
                idx
                for idx, image in enumerate(images)
                if image.stem.lower() == start_key or image.name.lower() == start_key
            ),
            None,
        )
        if start_idx is None:
            return []
        filtered = images[start_idx:]

    if max_images is not None and max_images >= 0:
        filtered = filtered[:max_images]
    return filtered


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"nmc811_runtime_{log_path}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def _log_event(logger: logging.Logger, event: str, **payload: Any) -> None:
    record = {"event": event, "timestamp": time.time(), **payload}
    logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True))


def _build_segmentation_core(args: argparse.Namespace) -> MobileSamInferenceCore:
    model_cfg = ModelLoadConfig(
        backend=args.backend,
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        device=args.device,
        config_path=args.sam2_config_path,
    )
    generator_cfg = MaskGeneratorConfig(
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )
    return MobileSamInferenceCore.from_config(
        model_config=model_cfg,
        generator_config=generator_cfg,
    )


def _single_image_batch_summary(image_id: str, error: Exception) -> dict[str, Any]:
    return {
        "status": "completed_with_errors",
        "images_total": 1,
        "images_succeeded": 0,
        "images_failed": 1,
        "images": [
            {
                "image_id": image_id,
                "status": "failed",
                "state": "raw",
                "state_transitions": [],
                "timings": {"total_seconds": 0.0},
                "error": serialize_error(error),
            }
        ],
        "exports": {"csv_path": None, "heatmap_manifest_path": None},
        "timings": {"total_seconds": 0.0},
    }


def main(argv: list[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else output_root / "runtime_execution.jsonl"
    logger = _setup_logger(log_path)

    started_at = time.perf_counter()
    input_root = Path(args.input_dir)
    images = _list_input_images(input_root)
    selected_images = _apply_resume_window(images, args.start_at, args.max_images)

    if not selected_images:
        result = {
            "status": "completed_with_errors",
            "input_dir": str(input_root),
            "output_dir": str(output_root),
            "images_total": 0,
            "images_succeeded": 0,
            "images_failed": 0,
            "images": [],
            "exports": {"csv_path": None, "heatmap_manifest_path": None},
            "continue_on_error": bool(args.continue_on_error),
            "timings": {"total_seconds": time.perf_counter() - started_at},
            "runtime_log": str(log_path),
        }
        print(json.dumps(result, ensure_ascii=False))
        return 0

    _log_event(
        logger,
        "BATCH_STARTED",
        input_dir=str(input_root),
        output_dir=str(output_root),
        images_total=len(selected_images),
        continue_on_error=bool(args.continue_on_error),
    )

    segmentation_core = _build_segmentation_core(args)
    run_config: dict[str, Any] = {
        "dependencies": {"segmentation_core": segmentation_core},
        "preprocessing": {
            "clip_limit": float(args.clip_limit),
            "tile_grid_size": tuple(args.tile_grid_size),
        },
    }
    if args.vram_stress_limit_bytes is not None:
        run_config["vram_stress_limit_bytes"] = int(args.vram_stress_limit_bytes)

    images_payload: list[dict[str, Any]] = []
    export_csv_path: str | None = None
    export_manifest_path: str | None = None

    iterator = tqdm(
        selected_images,
        total=len(selected_images),
        unit="image",
        disable=bool(args.no_progress),
        desc="Processing",
    )

    for image_path in iterator:
        image_started_at = time.perf_counter()
        image_output_dir = output_root / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="nmc811-single-") as temp_input:
            temp_input_dir = Path(temp_input)
            shutil.copy2(image_path, temp_input_dir / image_path.name)

            try:
                per_image_summary = run_batch(
                    input_dir=temp_input_dir,
                    output_dir=image_output_dir,
                    config=run_config,
                )
            except Exception as error:  # noqa: BLE001
                per_image_summary = _single_image_batch_summary(image_path.stem, error)

        image_summary = per_image_summary["images"][0]
        image_summary = dict(image_summary)
        image_summary["image_id"] = image_path.stem
        image_summary.setdefault("timings", {})
        image_summary["timings"]["runtime_wrapper_seconds"] = time.perf_counter() - image_started_at
        images_payload.append(image_summary)

        exports = per_image_summary.get("exports", {})
        if exports.get("csv_path"):
            export_csv_path = str(exports["csv_path"])
        if exports.get("heatmap_manifest_path"):
            export_manifest_path = str(exports["heatmap_manifest_path"])

        stage_timings = dict(image_summary.get("timings", {}))
        _log_event(
            logger,
            "IMAGE_COMPLETED",
            image_id=image_summary["image_id"],
            status=image_summary.get("status"),
            state=image_summary.get("state"),
            stage_timings=stage_timings,
        )

        error_payload = image_summary.get("error")
        if error_payload:
            error_code = str(error_payload.get("code", "ERR_UNKNOWN"))
            if not error_code.startswith("ERR_"):
                error_code = "ERR_UNKNOWN"
                error_payload = dict(error_payload)
                error_payload["code"] = error_code
                error_payload["message"] = ERROR_CATALOG.get(error_code, str(error_payload.get("message", "unknown")))
            _log_event(
                logger,
                "ERR_EVENT",
                image_id=image_summary["image_id"],
                error=error_payload,
            )
            if not args.continue_on_error:
                break

    images_succeeded = sum(1 for item in images_payload if item.get("status") == "success")
    images_failed = sum(1 for item in images_payload if item.get("status") == "failed")
    status = "completed" if images_failed == 0 else "completed_with_errors"

    timings_payload = {"total_seconds": time.perf_counter() - started_at}

    final_result = {
        "status": status,
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "images_total": len(images_payload),
        "images_succeeded": images_succeeded,
        "images_failed": images_failed,
        "images": images_payload,
        "exports": {
            "csv_path": export_csv_path,
            "heatmap_manifest_path": export_manifest_path,
        },
        "continue_on_error": bool(args.continue_on_error),
        "timings": timings_payload,
        "runtime_log": str(log_path),
    }

    _log_event(
        logger,
        "BATCH_COMPLETED",
        status=final_result["status"],
        images_total=final_result["images_total"],
        images_succeeded=final_result["images_succeeded"],
        images_failed=final_result["images_failed"],
        total_seconds=timings_payload["total_seconds"],
    )

    print(json.dumps(final_result, ensure_ascii=False))
    return 0 if images_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

