from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import yaml

from src.postprocessing.morphology_tuning import (
    persist_morphology_defaults,
    persist_tuning_record,
    tune_morphology_parameters,
)


def _write_synthetic_tiff(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    canvas = np.zeros((96, 96), dtype=np.float32)
    canvas[20:68, 18:44] = 700
    canvas[20:68, 50:78] = 700
    canvas[38:50, 44:50] = 700
    noise = rng.normal(0, 40, size=canvas.shape).astype(np.float32)
    image = np.clip(8500 + canvas + noise, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    tifffile.imwrite(path, image)


def test_tune_morphology_parameters_evaluates_grid_and_selects_candidate(tmp_path: Path) -> None:
    image_paths: list[Path] = []
    for idx in range(6):
        image_path = tmp_path / f"img_RDBS_{idx:04d}.tif"
        _write_synthetic_tiff(image_path, seed=idx)
        image_paths.append(image_path)

    result = tune_morphology_parameters(
        image_paths=image_paths,
        kernel_shapes=["ellipse", "rect"],
        kernel_sizes=[3, 5, 7],
        representative_count=4,
    )

    assert len(result["evaluated_variants"]) == 6
    assert len(result["representative_images"]) == 4
    assert result["selected_parameters"]["kernel_shape"] in {"ellipse", "rect"}
    assert result["selected_parameters"]["kernel_size"] in {3, 5, 7}
    assert result["selected_parameters"]["quality_score"] == max(
        variant["quality_score"] for variant in result["evaluated_variants"]
    )


def test_persist_morphology_defaults_updates_config(tmp_path: Path) -> None:
    config_path = tmp_path / "params.yaml"
    config_path.write_text("preprocessing:\n  clahe:\n    clip_limit: 3.0\n", encoding="utf-8")

    persist_morphology_defaults(config_path=config_path, kernel_shape="ellipse", kernel_size=5)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["preprocessing"]["clahe"]["clip_limit"] == 3.0
    assert payload["postprocessing"]["morphology"]["kernel_shape"] == "ellipse"
    assert payload["postprocessing"]["morphology"]["kernel_size"] == 5


def test_persist_tuning_record_contains_reproducible_evidence(tmp_path: Path) -> None:
    tuning_record = {
        "selected_parameters": {
            "kernel_shape": "ellipse",
            "kernel_size": 5,
            "quality_score": 0.81,
            "separation_score": 0.86,
            "erosion_penalty": 0.17,
        },
        "representative_images": ["img_RDBS_0050.tif", "img_RDBS_0300.tif"],
        "selection_rationale": "Highest quality score balancing separation and erosion.",
        "evaluated_variants": [
            {
                "kernel_shape": "ellipse",
                "kernel_size": 5,
                "quality_score": 0.81,
                "separation_score": 0.86,
                "erosion_penalty": 0.17,
            }
        ],
    }
    record_path = tmp_path / "morphology_tuning_record.yaml"

    persist_tuning_record(record_path=record_path, tuning_record=tuning_record)

    payload = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    assert payload["selected_parameters"]["kernel_shape"] == "ellipse"
    assert payload["selected_parameters"]["kernel_size"] == 5
    assert payload["selection_rationale"]
    assert len(payload["evaluated_variants"]) == 1
