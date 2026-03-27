from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import yaml

from src.preprocessing.clahe_tuning import (
    persist_clahe_defaults,
    persist_tuning_record,
    tune_clahe_parameters,
)


def _write_synthetic_tiff(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base_gradient = np.tile(np.linspace(9000, 13000, 128, dtype=np.float32), (128, 1))
    noise = rng.normal(0, 120, size=(128, 128)).astype(np.float32)
    particle = np.zeros((128, 128), dtype=np.float32)
    particle[40:88, 45:83] = 600
    image = np.clip(base_gradient + noise + particle, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    tifffile.imwrite(path, image)


def test_tune_clahe_parameters_evaluates_grid_and_selects_candidate(tmp_path: Path) -> None:
    image_paths: list[Path] = []
    for idx in range(6):
        image_path = tmp_path / f"img_{idx:04d}.tif"
        _write_synthetic_tiff(image_path, seed=idx)
        image_paths.append(image_path)

    clip_limits = [1.5, 2.0, 2.5, 3.0]
    tile_grid_sizes = [(6, 6), (8, 8), (10, 10)]

    result = tune_clahe_parameters(
        image_paths=image_paths,
        clip_limits=clip_limits,
        tile_grid_sizes=tile_grid_sizes,
        representative_count=4,
    )

    assert len(result["evaluated_combinations"]) == 12
    assert result["selected_parameters"]["clip_limit"] in clip_limits
    assert tuple(result["selected_parameters"]["tile_grid_size"]) in tile_grid_sizes
    assert len(result["representative_images"]) == 4
    assert result["selected_parameters"]["robust_score"] == max(
        combo["robust_score"] for combo in result["evaluated_combinations"]
    )


def test_persist_clahe_defaults_updates_config(tmp_path: Path) -> None:
    config_path = tmp_path / "params.yaml"
    config_path.write_text("pipeline:\n  stage: preprocessing\n", encoding="utf-8")

    persist_clahe_defaults(config_path=config_path, clip_limit=2.5, tile_grid_size=(8, 8))

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["pipeline"]["stage"] == "preprocessing"
    assert payload["preprocessing"]["clahe"]["clip_limit"] == 2.5
    assert payload["preprocessing"]["clahe"]["tile_grid_size"] == [8, 8]


def test_persist_tuning_record_contains_reproducible_evidence(tmp_path: Path) -> None:
    tuning_record = {
        "selected_parameters": {
            "clip_limit": 2.5,
            "tile_grid_size": [8, 8],
            "robust_score": 0.95,
        },
        "representative_images": ["img_RDBS_0050.tif", "img_RDBS_0300.tif"],
        "selection_rationale": "Highest robust score and low cross-image variability.",
        "evaluated_combinations": [{"clip_limit": 2.5, "tile_grid_size": [8, 8], "robust_score": 0.95}],
    }
    record_path = tmp_path / "clahe_tuning_record.yaml"

    persist_tuning_record(record_path=record_path, tuning_record=tuning_record)

    payload = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    assert payload["selected_parameters"]["clip_limit"] == 2.5
    assert payload["selected_parameters"]["tile_grid_size"] == [8, 8]
    assert "selection_rationale" in payload
    assert len(payload["evaluated_combinations"]) == 1
