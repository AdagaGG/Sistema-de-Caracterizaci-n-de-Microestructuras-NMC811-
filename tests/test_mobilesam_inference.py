from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.mobilesam_inference import (
    MaskGeneratorConfig,
    MobileSamInferenceCore,
    ModelLoadConfig,
)
from src.utils.error_codes import ERROR_CATALOG, PrdError


class _DummyModel:
    def __init__(self) -> None:
        self.is_eval = False
        self.device = "cpu"

    def eval(self) -> "_DummyModel":
        self.is_eval = True
        return self

    def to(self, device: str) -> "_DummyModel":
        self.device = device
        return self


class _DummyGenerator:
    def __init__(self, model: _DummyModel, **kwargs: object) -> None:
        self.model = model
        self.kwargs = kwargs
        self.last_image: np.ndarray | None = None

    def generate(self, image: np.ndarray) -> list[dict[str, object]]:
        self.last_image = image
        return [
            {
                "segmentation": np.array(
                    [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
                    dtype=np.uint8,
                ),
                "predicted_iou": 0.95,
            },
            {
                "segmentation": np.array(
                    [[1, 1, 1], [1, 1, 1], [0, 0, 1]],
                    dtype=bool,
                ),
                "area": 7,
                "bbox": (0.2, 0.8, 2.7, 2.9),
            },
        ]


class _FailingGenerator:
    def generate(self, image: np.ndarray) -> list[dict[str, object]]:
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.23 GiB")


class _DummyCudaApi:
    def __init__(self, snapshots: list[tuple[int, int, int]]) -> None:
        self._snapshots = list(snapshots)
        self._last = snapshots[-1] if snapshots else (0, 0, 0)
        self._cursor = 0

    def is_available(self) -> bool:
        return True

    def memory_allocated(self) -> int:
        if self._snapshots:
            self._last = self._snapshots[0]
        return self._last[0]

    def memory_reserved(self) -> int:
        return self._last[1]

    def max_memory_allocated(self) -> int:
        value = self._last[2]
        if self._snapshots:
            self._snapshots.pop(0)
        return value

    def empty_cache(self) -> None:
        return None


class _DummyPredictor:
    def __init__(self) -> None:
        self.last_set_image: np.ndarray | None = None
        self.last_predict_args: dict[str, object] = {}

    def set_image(self, image: np.ndarray) -> None:
        self.last_set_image = image

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.last_predict_args = kwargs
        masks = np.array(
            [
                [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [1, 1, 1], [0, 0, 1]],
            ],
            dtype=bool,
        )
        scores = np.array([0.42, 0.99], dtype=float)
        logits = np.zeros((2, 3, 3), dtype=float)
        return masks, scores, logits


def test_model_loading_and_generator_parameters_are_configurable() -> None:
    model_cfg = ModelLoadConfig(
        backend="mobilesam",
        checkpoint_path="weights/mobile_sam.pt",
        model_type="vit_t",
        device="cuda:0",
    )
    mask_cfg = MaskGeneratorConfig(
        points_per_side=24,
        pred_iou_thresh=0.91,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=80,
    )

    captured: dict[str, object] = {}

    def _loader(config: ModelLoadConfig) -> _DummyModel:
        captured["model_cfg"] = config
        return _DummyModel().to(config.device).eval()

    def _generator_factory(model: _DummyModel, **kwargs: object) -> _DummyGenerator:
        captured["generator_kwargs"] = kwargs
        return _DummyGenerator(model=model, **kwargs)

    core = MobileSamInferenceCore.from_config(
        model_config=model_cfg,
        generator_config=mask_cfg,
        model_loader=_loader,
        generator_factory=_generator_factory,
    )

    assert isinstance(core.model, _DummyModel)
    assert core.model.is_eval is True
    assert core.model.device == "cuda:0"
    assert captured["model_cfg"] == model_cfg
    assert captured["generator_kwargs"] == {
        "points_per_side": 24,
        "pred_iou_thresh": 0.91,
        "stability_score_thresh": 0.92,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 80,
    }


def test_generate_masks_enforces_stable_schema_invariants() -> None:
    generator = _DummyGenerator(model=_DummyModel())
    core = MobileSamInferenceCore(model=_DummyModel(), mask_generator=generator)

    grayscale = np.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        dtype=np.uint8,
    )

    masks = core.generate_masks(grayscale)

    assert generator.last_image is not None
    assert generator.last_image.dtype == np.uint8
    assert generator.last_image.shape == (3, 3, 3)

    assert len(masks) == 2
    assert [mask["area"] for mask in masks] == [7, 3]

    for mask in masks:
        assert set(mask.keys()) == {"segmentation", "area", "bbox"}
        assert mask["segmentation"].dtype == np.bool_
        assert isinstance(mask["area"], int)
        assert isinstance(mask["bbox"], tuple)
        assert len(mask["bbox"]) == 4
        assert all(isinstance(v, int) for v in mask["bbox"])


def test_segment_image_emits_vram_telemetry_and_cleanup_hook_per_image() -> None:
    cleanup_calls: list[str] = []

    def _cleanup_hook() -> None:
        cleanup_calls.append("cleanup")

    core = MobileSamInferenceCore(
        model=_DummyModel(),
        mask_generator=_DummyGenerator(model=_DummyModel()),
        cuda_api=_DummyCudaApi(
            [
                (12, 34, 56),  # before
                (20, 60, 80),  # after inference
                (4, 10, 80),  # after cleanup
            ]
        ),
        cleanup_hook=_cleanup_hook,
    )

    result = core.segment_image(np.zeros((3, 3), dtype=np.uint8))

    assert len(result["masks"]) == 2
    assert cleanup_calls == ["cleanup"]
    assert result["telemetry"] == {
        "cuda_available": True,
        "before": {
            "allocated_bytes": 12,
            "reserved_bytes": 34,
            "max_allocated_bytes": 56,
        },
        "after_inference": {
            "allocated_bytes": 20,
            "reserved_bytes": 60,
            "max_allocated_bytes": 80,
        },
        "after_cleanup": {
            "allocated_bytes": 4,
            "reserved_bytes": 10,
            "max_allocated_bytes": 80,
        },
        "cleanup_invoked": True,
    }
    assert core.vram_telemetry_log == [result["telemetry"]]


def test_segment_image_oom_raises_err_vram_001_exact_message() -> None:
    core = MobileSamInferenceCore(model=_DummyModel(), mask_generator=_FailingGenerator())

    with pytest.raises(PrdError) as exc_info:
        core.segment_image(np.zeros((3, 3), dtype=np.uint8))

    err = exc_info.value
    assert err.code == "ERR_VRAM_001"
    assert err.message == ERROR_CATALOG["ERR_VRAM_001"]


def test_segment_image_stress_threshold_raises_err_vram_001() -> None:
    core = MobileSamInferenceCore(
        model=_DummyModel(),
        mask_generator=_DummyGenerator(model=_DummyModel()),
        cuda_api=_DummyCudaApi(
            [
                (10, 30, 40),  # before
                (20, 120, 140),  # after inference (stress)
                (5, 20, 140),  # after cleanup
            ]
        ),
    )

    with pytest.raises(PrdError) as exc_info:
        core.segment_image(np.zeros((3, 3), dtype=np.uint8), vram_stress_limit_bytes=100)

    err = exc_info.value
    assert err.code == "ERR_VRAM_001"
    assert err.message == ERROR_CATALOG["ERR_VRAM_001"]


def test_refine_masks_supports_manual_point_and_box_prompts() -> None:
    predictor = _DummyPredictor()
    core = MobileSamInferenceCore(
        model=_DummyModel(),
        mask_generator=_DummyGenerator(model=_DummyModel()),
        predictor=predictor,
    )
    image = np.zeros((3, 3), dtype=np.uint8)
    point_coords = np.array([[1, 1], [2, 1]], dtype=np.float32)
    point_labels = np.array([1, 0], dtype=np.int32)
    box = (0, 0, 2, 2)

    refined = core.refine_masks(
        image=image,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    assert predictor.last_set_image is not None
    assert predictor.last_set_image.shape == (3, 3, 3)
    assert predictor.last_predict_args["point_coords"] is point_coords
    assert predictor.last_predict_args["point_labels"] is point_labels
    assert np.asarray(predictor.last_predict_args["box"]).tolist() == [0, 0, 2, 2]
    assert predictor.last_predict_args["multimask_output"] is True
    assert len(refined) == 2
    assert refined[0]["score"] == pytest.approx(0.99)
    assert refined[1]["score"] == pytest.approx(0.42)

