from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.utils.error_codes import raise_prd_error


@dataclass(frozen=True)
class ModelLoadConfig:
    backend: str = "mobilesam"
    checkpoint_path: str | Path = ""
    model_type: str = "vit_t"
    device: str = "cpu"
    config_path: str | Path | None = None


@dataclass(frozen=True)
class MaskGeneratorConfig:
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0

    def to_generator_kwargs(self) -> dict[str, Any]:
        return {
            "points_per_side": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "crop_n_layers": self.crop_n_layers,
            "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor,
            "min_mask_region_area": self.min_mask_region_area,
        }


class MobileSamInferenceCore:
    def __init__(
        self,
        model: Any,
        mask_generator: Any,
        predictor: Any | None = None,
        cuda_api: Any | None = None,
        cleanup_hook: Callable[[], None] | None = None,
    ) -> None:
        self.model = model
        self.mask_generator = mask_generator
        self.predictor = predictor
        self.cuda_api = cuda_api
        self.cleanup_hook = cleanup_hook
        self.vram_telemetry_log: list[dict[str, Any]] = []

    @classmethod
    def from_config(
        cls,
        model_config: ModelLoadConfig,
        generator_config: MaskGeneratorConfig,
        model_loader: Callable[[ModelLoadConfig], Any] | None = None,
        generator_factory: Callable[..., Any] | None = None,
        predictor_factory: Callable[..., Any] | None = None,
        cuda_api: Any | None = None,
        cleanup_hook: Callable[[], None] | None = None,
    ) -> "MobileSamInferenceCore":
        loader = model_loader or default_model_loader

        model = loader(model_config)
        generator_kwargs = generator_config.to_generator_kwargs()

        if generator_factory is None:
            mask_generator = default_generator_factory(
                model,
                backend=model_config.backend,
                **generator_kwargs,
            )
        else:
            mask_generator = generator_factory(model, **generator_kwargs)

        predictor = None
        if predictor_factory is not None:
            predictor = predictor_factory(model, model_config.backend)

        return cls(
            model=model,
            mask_generator=mask_generator,
            predictor=predictor,
            cuda_api=cuda_api,
            cleanup_hook=cleanup_hook,
        )

    def generate_masks(self, image: np.ndarray) -> list[dict[str, Any]]:
        rgb_uint8 = _to_rgb_uint8(image)
        raw_masks = self.mask_generator.generate(rgb_uint8)

        normalized = [_normalize_mask_item(mask) for mask in raw_masks]
        normalized.sort(key=lambda item: item["area"], reverse=True)
        return normalized

    def segment_image(
        self,
        image: np.ndarray,
        vram_stress_limit_bytes: int | None = None,
    ) -> dict[str, Any]:
        telemetry: dict[str, Any] = {
            "cuda_available": self._cuda_available(),
            "before": self._read_cuda_stats(),
            "after_inference": {"allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0},
            "after_cleanup": {"allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0},
            "cleanup_invoked": False,
        }

        try:
            masks = self.generate_masks(image)
            telemetry["after_inference"] = self._read_cuda_stats()
        except RuntimeError as exc:
            if _is_oom_error(exc):
                telemetry["cleanup_invoked"] = self._cleanup_vram()
                telemetry["after_cleanup"] = self._read_cuda_stats()
                self.vram_telemetry_log.append(telemetry)
                raise_prd_error("ERR_VRAM_001", stage="segmentation", context={"reason": "oom"})
            raise

        telemetry["cleanup_invoked"] = self._cleanup_vram()
        telemetry["after_cleanup"] = self._read_cuda_stats()

        if vram_stress_limit_bytes is not None:
            peak_reserved = int(telemetry["after_inference"]["reserved_bytes"])
            peak_allocated = int(telemetry["after_inference"]["allocated_bytes"])
            peak_max_allocated = int(telemetry["after_inference"]["max_allocated_bytes"])
            if (
                peak_reserved > vram_stress_limit_bytes
                or peak_allocated > vram_stress_limit_bytes
                or peak_max_allocated > vram_stress_limit_bytes
            ):
                self.vram_telemetry_log.append(telemetry)
                raise_prd_error("ERR_VRAM_001", stage="segmentation", context={"reason": "stress"})

        self.vram_telemetry_log.append(telemetry)
        return {"masks": masks, "telemetry": telemetry}

    def refine_masks(
        self,
        image: np.ndarray,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: tuple[float, float, float, float] | np.ndarray | None = None,
        multimask_output: bool = False,
    ) -> list[dict[str, Any]]:
        if self.predictor is None:
            raise ValueError("Manual refinement predictor is not configured")

        rgb_uint8 = _to_rgb_uint8(image)
        self.predictor.set_image(rgb_uint8)

        predict_kwargs: dict[str, Any] = {"multimask_output": multimask_output}
        if point_coords is not None:
            predict_kwargs["point_coords"] = point_coords
        if point_labels is not None:
            predict_kwargs["point_labels"] = point_labels
        if box is not None:
            predict_kwargs["box"] = np.asarray(box, dtype=np.float32)

        masks, scores, _ = self.predictor.predict(**predict_kwargs)

        refined: list[dict[str, Any]] = []
        for idx in range(len(masks)):
            item = _normalize_mask_item({"segmentation": masks[idx]})
            item["score"] = float(scores[idx])
            refined.append(item)

        refined.sort(key=lambda item: item["score"], reverse=True)
        return refined

    def _cleanup_vram(self) -> bool:
        invoked = False
        if self.cleanup_hook is not None:
            self.cleanup_hook()
            invoked = True

        gc.collect()
        cuda = self.cuda_api
        if cuda is None:
            try:
                import torch

                cuda = torch.cuda
            except ModuleNotFoundError:
                cuda = None

        if cuda is not None and hasattr(cuda, "empty_cache"):
            if self._cuda_available(cuda):
                cuda.empty_cache()
                invoked = True

        return invoked

    def _read_cuda_stats(self) -> dict[str, int]:
        cuda = self.cuda_api
        if cuda is None:
            try:
                import torch

                cuda = torch.cuda
            except ModuleNotFoundError:
                cuda = None

        if cuda is None or not self._cuda_available(cuda):
            return {"allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0}

        return {
            "allocated_bytes": int(cuda.memory_allocated()),
            "reserved_bytes": int(cuda.memory_reserved()),
            "max_allocated_bytes": int(cuda.max_memory_allocated()),
        }

    def _cuda_available(self, cuda: Any | None = None) -> bool:
        api = cuda or self.cuda_api
        if api is None:
            try:
                import torch

                api = torch.cuda
            except ModuleNotFoundError:
                return False

        checker = getattr(api, "is_available", None)
        if callable(checker):
            return bool(checker())
        return False


def default_model_loader(config: ModelLoadConfig) -> Any:
    backend = config.backend.lower()

    if backend == "mobilesam":
        mobile_sam = importlib.import_module("mobile_sam")
        registry = getattr(mobile_sam, "sam_model_registry")
        model = registry[config.model_type](checkpoint=str(config.checkpoint_path))
        if hasattr(model, "to"):
            model = model.to(config.device)
        if hasattr(model, "eval"):
            model = model.eval()
        return model

    if backend == "sam2":
        build_sam_module = importlib.import_module("sam2.build_sam")
        build_sam2 = getattr(build_sam_module, "build_sam2")
        if config.config_path is None:
            raise ValueError("config_path is required for sam2 backend")
        model = build_sam2(str(config.config_path), str(config.checkpoint_path), device=config.device)
        if hasattr(model, "eval"):
            model = model.eval()
        return model

    raise ValueError(f"Unsupported backend: {config.backend}")


def default_generator_factory(model: Any, backend: str, **kwargs: Any) -> Any:
    normalized_backend = backend.lower()

    if normalized_backend == "mobilesam":
        mobile_sam = importlib.import_module("mobile_sam")
        generator_cls = getattr(mobile_sam, "SamAutomaticMaskGenerator")
        return generator_cls(model, **kwargs)

    if normalized_backend == "sam2":
        try:
            sam2_module = importlib.import_module("sam2.automatic_mask_generator")
        except ModuleNotFoundError:
            sam2_module = importlib.import_module("sam2.sam2_automatic_mask_generator")
        generator_cls = getattr(sam2_module, "SAM2AutomaticMaskGenerator")
        return generator_cls(model, **kwargs)

    raise ValueError(f"Unsupported backend: {backend}")


def default_predictor_factory(model: Any, backend: str) -> Any:
    normalized_backend = backend.lower()

    if normalized_backend == "mobilesam":
        mobile_sam = importlib.import_module("mobile_sam")
        predictor_cls = getattr(mobile_sam, "SamPredictor")
        return predictor_cls(model)

    if normalized_backend == "sam2":
        predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        predictor_cls = getattr(predictor_module, "SAM2ImagePredictor")
        return predictor_cls(model)

    raise ValueError(f"Unsupported backend: {backend}")


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.integer):
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            raise ValueError("Unsupported image dtype for inference")

    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)

    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)

    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr

    raise ValueError("Expected image shape (H, W), (H, W, 1), or (H, W, 3)")


def _normalize_mask_item(mask: dict[str, Any]) -> dict[str, Any]:
    if "segmentation" not in mask:
        raise ValueError("Mask item missing 'segmentation'")

    segmentation = np.asarray(mask["segmentation"], dtype=bool)
    area = int(mask.get("area", int(segmentation.sum())))

    raw_bbox = mask.get("bbox")
    if raw_bbox is None:
        bbox = _bbox_from_segmentation(segmentation)
    else:
        if len(raw_bbox) != 4:
            raise ValueError("bbox must contain exactly 4 values")
        bbox_values = [int(round(float(v))) for v in raw_bbox]
        bbox = (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3])

    return {
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
    }


def _bbox_from_segmentation(segmentation: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(segmentation)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)

    x_min = int(xs.min())
    y_min = int(ys.min())
    width = int(xs.max() - x_min + 1)
    height = int(ys.max() - y_min + 1)
    return (x_min, y_min, width, height)


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


__all__ = [
    "MaskGeneratorConfig",
    "MobileSamInferenceCore",
    "ModelLoadConfig",
    "default_generator_factory",
    "default_model_loader",
    "default_predictor_factory",
]


