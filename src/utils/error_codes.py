from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ERROR_CATALOG: dict[str, str] = {
    "ERR_VRAM_001": "GPU memory exceeded during segmentation. Reduce image resolution or tile size.",
    "ERR_MASK_002": "No valid particles detected after filtering. Check thresholds or image quality.",
    "ERR_IO_003": "Failed to read 16-bit TIFF file. Verify file integrity and tifffile library version.",
    "ERR_METRIC_004": "Circularity calculation failed: perimeter is zero. Check contour extraction.",
}


@dataclass(eq=True)
class PrdError(Exception):
    code: str
    message: str
    stage: str | None = None
    context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.context = dict(self.context or {})
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "context": dict(self.context or {}),
        }


def get_error(code: str) -> dict[str, str]:
    return {"code": code, "message": ERROR_CATALOG[code]}


def build_error_payload(
    code: str,
    stage: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error = get_error(code)
    return {
        "code": error["code"],
        "message": error["message"],
        "stage": stage,
        "context": dict(context or {}),
    }


def raise_prd_error(
    code: str,
    stage: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    payload = build_error_payload(code=code, stage=stage, context=context)
    raise PrdError(
        code=payload["code"],
        message=payload["message"],
        stage=payload["stage"],
        context=payload["context"],
    )


def serialize_error(error: Exception) -> dict[str, Any]:
    if isinstance(error, PrdError):
        return error.to_dict()

    return {
        "code": "ERR_UNKNOWN",
        "message": str(error),
        "stage": None,
        "context": {},
    }


def log_prd_error(
    logger: Any,
    code: str,
    stage: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_error_payload(code=code, stage=stage, context=context)
    logger.error(f"{payload['code']}: {payload['message']}", extra={"error": payload})
    return payload


__all__ = [
    "ERROR_CATALOG",
    "PrdError",
    "build_error_payload",
    "get_error",
    "log_prd_error",
    "raise_prd_error",
    "serialize_error",
]
