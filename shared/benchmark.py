"""
Fixed template benchmark layer — DO NOT REPLACE.

Teams customize `team_config.MODEL_PROFILE` and replace `solution/` only.
Streamlit + scripts always call this module for timing and footprint.
"""

from __future__ import annotations

import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

import team_config as cfg


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def _auto_product_head_mb() -> float:
    """Best-effort size estimate from baseline product head (if still present)."""
    try:
        from shared.data_utils import load_train_labels
        from solution.brand_rules import extract_product
        from solution.product_model import ProductPredictor

        labels = load_train_labels()
        if labels is None:
            return 0.0
        model = ProductPredictor(min_class_count=3, prob_threshold=0.60, max_features=3000)
        model.fit(labels, extract_product)
        return round(model.model_size_mb(), 2)
    except Exception:
        return 0.0


def get_model_profile() -> dict[str, Any]:
    """Model footprint shown in Streamlit and benchmark reports."""
    defaults: dict[str, Any] = {
        "pipeline": "EasyOCR (vi+en) + regex brands + sklearn product head",
        "runtime_device": "CPU",
        "product_head_mb": None,
        "ocr_backend_note": "EasyOCR weights ~200 MB (downloaded once, not in repo)",
        "lightweight_notes": (
            "Edit MODEL_PROFILE in team_config.py when you change models. "
            "Latency is always measured by shared/benchmark.py."
        ),
    }
    overrides = dict(getattr(cfg, "MODEL_PROFILE", {}) or {})
    profile = {**defaults, **overrides}

    if profile.get("product_head_mb") is None:
        profile["product_head_mb"] = _auto_product_head_mb()

    return profile


def run_predict_with_metrics(
    img: Image.Image,
    min_conf: float | None = None,
) -> dict[str, Any]:
    """
    Call team `predict_from_image` and always attach timing_ms (template-owned).
    """
    from solution.pipeline import predict_from_image

    conf = cfg.DEFAULT_MIN_CONF if min_conf is None else min_conf
    t0 = time.perf_counter()

    try:
        result = predict_from_image(img, conf, include_timing=True)
    except TypeError:
        result = predict_from_image(img, conf)

    wall_ms = (time.perf_counter() - t0) * 1000
    timing = dict(result.get("timing_ms") or {})

    if not timing:
        timing = {"ocr": round(wall_ms, 1), "extract": 0.0, "total": round(wall_ms, 1)}
    elif not timing.get("total"):
        timing["total"] = round(wall_ms, 1)

    result["timing_ms"] = {
        "ocr": round(float(timing.get("ocr", 0.0)), 1),
        "extract": round(float(timing.get("extract", 0.0)), 1),
        "total": round(float(timing.get("total", wall_ms)), 1),
    }
    return result


def run_benchmark(
    *,
    limit: int = 6,
    warmup: int = 1,
    image_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """
    Benchmark latency on sample private-test images.

    Used by CLI and Streamlit deploy smoke check.
    """
    from shared.data_utils import find_private_root, private_images_dir

    if image_paths is None:
        root = find_private_root()
        if root is None:
            raise FileNotFoundError("private_test not found under data/private_test/")
        img_dir = private_images_dir(root)
        image_paths = sorted(img_dir.glob("*.jpg"))[:limit]
    else:
        image_paths = list(image_paths)[:limit]

    if not image_paths:
        raise FileNotFoundError("No JPG files available for benchmark")

    profile = get_model_profile()

    if warmup:
        run_predict_with_metrics(Image.open(image_paths[0]).convert("RGB"))

    totals: list[float] = []
    ocr_times: list[float] = []
    extract_times: list[float] = []

    for path in image_paths:
        pred = run_predict_with_metrics(Image.open(path).convert("RGB"))
        timing = pred["timing_ms"]
        totals.append(timing["total"])
        ocr_times.append(timing["ocr"])
        extract_times.append(timing["extract"])

    return {
        "images": len(image_paths),
        "latency_ms": {
            "total_avg": round(statistics.mean(totals), 1),
            "total_p50": round(_percentile(totals, 50), 1),
            "total_p95": round(_percentile(totals, 95), 1),
            "ocr_avg": round(statistics.mean(ocr_times), 1),
            "extract_avg": round(statistics.mean(extract_times), 1),
        },
        "model_profile": profile,
    }


@lru_cache(maxsize=1)
def get_deploy_smoke_benchmark() -> dict[str, Any]:
    """One-image smoke benchmark — cached on Streamlit Cloud deploy / first load."""
    try:
        return run_benchmark(limit=1, warmup=1)
    except Exception as exc:
        return {
            "images": 0,
            "latency_ms": {},
            "model_profile": get_model_profile(),
            "error": str(exc),
        }
