#!/usr/bin/env python3
"""Benchmark inference latency and report model footprint (template-owned)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.benchmark import get_model_profile, run_predict_with_metrics  # noqa: E402
from shared.data_utils import find_private_root, private_images_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark team solution latency + footprint")
    parser.add_argument("--limit", type=int, default=6, help="Number of images (default: 6)")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before timing (loads OCR weights)",
    )
    args = parser.parse_args()

    root = find_private_root()
    if root is None:
        raise SystemExit("private_test not found under data/private_test/")

    img_dir = private_images_dir(root)
    image_paths = sorted(img_dir.glob("*.jpg"))[: args.limit]
    if not image_paths:
        raise SystemExit(f"No JPG files under {img_dir}")

    print("Model profile:")
    print(json.dumps(get_model_profile(), indent=2, ensure_ascii=False))
    print()

    if args.warmup:
        print(f"Warmup ({args.warmup} image)...")
        for _ in range(args.warmup):
            run_predict_with_metrics(Image.open(image_paths[0]).convert("RGB"))

    print(f"Benchmarking {len(image_paths)} image(s)...")
    totals = []
    ocr_times = []
    extract_times = []
    for path in image_paths:
        pred = run_predict_with_metrics(Image.open(path).convert("RGB"))
        timing = pred["timing_ms"]
        totals.append(timing["total"])
        ocr_times.append(timing["ocr"])
        extract_times.append(timing["extract"])
        print(
            f"  {path.stem}: total={timing['total']:.1f}ms "
            f"(ocr={timing['ocr']:.1f}, extract={timing['extract']:.1f})"
        )

    import statistics

    def _pct(values: list[float], p: float) -> float:
        ordered = sorted(values)
        idx = max(0, min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1))))
        return ordered[idx]

    report = {
        "images": len(image_paths),
        "latency_ms": {
            "total_avg": round(statistics.mean(totals), 1),
            "total_p50": round(_pct(totals, 50), 1),
            "total_p95": round(_pct(totals, 95), 1),
            "ocr_avg": round(statistics.mean(ocr_times), 1),
            "extract_avg": round(statistics.mean(extract_times), 1),
        },
        "model_profile": get_model_profile(),
    }

    print()
    print("Summary:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
