from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


def clean_ocr_text(text: object, max_chars: int = 500) -> str:
    """Apply competition-safe OCR text formatting."""
    value = "" if text is None else str(text)
    value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if is_short_noise(value):
        return ""
    return value[:max_chars].strip()


def is_short_noise(text: str) -> bool:
    """Filter detector false positives from no-text images."""
    tokens = re.findall(r"[A-Za-zÀ-ỹĐđ0-9]+", text)
    if not tokens:
        return False
    if len(tokens) <= 4 and all(len(token) <= 1 for token in tokens):
        return True
    if len(tokens) <= 3 and sum(char.isdigit() for char in text) >= sum(char.isalpha() for char in text):
        return True
    return False


def _result_data(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        data = item.get("res", item)
        return data if isinstance(data, dict) else None

    try:
        data = item["res"]
    except Exception:
        data = getattr(item, "res", None)

    if data is None and hasattr(item, "json"):
        try:
            item_json = item.json() if callable(item.json) else item.json
            data = item_json.get("res", item_json)
        except Exception:
            data = None

    return data if isinstance(data, dict) else None


def _first_available(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def extract_detector_boxes(result: Any) -> list[Any]:
    result_items = result if isinstance(result, list) else [result]
    for item in result_items:
        data = _result_data(item)
        if not data:
            continue
        boxes = _first_available(data, "dt_polys", "rec_polys", "rec_boxes")
        if boxes is not None:
            return list(boxes)
    return []


def box_position(box: Any) -> tuple[float, float]:
    try:
        points = getattr(box, "tolist", lambda: box)()
        if len(points) == 4 and not isinstance(points[0], (list, tuple)):
            x_min, y_min, _, _ = points
            return float(y_min), float(x_min)
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return min(ys), min(xs)
    except Exception:
        return 0.0, 0.0


def sorted_boxes_top_to_bottom_left_to_right(boxes: list[Any]) -> list[Any]:
    return sorted(boxes, key=box_position)


def crop_boxes(image: Image.Image, boxes: list[Any], margin: int = 2) -> list[Image.Image]:
    crops: list[Image.Image] = []
    width, height = image.size
    for box in sorted_boxes_top_to_bottom_left_to_right(boxes):
        try:
            points = getattr(box, "tolist", lambda: box)()
            if len(points) == 4 and not isinstance(points[0], (list, tuple)):
                x_min, y_min, x_max, y_max = [int(v) for v in points]
            else:
                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(width, x_max + margin)
            y_max = min(height, y_max + margin)
            if x_max > x_min and y_max > y_min:
                crops.append(image.crop((x_min, y_min, x_max, y_max)))
        except Exception:
            continue
    return crops


@dataclass
class OCRResult:
    text: str
    seconds: float
    error: str = ""


class PaddleVietOCR:
    """CPU OCR: PaddleOCR text detection + VietOCR recognition."""

    def __init__(self) -> None:
        from paddleocr import TextDetection
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        # Định nghĩa đường dẫn weights cục bộ trong package
        base_weight_dir = Path(__file__).resolve().parents[1] / "weights"
        paddle_weight_dir = base_weight_dir / "PP-OCRv6_tiny_det"
        vietocr_weight_path = base_weight_dir / "vgg_seq2seq.pth"

        # 🌟 SỬA TẠI ĐÂY: Tên model bắt buộc phải giữ nguyên chuỗi chuẩn để không bị lỗi Registry
        detector_model = "PP-OCRv6_tiny_det"
        detector_box_thresh = float(os.getenv("PADDLE_TEXT_DET_BOX_THRESH", "0.6"))

        # 🌟 SỬA TẠI ĐÂY: Nếu folder cục bộ tồn tại, truyền nó vào tham số model_dir để ép chạy offline
        if paddle_weight_dir.exists():
            self.detector = TextDetection(
                model_name=detector_model, 
                model_dir=str(paddle_weight_dir),  # Ép PaddleX đọc cấu hình offline từ đây
                device="cpu", 
                box_thresh=detector_box_thresh, 
                enable_mkldnn=False
            )
        else:
            # Phương án dự phòng nếu chưa có folder weights
            self.detector = TextDetection(
                model_name=detector_model, 
                device="cpu", 
                box_thresh=detector_box_thresh, 
                enable_mkldnn=False
            )
        
        config = Cfg.load_config_from_name("vgg_seq2seq")
        config["device"] = "cpu"
        # Ép VietOCR lấy weights trực tiếp từ file .pth đã đóng gói cục bộ
        if vietocr_weight_path.exists():
            config["weights"] = str(vietocr_weight_path)
            
        if "predictor" in config:
            config["predictor"]["beamsearch"] = False
        self.recognizer = Predictor(config)
    def recognize(self, image_path: str | Path) -> OCRResult:
        start = time.perf_counter()
        try:
            image_path = Path(image_path)
            image = Image.open(image_path).convert("RGB")
            detector_result = self.detector.predict(str(image_path), batch_size=1)
            boxes = extract_detector_boxes(detector_result)
            crops = crop_boxes(image, boxes)
            text = " ".join(self.recognizer.predict(crop) for crop in crops)
            return OCRResult(clean_ocr_text(text), time.perf_counter() - start)
        except Exception as exc:
            return OCRResult("", time.perf_counter() - start, str(exc))


_reader: PaddleVietOCR | None = None


def get_reader() -> PaddleVietOCR:
    global _reader
    if _reader is None:
        _reader = PaddleVietOCR()
    return _reader


def get_ocr_text(image_path: str) -> str:
    """Compatibility wrapper for callers that only need text."""
    return get_reader().recognize(image_path).text