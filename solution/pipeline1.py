# solution/pipeline.py
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from PIL import Image

# Nhúng trực tiếp các module xử lý mạnh nhất từ thư mục src/ của nhóm bạn
from solution.src.ocr import PaddleVietOCR, clean_ocr_text, extract_detector_boxes, crop_boxes
from solution.src.extractor import extract_brand_and_product, load_extractor_offline
from team_config import DEFAULT_MIN_CONF

# Khởi tạo biến toàn cục để lưu cache bộ đọc (Tránh nạp lại gây tràn RAM trên Streamlit Cloud)
_reader = None

def get_pipeline_reader():
    """Hàm khởi tạo và nạp trọng số offline một lần duy nhất khi Streamlit bật lên."""
    global _reader
    if _reader is None:
        print("🔮 [Team 27] Khởi tạo hệ thống Offline Vision Stack (PaddleOCR + VietOCR)...")
        _reader = PaddleVietOCR()
        print("🧠 [Team 27] Nạp tệp trọng số NLP Offline Guard Matrix...")
        load_extractor_offline()
    return _reader

def run_ocr_on_pil(image: Image.Image) -> str:
    """Quét chữ trực tiếp từ đối tượng PIL Image tải lên từ Streamlit giao diện."""
    try:
        reader = get_pipeline_reader()
        img_rgb = image.convert("RGB")
        
        # Tạo file tạm thời để PaddleX bốc ảnh trích xuất đa giác text boxes
        tmp_path = Path("data/tmp_streamlit_preview.jpg")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        img_rgb.save(tmp_path, "JPEG")
        
        detector_result = reader.detector.predict(str(tmp_path), batch_size=1)
        if tmp_path.exists():
            tmp_path.unlink() # Tiêu hủy file tạm trên đĩa sandbox lập tức
            
        boxes = extract_detector_boxes(detector_result)
        crops = crop_boxes(img_rgb, boxes)
        
        # Nhận diện chuỗi chữ tiếng Việt qua VietOCR Predictor
        text = " ".join(reader.recognizer.predict(crop) for crop in crops)
        return clean_ocr_text(text)
    except Exception as e:
        print(f"❌ [Team 27] Streamlit OCR Engine Error: {e}")
        return ""

def predict_from_image(
    img: Image.Image,
    min_conf: float = DEFAULT_MIN_CONF
) -> dict[str, str]:
    """
    🎯 MAIN ENTRY POINT: Khớp nối 100% hợp đồng API của Ban tổ chức.
    Streamlit_app.py và scripts/run_submission.py sẽ chỉ gọi duy nhất hàm này.
    """
    # Giai đoạn 1: Chạy nhánh Vision lấy văn bản thô sạch từ PIL Image
    ocr_text = run_ocr_on_pil(img)
    
    # Chuẩn hóa khoảng trắng đầu vào cho nhánh NLP
    ocr_text_clean = " ".join(str(ocr_text).strip().split())
    if not ocr_text_clean:
        ocr_text_clean = " "
        
    # Giai đoạn 2: Chạy nhánh NLP qua bộ gác cổng ranh giới từ nghiêm ngặt
    brand_name, product_name = extract_brand_and_product(ocr_text_clean)
    
    # Giai đoạn 3: Ép định dạng về khoảng trắng đơn " " nếu trường trống (Kaggle Convention)
    brand_final = str(brand_name).strip() if brand_name else " "
    product_final = str(product_name).strip() if product_name else " "
    
    return {
        "ocr_text": ocr_text_clean,
        "brand_name": brand_final if brand_final else " ",
        "product_name": product_final if product_final else " ",
    }


def predict_from_text(ocr_text: str) -> tuple[str, str]:
    """Hàm phụ trợ đáp ứng hợp đồng import của __init__.py mặc định từ BTC."""
    ocr_text_clean = " ".join(str(ocr_text).strip().split())
    brand_name, product_name = extract_brand_and_product(ocr_text_clean)
    return (brand_name if brand_name else " ", product_name if product_name else " ")   