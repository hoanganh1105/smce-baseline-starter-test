import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# 🌟 CẬP NHẬT: Bộ luật trả về trực tiếp bộ đôi (Thương hiệu, Sản phẩm)
BRAND_RULES = [
    # CHỦ ĐỀ 1: HẠ LONG CANFOCO
    (r"ha\s*long\s*canfoco.*pate.*c[ộo]t|c[ộo]t\s*đ[èe]n.*canfoco", ("Ha Long Canfoco", "Pate Cột Đèn Hải Phòng Hạ Long Canfoco")),
    (r"ha\s*long\s*canfoco.*pate|pate.*canfoco", ("Ha Long Canfoco", "Ha Long Canfoco Pate")),
    (r"ha\s*long\s*canfoco|halong\s*canfoco|canfoco", ("Ha Long Canfoco", "Ha Long Canfoco")),
    (r"ĐỒ\s*HỘP\s*HẠ\s*LONG", ("ĐỒ HỘP HẠ LONG", "ĐỒ HỘP HẠ LONG")),
    (r"Đồ\s*hộp\s*Hạ\s*Long", ("Đồ hộp Hạ Long", "Đồ hộp Hạ Long")),
    (r"đ[ồo]\s*h[ộo]p\s*h[ạa]\s*long", ("Đồ Hộp Hạ Long", "Đồ Hộp Hạ Long")),
    (r"pat[êe]\s*CỘT\s*ĐÈN\s*HẢI\s*PHÒNG", ("Pate Cột Đèn Hải Phòng", "patê CỘT ĐÈN HẢI PHÒNG")),
    (r"Pat[êe]\s*C[ộo]t\s*Đ[èe]n\s*H[ảa]i\s*Ph[òo]ng", ("Pate Cột Đèn Hải Phòng", "Patê Cột Đèn Hải Phòng")),
    (r"pate\s*c[ộo]t\s*đ[èe]n", ("Pate Cột Đèn Hải Phòng", "Pate Cột Đèn Hải Phòng")),
    (r"pate\s*cột\s*đèn", ("Pate Cột Đèn Hải Phòng", "pate cột đèn")),
    
    # CHỦ ĐỀ 2: HIGHLANDS COFFEE
    (r"highlands\s*coffee.*tr[àa]\s*sen|tr[àa]\s*sen.*highlands", ("Highlands Coffee", "Highlands Coffee Trà Sen Vàng")),
    (r"highlands\s*coffee.*tr[àa]\s*v[ảa]i|tr[àa]\s*v[ảa]i.*highlands", ("Highlands Coffee", "Highlands Coffee Trà Vải")),
    (r"HIGHLANDS\s*COFFEE", ("Highlands Coffee", "HIGHLANDS COFFEE")),
    (r"HIGHLAND\s*COFFEE", ("Highlands Coffee", "HIGHLAND COFFEE")),
    (r"TRÀ\s*SEN\s*VÀNG", ("Highlands Coffee", "TRÀ SEN VÀNG")),
    (r"highlands\s*coffee|highland\s*coffee", ("Highlands Coffee", "Highlands Coffee")),

    # CHỦ ĐỀ 3: NESTLÉ & SỮA NAN
    (r"nan\s*opti\s*pro\s*plus|nan\s*optipro\s*plus", ("Nestlé", "Nestlé NAN OPTIPRO PLUS")),
    (r"nan\s*infini\s*pro\s*a2|nan\s*infinipro\s*a2", ("Nestlé", "Nestlé NAN INFINIPRO A2")),
    (r"nan\s*supreme\s*pro|nan\s*supremepro", ("Nestlé", "Nestlé NAN SUPREMEpro")),
    (r"nan\s*opti\s*pro|nan\s*optipro", ("Nestlé", "Nestlé NAN OPTIPRO")),
    (r"Nan\s*optipro", ("Nestlé", "Nan optipro")),
    (r"Nestlé\s*NAN\s*Opti\s*Pro\s*Plus", ("Nestlé", "Nestlé NAN Opti Pro Plus")),
    (r"Nestlé\s*NA\s*OPTI\s*pro", ("Nestlé", "Nestlé NA OPTI pro")),
    (r"Sữa\s*bột\s*Nestlé\s*NAN\s*Infinipro\s*A2", ("Nestlé", "Sữa bột Nestlé NAN Infinipro A2")),
    (r"Sữa\s*Nestle\s*NAN\s*IFINIPRO\s*A2", ("Nestlé", "Sữa Nestle NAN IFINIPRO A2")),
    (r"NESTLÉ\s*NAN\s*INFINIPRO\s*A5", ("Nestlé", "NESTLÉ NAN INFINIPRO A5")),
    (r"s[ữ sữa]\s*nan|nan\b", ("Nestlé", "Nestlé NAN")),
    (r"NAN\b", ("Nestlé", "NAN")),
    (r"sữa\s*NAN", ("Nestlé", "sữa NAN")),
    (r"SỮA\s*CÔNG\s*THỨC\s*NESTLÉ", ("Nestlé", "SỮA CÔNG THỨC NESTLÉ")),
    (r"sữa\s*Nestlé", ("Nestlé", "sữa Nestlé")),
    (r"Sữa\s*Nestle", ("Nestlé", "Sữa Nestle")),
    (r"milo\b", ("Nestlé", "Nestlé Milo")),
    (r"nestl[ée]|nestle", ("Nestlé", "Nestlé")),

    # CHỦ ĐỀ 4: CÁC THƯƠNG HIỆU KHÁC
    (r"the\s*coffee\s*house", ("The Coffee House", "The Coffee House")),
    (r"ph[úu]c\s*long", ("Phúc Long", "Phúc Long")),
    (r"th\s*true|thtrue", ("TH True Milk", "TH True Milk")),
    (r"vinamilk", ("Vinamilk", "Vinamilk")),
    (r"nutifood|nuti\b", ("Nutifood", "Nutifood")),
    (r"dutch\s*lady|c[ôo]\s*g[áa]i\s*h[àa]\s*lan", ("Dutch Lady", "Dutch Lady")),
    (r"ensure\b", ("Abbott Ensure", "Abbott Ensure")),
    (r"pediasure", ("Abbott PediaSure", "Abbott PediaSure")),
    (r"aptamil", ("Aptamil", "Aptamil")),
    (r"\bpate\b|pat[êe]", ("Pate", "Pate"))
]

def infer_brand_from_product(product_name: str) -> str:
    """🌟 HÀM BỔ SUNG: Suy luận ngược tên Brand từ Product Name của mô hình ML."""
    if not product_name or product_name.strip() == "":
        return ""
    prod_lower = product_name.lower()
    
    if "highland" in prod_lower:
        return "Highlands Coffee"
    elif "nan" in prod_lower or "milo" in prod_lower or "nestlé" in prod_lower or "nestle" in prod_lower:
        return "Nestlé"
    elif "canfoco" in prod_lower or "hạ long" in prod_lower or "ha long" in prod_lower:
        return "Ha Long Canfoco"
    elif "cột đèn" in prod_lower or "pate" in prod_lower:
        return "Pate Cột Đèn Hải Phòng"
    elif "the coffee house" in prod_lower:
        return "The Coffee House"
    elif "phúc long" in prod_lower or "phuc long" in prod_lower:
        return "Phúc Long"
    elif "th true" in prod_lower:
        return "TH True Milk"
    elif "vinamilk" in prod_lower:
        return "Vinamilk"
    elif "nutifood" in prod_lower or "nuti" in prod_lower:
        return "Nutifood"
    elif "dutch lady" in prod_lower or "cô gái hà lan" in prod_lower:
        return "Dutch Lady"
    elif "ensure" in prod_lower:
        return "Abbott Ensure"
    elif "pediasure" in prod_lower:
        return "Abbott PediaSure"
    elif "aptamil" in prod_lower:
        return "Aptamil"
    
    return " "

def extract_product_rules(text: str) -> tuple[str, str]:
    """🌟 CẬP NHẬT: Trả về bộ đôi (brand_name, product_name)"""
    if not text or not text.strip():
        return "", ""
    tl = text.lower()
    tl = tl.replace("patê", "pate").replace("highland coffee", "highlands coffee")
    tl = re.sub(r"\bnat\b|\bna\b", "nan", tl) 

    for pattern, (brand, target_label) in BRAND_RULES:
        if re.search(pattern, tl, re.IGNORECASE):
            return brand, target_label
    return "", ""

class ProductPredictor:
    def __init__(self, min_class_count=1, prob_threshold=0.75, max_features=4000):
        self.min_class_count = min_class_count
        self.prob_threshold = prob_threshold
        self.max_features = max_features
        self._has_clf = self._prod_clf = None
        self.is_trained = False

    def fit(self, train_labels: pd.DataFrame):
        df = train_labels.copy()
        df["ocr_text"] = df["ocr_text"].astype(str).str.strip()
        df["product_name"] = df["product_name"].astype(str).str.strip()
        
        df = df[df["ocr_text"] != ""]
        
        self._has_clf = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=self.max_features, min_df=2)),
            ("clf", GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=42)),
        ])
        self._has_clf.fit(df["ocr_text"], (df["product_name"] != "").astype(int))
        
        pos = df[df["product_name"] != ""]
        keep = pos["product_name"].value_counts()
        keep = keep[keep >= self.min_class_count].index
        pos = pos[pos["product_name"].isin(keep)]
        
        self._prod_clf = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=self.max_features, min_df=2)),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)),
        ])
        
        if len(pos) > 0:
            self._prod_clf.fit(pos["ocr_text"], pos["product_name"])
            self.is_trained = True
        else:
            self.is_trained = False
            
        return self

    def predict_dual(self, ocr_text: str) -> tuple[str, str]:
            """🌟 HÀM MỚI: Trả về đồng thời cả Brand và Product Name kết hợp Keyword Guard."""
            ocr_text = "" if ocr_text is None else str(ocr_text).strip()
            if not ocr_text:
                return "", ""
            
            # 1. Luôn ưu tiên quét luật Regex cứng trước để ăn điểm chắc chắn
            brand_ruled, product_ruled = extract_product_rules(ocr_text)
            if product_ruled:
                return brand_ruled, product_ruled
                
            # 2. Nếu không trúng luật cứng, kiểm tra mô hình ML
            if not self.is_trained or self._has_clf is None or self._prod_clf is None:
                return "", ""
                
            try:
                # Kiểm tra bộ gác cổng nhị phân (Product Presence Classifier)
                proba = self._has_clf.predict_proba([ocr_text])[0]
                if 1 not in self._has_clf.classes_ or proba[list(self._has_clf.classes_).index(1)] < self.prob_threshold:
                    return "", ""
                
                # Dự đoán nhãn sản phẩm và suy diễn thương hiệu từ mô hình ML
                pred_product = str(self._prod_clf.predict([ocr_text])[0])
                pred_brand = infer_brand_from_product(pred_product)
                
                # 🌟 BỘ LỌC BẢO VỆ (KEYWORD GUARD): Chống ML đoán bừa trên văn bản nhiễu
                txt_lower = ocr_text.lower()
                brand_lower = pred_brand.lower()
                
                # Định nghĩa ma trận từ khóa bắt buộc phải xuất hiện trong văn bản OCR thô
                kw_map = {
                    "nestlé": ["nestle", "nestlé", "nan", "milo"],
                    "ha long canfoco": ["ha long", "hạ long", "canfoco", "đồ hộp", "do hop"],
                    "pate cột đèn hải phòng": ["cột đèn", "cot den", "pate", "patê"],
                    "highlands coffee": ["highland", "highlands", "coffee", "cà phê"],
                    "the coffee house": ["coffee house", "the coffee"],
                    "phúc long": ["phúc long", "phuc long"],
                    "th true milk": ["th true", "th milk"],
                    "vinamilk": ["vinamilk", "dielac"],
                    "nutifood": ["nutifood", "nuti"],
                    "abbott ensure": ["ensure"],
                    "abbott pediasure": ["pediasure", "pedia"],
                    "aptamil": ["aptamil"]
                }
                
                # Nếu thương hiệu dự đoán nằm trong danh sách kiểm tra
                if brand_lower in kw_map:
                    # Nếu văn bản OCR thô KHÔNG chứa bất kỳ từ khóa cốt lõi nào -> Ép kết quả về rỗng (Nhiễu)
                    if not any(kw in txt_lower for kw in kw_map[brand_lower]):
                        return "", ""
                        
                return pred_brand, pred_product
            except Exception:
                return "", ""

predictor_instance = ProductPredictor()

def setup_extractor(train_csv_path: str):
    df = pd.read_csv(train_csv_path, keep_default_na=False)
    print(f"🧠 Đang huấn luyện bộ đôi Cây quyết định (Gradient Boosting + Random Forest) trên {len(df)} dòng...")
    predictor_instance.fit(df)
    if predictor_instance.is_trained:
        print("✅ Huấn luyện cấu trúc phi tuyến hoàn tất!")
        weight_dir = Path(__file__).resolve().parent.parent / "weights"
        weight_dir.mkdir(parents=True, exist_ok=True)
        
        model_pack = {
            "has_clf": predictor_instance._has_clf,
            "prod_clf": predictor_instance._prod_clf,
            "is_trained": predictor_instance.is_trained
        }
        joblib.dump(model_pack, weight_dir / "nlp_extractor_weights.pkl")
        print(f"💾 Đã đóng gói thành công tệp trọng số NLP tại: weights/nlp_extractor_weights.pkl")

def load_extractor_offline():
    """🌟 HÀM NÂNG CẤP: Tự động tải weights từ Google Drive nếu chạy trên Streamlit Cloud."""
    global predictor_instance
    
    # Định vị chính xác thư mục chứa weights trong cấu trúc mới
    weight_dir = Path(__file__).resolve().parent.parent.parent / "weights"
    weight_path = weight_dir / "nlp_extractor_weights.pkl"
    
    # Nếu file chưa tồn tại (kịch bản khi vừa deploy lên Streamlit Cloud sạch)
    if not weight_path.exists():
        print("🌐 [Team 27] Không tìm thấy file weights cục bộ. Đang kích hoạt luồng tải tự động từ Google Drive...")
        try:
            import gdown
            weight_dir.mkdir(parents=True, exist_ok=True)
            
            # Trích xuất ID file từ link Drive chính thức của Hoàng Anh
            file_id = "1Z8BgcBwfmdOx3XFzTWcthuiH0-Gcx1ib"
            download_url = f"https://drive.google.com/uc?id={file_id}"
            
            print(f"⏳ Đang tải tệp nlp_extractor_weights.pkl (115MB) về: {weight_path}")
            gdown.download(download_url, str(weight_path), quiet=False)
            print("✨ Tải file weights thành công!")
        except Exception as e:
            print(f"❌ Lỗi tự động tải weights từ Drive: {e}. Hệ thống sẽ chuyển sang chế độ dự phòng (Fallback).")

    # Tiến hành nạp weights vào RAM nếu file đã sẵn sàng
    if weight_path.exists():
        print(f"📂 Đang nạp tệp trọng số NLP offline từ package: {weight_path}")
        model_pack = joblib.load(weight_path)
        predictor_instance._has_clf = model_pack["has_clf"]
        predictor_instance._prod_clf = model_pack["prod_clf"]
        predictor_instance.is_trained = model_pack["is_trained"]
        print("✅ Nạp weights NLP thành công! Bộ lọc đa lớp kết hợp Guard Matrix đã sẵn sàng.")
    else:
        print("⚠️ Chế độ dự phòng: Chỉ sử dụng Regex Rules Layer để bóc tách nhãn.")

def extract_brand_and_product(ocr_text: str) -> tuple[str, str]:
    """🌟 HÀM MỚI: Interface phục vụ cho file chạy chính."""
    return predictor_instance.predict_dual(ocr_text)