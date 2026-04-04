"""
data_pipeline.py — Module thu thập, làm sạch, và tiền xử lý dữ liệu
cho dự án AI Travel Planner & Recommender System.

Datasets:
  1. Vietnam Weather Data (181K records, 40 tỉnh, 2009-2021) — Vietnam-specific
  2. 515K Hotel Reviews Europe — used for ML sentiment training (see NOTE below)
  3. Travel Review Ratings (UCI — 24 loại hình, 5456 users) — worldwide user preferences
  4. Hotel Booking Demand (119K bookings) — worldwide booking patterns
  5. Worldwide Travel Cities Ratings & Climate (560 cities) — city classification

NOTE on geographic scope:
  Datasets 2-5 are worldwide/European, NOT Vietnam-specific. This is intentional:
  - No large, open Vietnam hotel review or booking datasets exist on Kaggle/UCI.
  - These datasets are used to train general ML models (sentiment, classification,
    user clustering). The techniques (TF-IDF, Decision Tree, Naive Bayes) are
    domain-agnostic and transfer across geographies.
  - Vietnam-specific data (weather, 50 tourist places, distance/cost matrices)
    is already covered by Dataset 1 and the built-in VN_TOURIST_PLACES constants.
    OSM data (data/osm/vietnam_places.csv) can further enrich this if extracted.
  - Switching to a small 20K Vietnamese review dataset would reduce ML training
    quality by 25x with no meaningful gain for the AI components.

Sử dụng trong Google Colab — tự động download từ Kaggle qua opendatasets.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING SETUP
# ============================================================
logger = logging.getLogger("data_pipeline")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ============================================================
# 0. CONSTANTS & CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

# --- Configurable parameters (teammates: adjust these as needed) ---
TRAVEL_COST_PER_KM = 3000       # VND — average cost per km (bus/taxi)
TRAVEL_AVG_SPEED_KMH = 40       # km/h — average road speed in Vietnam
DEFAULT_VISIT_HOURS = {          # hours per place category
    "culture": 2.0,
    "entertainment": 2.0,
    "nature": 3.0,
    "beach": 3.0,
    "adventure": 3.0,
}
DEFAULT_OPENING_HOURS = {        # (open, close) per category
    "entertainment": (9, 22),
    "_default": (7, 17),
}
RAIN_THRESHOLD_MM = 1.0          # mm — above this = rainy
HUMIDITY_THRESHOLD = 80          # % — above this = humid
TEMPERATURE_HOT_THRESHOLD = 35   # °C — above this = hot
OUTDOOR_LIMITS = {               # thresholds for outdoor_suitable flag
    "max_rain_mm": 5,
    "max_temp": 38,
    "max_humidity": 90,
}

KAGGLE_DATASETS = {
    "vietnam_weather": "vanviethieuanh/vietnam-weather-data",
    "hotel_reviews": "jiashenliu/515k-hotel-reviews-data-in-europe",
    "travel_ratings": "ishbhms/travel-review-ratings",
    "hotel_bookings": "jessemostipak/hotel-booking-demand",
    "world_cities": "furkanima/worldwide-travel-cities-ratings-and-climate",
}

# Mapping tên tỉnh Vietnam (chuẩn hoá)
VN_PROVINCE_COORDS = {
    "Ha Noi": (21.0285, 105.8542),
    "Ho Chi Minh": (10.8231, 106.6297),
    "Da Nang": (16.0544, 108.2022),
    "Hue": (16.4637, 107.5909),
    "Nha Trang": (12.2388, 109.1967),
    "Da Lat": (11.9404, 108.4583),
    "Hai Phong": (20.8449, 106.6881),
    "Can Tho": (10.0452, 105.7469),
    "Vung Tau": (10.3460, 107.0843),
    "Phu Quoc": (10.2270, 103.9631),
    "Quang Ninh": (21.0064, 107.2925),
    "Lao Cai": (22.4856, 103.9707),
    "Ninh Binh": (20.2506, 105.9745),
    "Quang Nam": (15.5394, 108.0191),
    "Binh Thuan": (11.0904, 108.0721),
    "Khanh Hoa": (12.2585, 109.0526),
    "Lam Dong": (11.5753, 108.1429),
    "Binh Dinh": (13.7830, 109.2197),
    "Phu Yen": (13.0882, 109.0929),
    "Gia Lai": (13.9833, 108.0000),
    "Dak Lak": (12.7100, 108.2378),
    "Thanh Hoa": (19.8067, 105.7852),
    "Nghe An": (18.6783, 105.6813),
    "Ha Tinh": (18.3559, 105.8877),
    "Quang Binh": (17.4690, 106.6222),
    "Quang Tri": (16.7500, 107.1854),
    "Thua Thien Hue": (16.4637, 107.5909),
    "Kon Tum": (14.3498, 108.0004),
    "Dak Nong": (12.0000, 107.6917),
    "An Giang": (10.3899, 105.4353),
    "Ben Tre": (10.2434, 106.3756),
    "Dong Thap": (10.4938, 105.6882),
    "Kien Giang": (9.8250, 105.1259),
    "Long An": (10.5362, 106.4133),
    "Soc Trang": (9.6025, 105.9740),
    "Tien Giang": (10.4493, 106.3421),
    "Tra Vinh": (9.9347, 106.3456),
    "Vinh Long": (10.2538, 105.9722),
    "Bac Lieu": (9.2940, 105.7216),
    "Ca Mau": (9.1527, 105.1961),
}

# Toạ độ các điểm du lịch nổi tiếng Việt Nam
# Format: "Tên địa điểm": (latitude, longitude, category, province, entry_fee_vnd)
# Categories: "nature" | "culture" | "beach" | "adventure" | "entertainment"
# Distribution (50 places): culture 15 (30%), nature 15 (30%), beach 8 (16%),
#                            adventure 6 (12%), entertainment 6 (12%)
VN_TOURIST_PLACES = {
    # --- CULTURE (15) ---
    "Hoan Kiem Lake":           (21.0288, 105.8525, "culture", "Ha Noi",        0),
    "Temple of Literature":     (21.0275, 105.8360, "culture", "Ha Noi",        30000),
    "Ho Chi Minh Mausoleum":    (21.0369, 105.8350, "culture", "Ha Noi",        0),
    "Old Quarter Hanoi":        (21.0340, 105.8500, "culture", "Ha Noi",        0),
    "Imperial City Hue":        (16.4698, 107.5786, "culture", "Hue",           200000),
    "My Son Sanctuary":         (15.7644, 108.1241, "culture", "Quang Nam",     150000),
    "Hoi An Ancient Town":      (15.8801, 108.3380, "culture", "Quang Nam",     120000),
    "Cu Chi Tunnels":           (11.1415, 106.4627, "culture", "Ho Chi Minh",   110000),
    "Ben Thanh Market":         (10.7725, 106.6980, "culture", "Ho Chi Minh",   0),
    "Notre Dame Cathedral HCMC":(10.7798, 106.6990, "culture", "Ho Chi Minh",   0),
    "War Remnants Museum":      (10.7794, 106.6920, "culture", "Ho Chi Minh",   40000),
    "Po Nagar Towers":          (12.2655, 109.1952, "culture", "Khanh Hoa",     22000),
    "Crazy House Da Lat":       (11.9363, 108.4310, "culture", "Lam Dong",      80000),
    "Bai Dinh Pagoda":          (20.2731, 105.8644, "culture", "Ninh Binh",     100000),
    "Long Son Pagoda":          (12.2499, 109.1840, "culture", "Khanh Hoa",     0),

    # --- NATURE (15) ---
    "Ha Long Bay":              (20.9101, 107.1839, "nature", "Quang Ninh",     0),
    "Marble Mountains":         (16.0034, 108.2628, "nature", "Da Nang",        40000),
    "Golden Bridge":            (15.9940, 107.9969, "nature", "Da Nang",        900000),
    "Xuan Huong Lake":          (11.9460, 108.4410, "nature", "Lam Dong",       0),
    "Valley of Love":           (11.9660, 108.4390, "nature", "Lam Dong",       100000),
    "Mui Ne Sand Dunes":        (10.9333, 108.2869, "nature", "Binh Thuan",     0),
    "Sapa":                     (22.3363, 103.8438, "nature", "Lao Cai",        0),
    "Trang An Landscape":       (20.2500, 105.9000, "nature", "Ninh Binh",      200000),
    "Phong Nha Cave":           (17.5920, 106.2835, "nature", "Quang Binh",     150000),
    "Cat Ba Island":            (20.7267, 107.0458, "nature", "Hai Phong",      0),
    "Ban Gioc Waterfall":       (22.8567, 106.7072, "nature", "Cao Bang",       45000),
    "Ha Giang Rock Plateau":    (23.0079, 105.3144, "nature", "Ha Giang",       0),
    "Tam Coc":                  (20.2167, 105.9333, "nature", "Ninh Binh",      150000),
    "Pu Luong Nature Reserve":  (20.3390, 105.1660, "nature", "Thanh Hoa",      0),
    "Cuc Phuong National Park": (20.2317, 105.6508, "nature", "Ninh Binh",      150000),

    # --- BEACH (8) ---
    "Nha Trang Beach":          (12.2464, 109.1960, "beach", "Khanh Hoa",       0),
    "Phu Quoc Beach":           (10.2899, 103.9840, "beach", "Kien Giang",      0),
    "Da Nang Beach":            (16.0544, 108.2022, "beach", "Da Nang",         0),
    "Quy Nhon Beach":           (13.7765, 109.2196, "beach", "Binh Dinh",       0),
    "Phan Thiet Beach":         (10.9289, 108.1022, "beach", "Binh Thuan",      0),
    "Con Dao Beach":            (8.6810,  106.5983, "beach", "Ba Ria Vung Tau", 0),
    "Lang Co Beach":            (16.2167, 108.0500, "beach", "Thua Thien Hue",  0),
    "Sam Son Beach":            (19.7455, 105.9055, "beach", "Thanh Hoa",       0),

    # --- ADVENTURE (6) ---
    "Son Doong Cave":           (17.5556, 106.1467, "adventure", "Quang Binh",      70000000),
    "Fansipan Summit":          (22.3033, 103.7750, "adventure", "Lao Cai",         700000),
    "Hang En Cave":             (17.5444, 106.1556, "adventure", "Quang Binh",      5500000),
    "Bach Ma National Park":    (16.1979, 107.8562, "adventure", "Thua Thien Hue",  60000),
    "Moc Chau Highland":        (20.8290, 104.6849, "adventure", "Son La",          0),
    "Lung Cu Flag Tower":       (23.3714, 105.3353, "adventure", "Ha Giang",        20000),

    # --- ENTERTAINMENT (6) ---
    "Ba Na Hills":              (15.9975, 107.9964, "entertainment", "Da Nang",     900000),
    "Vinpearl Nha Trang":       (12.2167, 109.2340, "entertainment", "Khanh Hoa",   880000),
    "Landmark 81":              (10.7953, 106.7220, "entertainment", "Ho Chi Minh", 200000),
    "Dragon Bridge Da Nang":    (16.0604, 108.2272, "entertainment", "Da Nang",     0),
    "Night Market Hoi An":      (15.8792, 108.3367, "entertainment", "Quang Nam",   0),
    "West Lake Hanoi":          (21.0617, 105.8128, "entertainment", "Ha Noi",      0),
}


def ensure_dirs():
    """Tạo các thư mục cần thiết nếu chưa có."""
    for d in [RAW_DIR, CLEANED_DIR, PROCESSED_DIR, FEATURES_DIR]:
        os.makedirs(d, exist_ok=True)


# ============================================================
# 1. DOWNLOAD
# ============================================================

def download_all_datasets(use_opendatasets=True):
    """
    Tải tất cả datasets từ Kaggle.
    Trong Colab, dùng opendatasets (tự nhập Kaggle credentials).
    """
    ensure_dirs()

    if use_opendatasets:
        try:
            import opendatasets as od
        except ImportError:
            os.system("pip install opendatasets -q")
            import opendatasets as od

        for name, dataset_id in KAGGLE_DATASETS.items():
            url = f"https://www.kaggle.com/datasets/{dataset_id}"
            logger.info("DOWNLOAD %s: %s", name, url)
            try:
                od.download(url, data_dir=RAW_DIR)
                logger.info("  -> OK")
            except Exception as e:
                logger.error("  -> ERROR: %s", e)
    else:
        logger.info("Dùng Kaggle CLI: kaggle datasets download -d <dataset_id>")
        for name, dataset_id in KAGGLE_DATASETS.items():
            cmd = f'kaggle datasets download -d {dataset_id} -p "{RAW_DIR}" --unzip'
            logger.info("  %s", cmd)
            os.system(cmd)

    logger.info("DONE — Tất cả datasets đã được tải về: %s", RAW_DIR)


def find_csv(raw_dir, keyword):
    """Tìm file CSV trong thư mục raw chứa keyword trong tên hoặc đường dẫn."""
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".csv") and keyword.lower() in os.path.join(root, f).lower():
                return os.path.join(root, f)
    return None


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

def load_vietnam_weather():
    """Load Vietnam Weather Data."""
    path = find_csv(RAW_DIR, "vietnam") or find_csv(RAW_DIR, "weather")
    if path is None:
        raise FileNotFoundError("Không tìm thấy Vietnam Weather CSV trong data/raw/")
    df = pd.read_csv(path)
    logger.info("LOAD Vietnam Weather: %s rows, %d cols — %s", f"{df.shape[0]:,}", df.shape[1], path)
    return df


def load_hotel_reviews():
    """Load 515K Hotel Reviews."""
    path = (find_csv(RAW_DIR, "hotel_reviews")
            or find_csv(RAW_DIR, "Hotel_Reviews")
            or find_csv(RAW_DIR, "515k"))
    if path is None:
        raise FileNotFoundError("Không tìm thấy Hotel Reviews CSV trong data/raw/")
    df = pd.read_csv(path)
    logger.info("LOAD Hotel Reviews: %s rows, %d cols — %s", f"{df.shape[0]:,}", df.shape[1], path)
    return df


def load_travel_ratings():
    """Load Travel Review Ratings (UCI)."""
    path = (find_csv(RAW_DIR, "google_review")
            or find_csv(RAW_DIR, "review_rating")
            or find_csv(RAW_DIR, "travel-review"))
    if path is None:
        raise FileNotFoundError("Không tìm thấy Travel Ratings CSV trong data/raw/")
    df = pd.read_csv(path)
    logger.info("LOAD Travel Ratings: %s rows, %d cols — %s", f"{df.shape[0]:,}", df.shape[1], path)
    return df


def load_hotel_bookings():
    """Load Hotel Booking Demand (Kaggle: jessemostipak/hotel-booking-demand).
    File: 'hotel_bookings.csv' (119,390 rows)
    """
    path = (find_csv(RAW_DIR, "hotel_bookings")
            or find_csv(RAW_DIR, "hotel-booking-demand")
            or find_csv(RAW_DIR, "booking"))
    if path is None:
        raise FileNotFoundError(
            "Không tìm thấy Hotel Bookings CSV trong data/raw/. "
            "File cần tìm: 'hotel_bookings.csv' (kaggle: jessemostipak/hotel-booking-demand)"
        )
    df = pd.read_csv(path)
    logger.info("LOAD Hotel Bookings: %s rows, %d cols — %s", f"{df.shape[0]:,}", df.shape[1], path)
    return df


def load_world_cities():
    """Load Worldwide Travel Cities (Kaggle: furkanima/worldwide-travel-cities-ratings-and-climate).
    Actual file name: 'Worldwide Travel Cities Dataset (Ratings and Climate).csv'
    """
    path = (find_csv(RAW_DIR, "worldwide travel cities")
            or find_csv(RAW_DIR, "cities dataset")
            or find_csv(RAW_DIR, "ratings and climate"))
    if path is None:
        raise FileNotFoundError("Không tìm thấy World Cities CSV trong data/raw/")
    df = pd.read_csv(path)
    logger.info("LOAD World Cities: %s rows, %d cols — %s", f"{df.shape[0]:,}", df.shape[1], path)
    return df


# ============================================================
# 3. CLEAN DATA
# ============================================================

def _validate_dataframe(df, name, min_rows=1, required_cols=None):
    """Validate input DataFrame before cleaning."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name}: expected pandas DataFrame, got {type(df).__name__}")
    if df.empty or len(df) < min_rows:
        raise ValueError(f"{name}: DataFrame is empty or has fewer than {min_rows} rows")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning("%s: missing expected columns %s (available: %s)",
                           name, missing, list(df.columns[:10]))
    return True


def clean_vietnam_weather(df):
    """
    Làm sạch Vietnam Weather Data.
    - Chuẩn hoá cột, xử lý missing, tạo thêm cột thời gian.
    """
    _validate_dataframe(df, "Vietnam Weather")
    df = df.copy()

    # Chuẩn hoá tên cột
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "province" in cl or "station" in cl or "city" in cl:
            col_map[c] = "province"
        elif cl in ("max", "max_temp", "tmax"):
            col_map[c] = "temp_max"
        elif cl in ("min", "min_temp", "tmin"):
            col_map[c] = "temp_min"
        elif "wind_d" in cl or "wind_dir" in cl:
            col_map[c] = "wind_dir"
        elif "wind" in cl:
            col_map[c] = "wind_speed"
        elif "rain" in cl or "precip" in cl:
            col_map[c] = "rain_mm"
        elif "humidi" in cl or "humid" in cl:
            col_map[c] = "humidity"
        elif "cloud" in cl:
            col_map[c] = "cloud_cover"
        elif "pressure" in cl or "press" in cl:
            col_map[c] = "pressure"
        elif "date" in cl or "time" in cl:
            col_map[c] = "date"
    if col_map:
        df = df.rename(columns=col_map)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["season"] = df["month"].map(
            lambda m: "spring" if m in [2, 3, 4]
            else "summer" if m in [5, 6, 7]
            else "autumn" if m in [8, 9, 10]
            else "winter"
        )

    # Chuyển cột số
    num_cols = ["temp_max", "temp_min", "wind_speed", "rain_mm", "humidity", "cloud_cover", "pressure"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Xử lý missing: dùng median theo tỉnh
    if "province" in df.columns:
        for c in num_cols:
            if c in df.columns:
                df[c] = df.groupby("province")[c].transform(
                    lambda x: x.fillna(x.median())
                )

    # Fillna còn lại bằng median toàn bộ
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Loại bỏ rows không có date
    if "date" in df.columns:
        df = df.dropna(subset=["date"])

    # Tạo nhãn thời tiết cho IF-THEN rules (thresholds from config)
    if "rain_mm" in df.columns:
        df["is_rainy"] = (df["rain_mm"] > RAIN_THRESHOLD_MM).astype(int)
        df["rain_level"] = pd.cut(
            df["rain_mm"], bins=[-1, 0, 5, 20, 50, 999],
            labels=["none", "light", "moderate", "heavy", "extreme"]
        )
    if "humidity" in df.columns:
        df["is_humid"] = (df["humidity"] > HUMIDITY_THRESHOLD).astype(int)
    if "temp_max" in df.columns:
        df["is_hot"] = (df["temp_max"] > TEMPERATURE_HOT_THRESHOLD).astype(int)
    if all(c in df.columns for c in ["rain_mm", "temp_max", "humidity"]):
        df["outdoor_suitable"] = (
            (df["rain_mm"] <= OUTDOOR_LIMITS["max_rain_mm"])
            & (df["temp_max"] <= OUTDOOR_LIMITS["max_temp"])
            & (df["humidity"] <= OUTDOOR_LIMITS["max_humidity"])
        ).astype(int)

    logger.info("CLEAN Vietnam Weather: %s rows, %d cols", f"{df.shape[0]:,}", df.shape[1])
    return df


def clean_hotel_reviews(df):
    """
    Làm sạch 515K Hotel Reviews.
    - Xử lý text, tạo nhãn sentiment, loại bỏ reviews trống.
    """
    _validate_dataframe(df, "Hotel Reviews", required_cols=["Reviewer_Score"])
    df = df.copy()

    # Loại bỏ reviews rỗng
    if "Positive_Review" in df.columns:
        df["Positive_Review"] = df["Positive_Review"].fillna("").str.strip()
    if "Negative_Review" in df.columns:
        df["Negative_Review"] = df["Negative_Review"].fillna("").str.strip()

    # Tạo cột review gộp
    if "Positive_Review" in df.columns and "Negative_Review" in df.columns:
        df["full_review"] = (df["Positive_Review"] + " " + df["Negative_Review"]).str.strip()

    # Loại bỏ placeholder text
    placeholder = "no negative" # Common pattern: "No Negative", "Nothing", etc.
    if "Negative_Review" in df.columns:
        df["has_negative"] = ~df["Negative_Review"].str.lower().str.contains(
            "no negative|nothing|none|na|n a", na=False
        )
    if "Positive_Review" in df.columns:
        df["has_positive"] = ~df["Positive_Review"].str.lower().str.contains(
            "no positive|nothing|none|na|n a", na=False
        )

    # Tạo nhãn sentiment từ Reviewer_Score
    if "Reviewer_Score" in df.columns:
        df["Reviewer_Score"] = pd.to_numeric(df["Reviewer_Score"], errors="coerce")
        df["sentiment"] = pd.cut(
            df["Reviewer_Score"],
            bins=[0, 4, 6, 8, 10],
            labels=["negative", "neutral", "positive", "very_positive"]
        )
        df["sentiment_binary"] = (df["Reviewer_Score"] >= 7).astype(int)

    # Parse Review_Date
    if "Review_Date" in df.columns:
        df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")

    # Tạo text features
    if "full_review" in df.columns:
        df["review_word_count"] = df["full_review"].str.split().str.len().fillna(0).astype(int)
        df["review_char_count"] = df["full_review"].str.len().fillna(0).astype(int)

    # Drop rows thiếu score
    if "Reviewer_Score" in df.columns:
        df = df.dropna(subset=["Reviewer_Score"])

    logger.info("CLEAN Hotel Reviews: %s rows, %d cols", f"{df.shape[0]:,}", df.shape[1])
    return df


def clean_travel_ratings(df):
    """
    Làm sạch Travel Review Ratings (UCI).
    - Rename cột Category 1-24 thành tên mô tả.
    - 24 cột rating, chuẩn hoá, tạo nhãn traveler_type.
    """
    _validate_dataframe(df, "Travel Ratings")
    df = df.copy()

    # Kaggle dataset dùng "Category 1" -> "Category 24" thay vì tên mô tả.
    # Mapping theo thứ tự gốc UCI: https://archive.ics.uci.edu/dataset/485
    CATEGORY_NAMES = {
        "Category 1": "Churches",
        "Category 2": "Resorts",
        "Category 3": "Beaches",
        "Category 4": "Parks",
        "Category 5": "Theatres",
        "Category 6": "Museums",
        "Category 7": "Malls",
        "Category 8": "Zoo",
        "Category 9": "Restaurants",
        "Category 10": "Pubs_Bars",
        "Category 11": "Local_Services",
        "Category 12": "Burger_Pizza",
        "Category 13": "Hotels",
        "Category 14": "Juice_Bars",
        "Category 15": "Art_Galleries",
        "Category 16": "Dance_Clubs",
        "Category 17": "Swimming_Pools",
        "Category 18": "Gyms",
        "Category 19": "Bakeries",
        "Category 20": "Beauty_Spas",
        "Category 21": "Cafes",
        "Category 22": "View_Points",
        "Category 23": "Monuments",
        "Category 24": "Gardens",
    }
    rename_map = {k: v for k, v in CATEGORY_NAMES.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("  Renamed %d category columns to descriptive names", len(rename_map))

    # Bỏ cột User Id nếu có (chỉ giữ nếu cần)
    category_cols = [c for c in df.columns if c not in ["User", "User Id", "Unnamed: 0"]]

    # Chuyển sang numeric
    for c in category_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fillna bằng 0 (chưa rating)
    df[category_cols] = df[category_cols].fillna(0)

    # Tạo nhãn traveler_type bằng rule-based (trước khi dùng ML)
    if len(category_cols) >= 10:
        # Lấy top category cho mỗi user
        df["top_category"] = df[category_cols].idxmax(axis=1)
        df["avg_rating"] = df[category_cols].mean(axis=1)
        df["rating_std"] = df[category_cols].std(axis=1)
        df["num_rated"] = (df[category_cols] > 0).sum(axis=1)

    logger.info("CLEAN Travel Ratings: %s rows, %d cols", f"{df.shape[0]:,}", df.shape[1])
    return df


def clean_hotel_bookings(df):
    """
    Làm sạch Hotel Booking Demand.
    - Xử lý missing, tạo total_stays, tạo labels cho budget/season.
    Columns: hotel, is_canceled, lead_time, arrival_date_year/month/week_number/day_of_month,
             stays_in_weekend_nights, stays_in_week_nights, adults, children, babies, meal,
             country, market_segment, distribution_channel, adr, customer_type, etc.
    """
    _validate_dataframe(df, "Hotel Bookings",
                        required_cols=["stays_in_weekend_nights", "stays_in_week_nights", "adr"])
    df = df.copy()

    # Xử lý missing
    df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0).astype(int)
    df["babies"] = df["babies"].fillna(0).astype(int)
    df["agent"] = df["agent"].replace("NULL", np.nan)
    df["company"] = df["company"].replace("NULL", np.nan)

    # Tạo tổng số đêm lưu trú
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    # Tổng số khách
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # ADR (Average Daily Rate) — loại bỏ giá trị âm hoặc cực đoan
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
    df.loc[df["adr"] < 0, "adr"] = np.nan
    df["adr"] = df["adr"].fillna(df["adr"].median())

    # Tổng chi phí ước tính
    df["total_cost"] = df["adr"] * df["total_nights"]

    # Phân loại budget
    df["budget_level"] = pd.cut(
        df["total_cost"],
        bins=[-1, 100, 300, 600, float("inf")],
        labels=["budget", "mid_range", "premium", "luxury"]
    )

    # Tạo cột arrival_date từ các cột năm/tháng/ngày
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["arrival_month_num"] = df["arrival_date_month"].map(month_map)
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_month_num"].astype(str) + "-" +
        df["arrival_date_day_of_month"].astype(str),
        errors="coerce"
    )

    # Season
    df["season"] = df["arrival_month_num"].map(
        lambda m: "spring" if m in [3, 4, 5]
        else "summer" if m in [6, 7, 8]
        else "autumn" if m in [9, 10, 11]
        else "winter"
    )

    # Loại bỏ bookings không có khách (lỗi dữ liệu)
    df = df[df["total_guests"] > 0]

    # Loại bỏ bookings 0 đêm (day-use, không phổ biến)
    df = df[df["total_nights"] > 0]

    logger.info("CLEAN Hotel Bookings: %s rows, %d cols", f"{df.shape[0]:,}", df.shape[1])
    return df


def clean_world_cities(df):
    """Làm sạch Worldwide Travel Cities."""
    _validate_dataframe(df, "World Cities")
    df = df.copy()

    # Chuẩn hoá cột numeric
    rating_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ["culture", "adventure", "nature", "beach",
                                     "nightlife", "cuisine", "wellness", "urban",
                                     "rating", "score"]
    )]
    for c in rating_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Chuẩn hoá lat/lng
    for c in ["Latitude", "Longitude", "latitude", "longitude", "lat", "lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    logger.info("CLEAN World Cities: %s rows, %d cols", f"{df.shape[0]:,}", df.shape[1])
    return df


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """Tính khoảng cách Haversine (km) giữa 2 toạ độ."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_distance_matrix():
    """
    Xây dựng ma trận khoảng cách giữa các điểm du lịch Việt Nam.
    Returns: (place_names, distance_matrix_np)
    """
    places = VN_TOURIST_PLACES
    names = list(places.keys())
    n = len(names)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(
                places[names[i]][0], places[names[i]][1],
                places[names[j]][0], places[names[j]][1],
            )
            dist[i][j] = round(d, 2)
            dist[j][i] = round(d, 2)
    logger.info("FEATURE Distance matrix: %dx%d places", n, n)
    return names, dist


def build_cost_matrix():
    """
    Xây dựng ma trận chi phí di chuyển ước tính (VND).
    Chi phí dựa trên TRAVEL_COST_PER_KM (mặc định 3,000 VND/km).
    """
    names, dist_km = build_distance_matrix()
    cost_matrix = dist_km * TRAVEL_COST_PER_KM
    logger.info("FEATURE Cost matrix: %dx%d places (VND)", len(names), len(names))
    return names, cost_matrix


def build_travel_time_matrix():
    """
    Xây dựng ma trận thời gian di chuyển ước tính (giờ).
    Tốc độ dựa trên TRAVEL_AVG_SPEED_KMH (mặc định 40 km/h).
    """
    names, dist_km = build_distance_matrix()
    time_matrix = dist_km / TRAVEL_AVG_SPEED_KMH
    logger.info("FEATURE Travel time matrix: %dx%d places (hours)", len(names), len(names))
    return names, time_matrix


def build_weather_probability_table(df_weather):
    """
    Tính bảng xác suất thời tiết cho Bayesian Network.
    P(rain | province, month), P(hot | province, month), etc.
    """
    if not all(c in df_weather.columns for c in ["province", "month", "is_rainy"]):
        logger.warning("Cần cột province, month, is_rainy. Bỏ qua.")
        return None

    # P(rain | province, month)
    rain_prob = df_weather.groupby(["province", "month"])["is_rainy"].mean().reset_index()
    rain_prob.columns = ["province", "month", "p_rain"]

    # P(outdoor_suitable | province, month)
    if "outdoor_suitable" in df_weather.columns:
        outdoor_prob = df_weather.groupby(["province", "month"])["outdoor_suitable"].mean().reset_index()
        outdoor_prob.columns = ["province", "month", "p_outdoor_ok"]
        rain_prob = rain_prob.merge(outdoor_prob, on=["province", "month"], how="left")

    # P(hot | province, month)
    if "is_hot" in df_weather.columns:
        hot_prob = df_weather.groupby(["province", "month"])["is_hot"].mean().reset_index()
        hot_prob.columns = ["province", "month", "p_hot"]
        rain_prob = rain_prob.merge(hot_prob, on=["province", "month"], how="left")

    # P(humid | province, month)
    if "is_humid" in df_weather.columns:
        humid_prob = df_weather.groupby(["province", "month"])["is_humid"].mean().reset_index()
        humid_prob.columns = ["province", "month", "p_humid"]
        rain_prob = rain_prob.merge(humid_prob, on=["province", "month"], how="left")

    logger.info("FEATURE Weather probability table: %d rows (province x month)", rain_prob.shape[0])
    return rain_prob


def build_places_dataframe(use_osm: bool = True) -> pd.DataFrame:
    """
    Tạo DataFrame các điểm du lịch Việt Nam.

    Args:
        use_osm: Nếu True, bổ sung điểm từ OSM JSON (data/osm/pbf_asia.json)
                 khi file tồn tại. Base VN_TOURIST_PLACES luôn được đưa vào.

    Returns:
        DataFrame với các cột: place_name, latitude, longitude, category,
        province, entry_fee_vnd, visit_duration_hours, opening_hour, closing_hour
    """
    # --- Build base DataFrame từ VN_TOURIST_PLACES (50 curated places) ---
    rows = []
    for name, info in VN_TOURIST_PLACES.items():
        lat, lng, category, province, entry_fee = info
        visit_hrs = DEFAULT_VISIT_HOURS.get(category, 3.0)
        open_h, close_h = DEFAULT_OPENING_HOURS.get(
            category, DEFAULT_OPENING_HOURS["_default"]
        )
        rows.append({
            "place_name":          name,
            "latitude":            lat,
            "longitude":           lng,
            "category":            category,
            "province":            province,
            "entry_fee_vnd":       entry_fee,
            "visit_duration_hours": visit_hrs,
            "opening_hour":        open_h,
            "closing_hour":        close_h,
        })
    base_df = pd.DataFrame(rows)
    logger.info("FEATURE Base places: %d (VN_TOURIST_PLACES)", len(base_df))

    # --- Bổ sung từ OSM JSON nếu có ---
    if use_osm:
        try:
            from modules.osm_loader import load_osm_places, merge_with_base
            osm_df = load_osm_places()
            base_df = merge_with_base(base_df, osm_df)
        except Exception as exc:
            logger.warning("OSM enrichment skipped: %s", exc)

    logger.info("FEATURE Places DataFrame: %d total places", len(base_df))
    return base_df


# ============================================================
# 5. SAVE PROCESSED DATA
# ============================================================

def save_cleaned(df, name):
    """Lưu DataFrame đã làm sạch."""
    ensure_dirs()
    path = os.path.join(CLEANED_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    logger.info("SAVE %s (%s rows)", path, f"{df.shape[0]:,}")


def save_features(arr, name):
    """Lưu feature array (.npy)."""
    ensure_dirs()
    path = os.path.join(FEATURES_DIR, f"{name}.npy")
    np.save(path, arr)
    logger.info("SAVE %s (shape: %s)", path, arr.shape)


def save_feature_csv(df, name):
    """Lưu feature DataFrame (.csv)."""
    ensure_dirs()
    path = os.path.join(FEATURES_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    logger.info("SAVE %s (%s rows)", path, f"{df.shape[0]:,}")


# ============================================================
# 6. OSM-BASED SUPPLEMENTARY FEATURES
# ============================================================

def build_restaurants_dataframe() -> Optional[pd.DataFrame]:
    """
    Tải dữ liệu nhà hàng Việt Nam từ OSM JSON (10,221 entries).

    Trả về DataFrame với các cột:
        name, latitude, longitude, province, cuisine,
        open_hour, close_hour, takeaway, outdoor_seating, wheelchair

    Trả về None nếu không tìm thấy file OSM.
    """
    try:
        from modules.osm_loader import load_osm_restaurants
        df = load_osm_restaurants()
        if df is not None and not df.empty:
            logger.info("FEATURE Restaurants: %d entries", len(df))
        return df
    except Exception as exc:
        logger.warning("build_restaurants_dataframe failed: %s", exc)
        return None


def build_hotels_dataframe() -> Optional[pd.DataFrame]:
    """
    Tải dữ liệu khách sạn Việt Nam từ OSM JSON (2,630 entries).

    Trả về DataFrame với các cột:
        name, latitude, longitude, province, property_type,
        star_rating, price_tier, estimated_price_vnd, wheelchair, internet_access

    Lưu ý: star_rating chỉ có dữ liệu cho ~40/2630 entries;
    estimated_price_vnd được ước lượng từ star_rating.

    Trả về None nếu không tìm thấy file OSM.
    """
    try:
        from modules.osm_loader import load_osm_hotels
        df = load_osm_hotels()
        if df is not None and not df.empty:
            logger.info("FEATURE Hotels: %d entries", len(df))
        return df
    except Exception as exc:
        logger.warning("build_hotels_dataframe failed: %s", exc)
        return None


# ============================================================
# 7. FULL PIPELINE
# ============================================================

def run_full_pipeline(skip_download=False):
    """
    Chạy toàn bộ pipeline: Download → Load → Clean → Features → Save.
    Returns dict chứa tất cả DataFrames và features.
    """
    ensure_dirs()
    result = {}

    # --- Download ---
    if not skip_download:
        logger.info("=" * 60)
        logger.info("PHASE 1: DOWNLOAD DATASETS")
        logger.info("=" * 60)
        download_all_datasets()

    # --- Load & Clean ---
    logger.info("=" * 60)
    logger.info("PHASE 2: LOAD & CLEAN DATA")
    logger.info("=" * 60)

    try:
        df_weather = clean_vietnam_weather(load_vietnam_weather())
        save_cleaned(df_weather, "vietnam_weather")
        result["weather"] = df_weather
    except Exception as e:
        logger.warning("SKIP Vietnam Weather: %s", e)

    try:
        df_reviews = clean_hotel_reviews(load_hotel_reviews())
        save_cleaned(df_reviews, "hotel_reviews")
        result["reviews"] = df_reviews
    except Exception as e:
        logger.warning("SKIP Hotel Reviews: %s", e)

    try:
        df_ratings = clean_travel_ratings(load_travel_ratings())
        save_cleaned(df_ratings, "travel_ratings")
        result["ratings"] = df_ratings
    except Exception as e:
        logger.warning("SKIP Travel Ratings: %s", e)

    try:
        df_bookings = clean_hotel_bookings(load_hotel_bookings())
        save_cleaned(df_bookings, "hotel_bookings")
        result["bookings"] = df_bookings
    except Exception as e:
        logger.warning("SKIP Hotel Bookings: %s", e)

    try:
        df_cities = clean_world_cities(load_world_cities())
        save_cleaned(df_cities, "world_cities")
        result["cities"] = df_cities
    except Exception as e:
        logger.warning("SKIP World Cities: %s", e)

    # --- Feature Engineering ---
    logger.info("=" * 60)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Distance matrix
    place_names, dist_matrix = build_distance_matrix()
    save_features(dist_matrix, "distance_matrix")
    result["distance_matrix"] = dist_matrix
    result["place_names"] = place_names

    # Cost matrix
    _, cost_matrix = build_cost_matrix()
    save_features(cost_matrix, "cost_matrix")
    result["cost_matrix"] = cost_matrix

    # Travel time matrix
    _, time_matrix = build_travel_time_matrix()
    save_features(time_matrix, "travel_time_matrix")
    result["time_matrix"] = time_matrix

    # Places DataFrame (base 50 + OSM enrichment)
    df_places = build_places_dataframe(use_osm=True)
    save_feature_csv(df_places, "vn_tourist_places")
    result["places"] = df_places

    # Restaurants from OSM
    df_restaurants = build_restaurants_dataframe()
    if df_restaurants is not None and not df_restaurants.empty:
        save_feature_csv(df_restaurants, "vn_restaurants")
        result["restaurants"] = df_restaurants

    # Hotels from OSM
    df_hotels = build_hotels_dataframe()
    if df_hotels is not None and not df_hotels.empty:
        save_feature_csv(df_hotels, "vn_hotels")
        result["hotels"] = df_hotels

    # Weather probability table
    if "weather" in result:
        weather_probs = build_weather_probability_table(result["weather"])
        if weather_probs is not None:
            save_feature_csv(weather_probs, "weather_probabilities")
            result["weather_probs"] = weather_probs

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    n_datasets = len([k for k in result if isinstance(result[k], pd.DataFrame)])
    logger.info("Datasets loaded: %d", n_datasets)
    logger.info("Feature matrices: distance, cost, travel_time (%d places)", len(place_names))
    if "restaurants" in result:
        logger.info("Restaurants: %d VN entries", len(result["restaurants"]))
    if "hotels" in result:
        logger.info("Hotels: %d VN entries", len(result["hotels"]))

    return result
