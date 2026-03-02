"""
data_pipeline.py — Module thu thập, làm sạch, và tiền xử lý dữ liệu
cho dự án AI Travel Planner & Recommender System.

Datasets:
  1. Vietnam Weather Data (181K records, 40 tỉnh, 2009-2021)
  2. 515K Hotel Reviews Europe
  3. Travel Review Ratings (UCI — 24 loại hình, 5456 users)
  4. Traveler Trip Data (~21K trips)
  5. Worldwide Travel Cities Ratings & Climate (560 cities)

Sử dụng trong Google Colab — tự động download từ Kaggle qua opendatasets.
"""

import os
import warnings
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings("ignore")

# ============================================================
# 0. CONSTANTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

KAGGLE_DATASETS = {
    "vietnam_weather": "vanviethieuanh/vietnam-weather-data",
    "hotel_reviews": "jiashenliu/515k-hotel-reviews-data-in-europe",
    "travel_ratings": "ishbhms/travel-review-ratings",
    "traveler_trips": "rkiattisak/traveler-trip-data",
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

# Toạ độ các điểm du lịch nổi tiếng Việt Nam (mở rộng)
VN_TOURIST_PLACES = {
    "Ha Long Bay": (20.9101, 107.1839, "nature", "Quang Ninh", 0),
    "Hoan Kiem Lake": (21.0288, 105.8525, "culture", "Ha Noi", 0),
    "Temple of Literature": (21.0275, 105.8360, "culture", "Ha Noi", 30000),
    "Ho Chi Minh Mausoleum": (21.0369, 105.8350, "culture", "Ha Noi", 0),
    "Old Quarter Hanoi": (21.0340, 105.8500, "culture", "Ha Noi", 0),
    "Imperial City Hue": (16.4698, 107.5786, "culture", "Hue", 200000),
    "Marble Mountains": (16.0034, 108.2628, "nature", "Da Nang", 40000),
    "Golden Bridge": (15.9940, 107.9969, "nature", "Da Nang", 900000),
    "My Son Sanctuary": (15.7644, 108.1241, "culture", "Quang Nam", 150000),
    "Hoi An Ancient Town": (15.8801, 108.3380, "culture", "Quang Nam", 120000),
    "Cu Chi Tunnels": (11.1415, 106.4627, "culture", "Ho Chi Minh", 110000),
    "Ben Thanh Market": (10.7725, 106.6980, "culture", "Ho Chi Minh", 0),
    "Notre Dame Cathedral HCMC": (10.7798, 106.6990, "culture", "Ho Chi Minh", 0),
    "War Remnants Museum": (10.7794, 106.6920, "culture", "Ho Chi Minh", 40000),
    "Nha Trang Beach": (12.2464, 109.1960, "beach", "Khanh Hoa", 0),
    "Vinpearl Nha Trang": (12.2167, 109.2340, "entertainment", "Khanh Hoa", 880000),
    "Po Nagar Towers": (12.2655, 109.1952, "culture", "Khanh Hoa", 22000),
    "Xuan Huong Lake": (11.9460, 108.4410, "nature", "Lam Dong", 0),
    "Crazy House Da Lat": (11.9363, 108.4310, "culture", "Lam Dong", 80000),
    "Valley of Love": (11.9660, 108.4390, "nature", "Lam Dong", 100000),
    "Mui Ne Sand Dunes": (10.9333, 108.2869, "nature", "Binh Thuan", 0),
    "Phu Quoc Beach": (10.2899, 103.9840, "beach", "Kien Giang", 0),
    "Sapa": (22.3363, 103.8438, "nature", "Lao Cai", 0),
    "Fansipan Summit": (22.3033, 103.7750, "adventure", "Lao Cai", 700000),
    "Trang An Landscape": (20.2500, 105.9000, "nature", "Ninh Binh", 200000),
    "Bai Dinh Pagoda": (20.2731, 105.8644, "culture", "Ninh Binh", 100000),
    "Phong Nha Cave": (17.5920, 106.2835, "nature", "Quang Binh", 150000),
    "Son Doong Cave": (17.5556, 106.1467, "adventure", "Quang Binh", 70000000),
    "Cat Ba Island": (20.7267, 107.0458, "nature", "Hai Phong", 0),
    "Ba Na Hills": (15.9975, 107.9964, "entertainment", "Da Nang", 900000),
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
            print(f"\n[DOWNLOAD] {name}: {url}")
            try:
                od.download(url, data_dir=RAW_DIR)
                print(f"  -> OK")
            except Exception as e:
                print(f"  -> ERROR: {e}")
    else:
        print("Dùng Kaggle CLI: kaggle datasets download -d <dataset_id>")
        for name, dataset_id in KAGGLE_DATASETS.items():
            cmd = f'kaggle datasets download -d {dataset_id} -p "{RAW_DIR}" --unzip'
            print(f"  {cmd}")
            os.system(cmd)

    print("\n[DONE] Tất cả datasets đã được tải về:", RAW_DIR)


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
    print(f"[LOAD] Vietnam Weather: {df.shape[0]:,} rows, {df.shape[1]} cols — {path}")
    return df


def load_hotel_reviews():
    """Load 515K Hotel Reviews."""
    path = find_csv(RAW_DIR, "hotel")
    if path is None:
        raise FileNotFoundError("Không tìm thấy Hotel Reviews CSV trong data/raw/")
    df = pd.read_csv(path)
    print(f"[LOAD] Hotel Reviews: {df.shape[0]:,} rows, {df.shape[1]} cols — {path}")
    return df


def load_travel_ratings():
    """Load Travel Review Ratings (UCI)."""
    path = find_csv(RAW_DIR, "google_review") or find_csv(RAW_DIR, "travel")
    if path is None:
        raise FileNotFoundError("Không tìm thấy Travel Ratings CSV trong data/raw/")
    df = pd.read_csv(path)
    print(f"[LOAD] Travel Ratings: {df.shape[0]:,} rows, {df.shape[1]} cols — {path}")
    return df


def load_traveler_trips():
    """Load Traveler Trip Data."""
    path = find_csv(RAW_DIR, "trip") or find_csv(RAW_DIR, "travel")
    if path is None:
        raise FileNotFoundError("Không tìm thấy Traveler Trips CSV trong data/raw/")
    df = pd.read_csv(path)
    print(f"[LOAD] Traveler Trips: {df.shape[0]:,} rows, {df.shape[1]} cols — {path}")
    return df


def load_world_cities():
    """Load Worldwide Travel Cities."""
    path = find_csv(RAW_DIR, "cities") or find_csv(RAW_DIR, "worldwide")
    if path is None:
        raise FileNotFoundError("Không tìm thấy World Cities CSV trong data/raw/")
    df = pd.read_csv(path)
    print(f"[LOAD] World Cities: {df.shape[0]:,} rows, {df.shape[1]} cols — {path}")
    return df


# ============================================================
# 3. CLEAN DATA
# ============================================================

def clean_vietnam_weather(df):
    """
    Làm sạch Vietnam Weather Data.
    - Chuẩn hoá cột, xử lý missing, tạo thêm cột thời gian.
    """
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

    # Tạo nhãn thời tiết cho IF-THEN rules
    if "rain_mm" in df.columns:
        df["is_rainy"] = (df["rain_mm"] > 1.0).astype(int)
        df["rain_level"] = pd.cut(
            df["rain_mm"], bins=[-1, 0, 5, 20, 50, 999],
            labels=["none", "light", "moderate", "heavy", "extreme"]
        )
    if "humidity" in df.columns:
        df["is_humid"] = (df["humidity"] > 80).astype(int)
    if "temp_max" in df.columns:
        df["is_hot"] = (df["temp_max"] > 35).astype(int)
    if all(c in df.columns for c in ["rain_mm", "temp_max", "humidity"]):
        df["outdoor_suitable"] = (
            (df["rain_mm"] <= 5) & (df["temp_max"] <= 38) & (df["humidity"] <= 90)
        ).astype(int)

    print(f"[CLEAN] Vietnam Weather: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def clean_hotel_reviews(df):
    """
    Làm sạch 515K Hotel Reviews.
    - Xử lý text, tạo nhãn sentiment, loại bỏ reviews trống.
    """
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

    print(f"[CLEAN] Hotel Reviews: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def clean_travel_ratings(df):
    """
    Làm sạch Travel Review Ratings (UCI).
    - 24 cột rating, chuẩn hoá, tạo nhãn traveler_type.
    """
    df = df.copy()

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

    print(f"[CLEAN] Travel Ratings: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def clean_traveler_trips(df):
    """
    Làm sạch Traveler Trip Data.
    - Parse dates, chuẩn hoá costs, tạo labels.
    """
    df = df.copy()

    # Parse dates
    for col in ["Start date", "End date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure Duration numeric
    if "Duration (days)" in df.columns:
        df["Duration (days)"] = pd.to_numeric(df["Duration (days)"], errors="coerce")

    # Ensure costs numeric
    for col in ["Accommodation cost", "Transportation cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce"
            )

    # Tổng chi phí
    cost_cols = [c for c in ["Accommodation cost", "Transportation cost"] if c in df.columns]
    if cost_cols:
        df["total_cost"] = df[cost_cols].sum(axis=1)
        df["budget_level"] = pd.cut(
            df["total_cost"],
            bins=[0, 500, 1500, 5000, float("inf")],
            labels=["budget", "mid_range", "premium", "luxury"]
        )

    # Cost per day
    if "total_cost" in df.columns and "Duration (days)" in df.columns:
        df["cost_per_day"] = df["total_cost"] / df["Duration (days)"].replace(0, np.nan)

    # Drop rows không hợp lệ
    df = df.dropna(subset=[c for c in cost_cols if c in df.columns])

    print(f"[CLEAN] Traveler Trips: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def clean_world_cities(df):
    """Làm sạch Worldwide Travel Cities."""
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

    print(f"[CLEAN] World Cities: {df.shape[0]:,} rows, {df.shape[1]} cols")
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
    print(f"[FEATURE] Distance matrix: {n}x{n} places")
    return names, dist


def build_cost_matrix():
    """
    Xây dựng ma trận chi phí di chuyển ước tính (VND).
    Giả sử chi phí ~3,000 VND/km (xe khách/taxi trung bình).
    """
    names, dist_km = build_distance_matrix()
    cost_per_km = 3000  # VND
    cost_matrix = dist_km * cost_per_km
    print(f"[FEATURE] Cost matrix: {len(names)}x{len(names)} places (VND)")
    return names, cost_matrix


def build_travel_time_matrix():
    """
    Xây dựng ma trận thời gian di chuyển ước tính (giờ).
    Giả sử tốc độ trung bình 40km/h (đường Việt Nam).
    """
    names, dist_km = build_distance_matrix()
    avg_speed = 40  # km/h
    time_matrix = dist_km / avg_speed
    print(f"[FEATURE] Travel time matrix: {len(names)}x{len(names)} places (hours)")
    return names, time_matrix


def build_weather_probability_table(df_weather):
    """
    Tính bảng xác suất thời tiết cho Bayesian Network.
    P(rain | province, month), P(hot | province, month), etc.
    """
    if not all(c in df_weather.columns for c in ["province", "month", "is_rainy"]):
        print("[WARNING] Cần cột province, month, is_rainy. Bỏ qua.")
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

    print(f"[FEATURE] Weather probability table: {rain_prob.shape[0]} rows (province x month)")
    return rain_prob


def build_places_dataframe():
    """
    Tạo DataFrame các điểm du lịch Việt Nam với đầy đủ thông tin.
    """
    rows = []
    for name, info in VN_TOURIST_PLACES.items():
        lat, lng, category, province, entry_fee = info
        rows.append({
            "place_name": name,
            "latitude": lat,
            "longitude": lng,
            "category": category,
            "province": province,
            "entry_fee_vnd": entry_fee,
            "visit_duration_hours": 2.0 if category in ["culture", "entertainment"] else 3.0,
            "opening_hour": 7 if category != "entertainment" else 9,
            "closing_hour": 17 if category != "entertainment" else 22,
        })
    df = pd.DataFrame(rows)
    print(f"[FEATURE] Places DataFrame: {df.shape[0]} places")
    return df


# ============================================================
# 5. SAVE PROCESSED DATA
# ============================================================

def save_cleaned(df, name):
    """Lưu DataFrame đã làm sạch."""
    ensure_dirs()
    path = os.path.join(CLEANED_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} ({df.shape[0]:,} rows)")


def save_features(arr, name):
    """Lưu feature array (.npy)."""
    ensure_dirs()
    path = os.path.join(FEATURES_DIR, f"{name}.npy")
    np.save(path, arr)
    print(f"[SAVE] {path} (shape: {arr.shape})")


def save_feature_csv(df, name):
    """Lưu feature DataFrame (.csv)."""
    ensure_dirs()
    path = os.path.join(FEATURES_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[SAVE] {path} ({df.shape[0]:,} rows)")


# ============================================================
# 6. FULL PIPELINE
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
        print("=" * 60)
        print("PHASE 1: DOWNLOAD DATASETS")
        print("=" * 60)
        download_all_datasets()

    # --- Load & Clean ---
    print("\n" + "=" * 60)
    print("PHASE 2: LOAD & CLEAN DATA")
    print("=" * 60)

    try:
        df_weather = clean_vietnam_weather(load_vietnam_weather())
        save_cleaned(df_weather, "vietnam_weather")
        result["weather"] = df_weather
    except Exception as e:
        print(f"[SKIP] Vietnam Weather: {e}")

    try:
        df_reviews = clean_hotel_reviews(load_hotel_reviews())
        save_cleaned(df_reviews, "hotel_reviews")
        result["reviews"] = df_reviews
    except Exception as e:
        print(f"[SKIP] Hotel Reviews: {e}")

    try:
        df_ratings = clean_travel_ratings(load_travel_ratings())
        save_cleaned(df_ratings, "travel_ratings")
        result["ratings"] = df_ratings
    except Exception as e:
        print(f"[SKIP] Travel Ratings: {e}")

    try:
        df_trips = clean_traveler_trips(load_traveler_trips())
        save_cleaned(df_trips, "traveler_trips")
        result["trips"] = df_trips
    except Exception as e:
        print(f"[SKIP] Traveler Trips: {e}")

    try:
        df_cities = clean_world_cities(load_world_cities())
        save_cleaned(df_cities, "world_cities")
        result["cities"] = df_cities
    except Exception as e:
        print(f"[SKIP] World Cities: {e}")

    # --- Feature Engineering ---
    print("\n" + "=" * 60)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 60)

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

    # Places DataFrame
    df_places = build_places_dataframe()
    save_feature_csv(df_places, "vn_tourist_places")
    result["places"] = df_places

    # Weather probability table
    if "weather" in result:
        weather_probs = build_weather_probability_table(result["weather"])
        if weather_probs is not None:
            save_feature_csv(weather_probs, "weather_probabilities")
            result["weather_probs"] = weather_probs

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Datasets loaded: {len([k for k in result if isinstance(result[k], pd.DataFrame)])}")
    print(f"Feature matrices: distance, cost, travel_time ({len(place_names)} places)")

    return result
