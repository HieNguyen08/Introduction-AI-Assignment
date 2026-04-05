"""
bayesian_net.py — Thành phần (D): Mạng Bayes cho dự đoán xác suất.

Module này xây dựng Bayesian Network cho:
  1. P(rain | province, month) — Xác suất mưa theo tỉnh và tháng
  2. P(outdoor_suitable | province, month) — Xác suất thời tiết phù hợp outdoor
  3. P(user_like | category, weather, group_type) — Xác suất user thích địa điểm
  4. Kết hợp với Knowledge Base (C) để đưa ra đề xuất
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


# ============================================================
# 0. CONSTANTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")

# Prior probabilities (mặc định khi không có dữ liệu)
DEFAULT_P_RAIN = 0.35  # Trung bình VN ~35% ngày mưa
DEFAULT_P_OUTDOOR = 0.60
DEFAULT_P_HOT = 0.20
DEFAULT_P_HUMID = 0.40

# Mapping từ tên tỉnh/thành (trong places data) → tên trạm khí tượng (trong weather data)
# Weather data dùng tên thành phố/trạm đo, không phải tên tỉnh
PROVINCE_TO_STATION: Dict[str, str] = {
    # Tỉnh/thành → trạm khí tượng gần nhất có trong data
    "da nang": "tam ky",          # Tam Ky (Quang Nam) gần nhất; Da Nang không có trong data
    "khanh hoa": "nha trang",     # Nha Trang là thủ phủ Khánh Hòa
    "lam dong": "da lat",         # Da Lat là thủ phủ Lâm Đồng
    "quang ninh": "hong gai",     # Hạ Long / Hồng Gai là thủ phủ Quảng Ninh
    "binh thuan": "phan thiet",   # Phan Thiết là thủ phủ Bình Thuận
    "binh dinh": "qui nhon",      # Quy Nhơn là thủ phủ Bình Định
    "kien giang": "rach gia",     # Rạch Giá là thủ phủ Kiên Giang
    "quang nam": "tam ky",        # Tam Kỳ là thủ phủ Quảng Nam
    "ba ria vung tau": "vung tau",
    "dong nai": "bien hoa",
    "long an": "tan an",
    "tien giang": "my tho",
    "an giang": "chau doc",
    "phu yen": "tuy hoa",
    "dak lak": "buon me thuot",
    "gia lai": "play cu",
    "ninh thuan": "phan rang",
    "quang binh": "hue",          # Đồng Hới không có data → fallback Hue
    "quang tri": "hue",           # Đông Hà không có data → fallback Hue
    "thua thien hue": "hue",
    "ninh binh": "nam dinh",      # Nam Định gần nhất
    "lao cai": "yen bai",         # Yên Bái gần nhất có data
    "ha giang": "yen bai",        # Yên Bái gần nhất có data
    "cao bang": "thai nguyen",    # Thái Nguyên gần nhất
    "son la": "hoa binh",         # Hòa Bình gần nhất
    "phu tho": "viet tri",
    "thai nguyen": "thai nguyen",
    "yen bai": "yen bai",
    "hoa binh": "hoa binh",
    "hai duong": "hai duong",
    "nam dinh": "nam dinh",
    "cam ranh": "nha trang",      # Cam Ranh nằm trong Khánh Hòa
}

# Trọng số cho user preference model
CATEGORY_AFFINITY = {
    # group_type -> category -> base_probability
    "solo": {
        "culture": 0.7, "nature": 0.8, "beach": 0.6,
        "adventure": 0.9, "entertainment": 0.5,
    },
    "couple": {
        "culture": 0.7, "nature": 0.9, "beach": 0.9,
        "adventure": 0.5, "entertainment": 0.7,
    },
    "family": {
        "culture": 0.8, "nature": 0.8, "beach": 0.9,
        "adventure": 0.3, "entertainment": 0.9,
    },
    "friends": {
        "culture": 0.5, "nature": 0.7, "beach": 0.8,
        "adventure": 0.9, "entertainment": 0.8,
    },
}


# ============================================================
# 1. BAYESIAN NETWORK NODE
# ============================================================

class BayesNode:
    """
    Node trong Bayesian Network.
    Mỗi node đại diện cho một biến ngẫu nhiên.
    """

    def __init__(self, name: str, parents: Optional[List[str]] = None):
        """
        Args:
            name: Tên biến (e.g., "rain", "outdoor_suitable")
            parents: Danh sách tên node cha
        """
        self.name = name
        self.parents = parents or []
        self.cpt: Dict[tuple, float] = {}  # Conditional Probability Table
        self._default_prob = 0.5

    def set_cpt(self, cpt: Dict[tuple, float]):
        """
        Thiết lập bảng xác suất có điều kiện (CPT).

        Args:
            cpt: Dict mapping (parent_value_1, ..., parent_value_n) -> probability
                 Nếu không có parent, dùng key là tuple rỗng ()
        """
        self.cpt = cpt

    def set_default(self, prob: float):
        """Thiết lập xác suất mặc định."""
        self._default_prob = prob

    def get_probability(self, *parent_values) -> float:
        """
        Truy vấn xác suất P(node=True | parents).

        Args:
            parent_values: Giá trị của các node cha theo thứ tự

        Returns:
            float: Xác suất P(node=True | parent_values)
        """
        key = tuple(parent_values)
        return self.cpt.get(key, self._default_prob)

    def __repr__(self):
        return f"BayesNode({self.name}, parents={self.parents}, cpt_size={len(self.cpt)})"


# ============================================================
# 2. BAYESIAN NETWORK
# ============================================================

class BayesianNetwork:
    """
    Mạng Bayes cho dự đoán thời tiết và sở thích du lịch.

    Cấu trúc mạng:
        Province, Month → Rain → Outdoor_Suitable
        Province, Month → Hot
        Province, Month → Humid
        Category, Weather, Group_Type → User_Like
    """

    def __init__(self):
        self.nodes: Dict[str, BayesNode] = {}
        self.weather_probs_df: Optional[pd.DataFrame] = None
        self._built = False

    def add_node(self, node: BayesNode):
        """Thêm node vào mạng."""
        self.nodes[node.name] = node

    def build_from_data(self, weather_probs_df: Optional[pd.DataFrame] = None,
                         weather_df: Optional[pd.DataFrame] = None):
        """
        Xây dựng mạng Bayes từ dữ liệu.

        Args:
            weather_probs_df: DataFrame bảng xác suất thời tiết
                              (từ build_weather_probability_table())
            weather_df: DataFrame dữ liệu thời tiết đã clean
                        (từ clean_vietnam_weather())
        """
        # Tải dữ liệu nếu chưa cung cấp
        if weather_probs_df is None:
            weather_probs_df = self._load_weather_probs()
        if weather_df is None:
            weather_df = self._load_weather_data()

        self.weather_probs_df = weather_probs_df

        # --- Node 1: Rain ---
        rain_node = BayesNode("rain", parents=["province", "month"])
        rain_node.set_default(DEFAULT_P_RAIN)
        if weather_probs_df is not None:
            cpt = {}
            for _, row in weather_probs_df.iterrows():
                key = (str(row["province"]), int(row["month"]))
                cpt[key] = float(row["p_rain"])
            rain_node.set_cpt(cpt)
        self.add_node(rain_node)

        # --- Node 2: Outdoor Suitable ---
        outdoor_node = BayesNode("outdoor_suitable", parents=["province", "month"])
        outdoor_node.set_default(DEFAULT_P_OUTDOOR)
        if weather_probs_df is not None and "p_outdoor_ok" in weather_probs_df.columns:
            cpt = {}
            for _, row in weather_probs_df.iterrows():
                key = (str(row["province"]), int(row["month"]))
                cpt[key] = float(row["p_outdoor_ok"])
            outdoor_node.set_cpt(cpt)
        self.add_node(outdoor_node)

        # --- Node 3: Hot ---
        hot_node = BayesNode("hot", parents=["province", "month"])
        hot_node.set_default(DEFAULT_P_HOT)
        if weather_probs_df is not None and "p_hot" in weather_probs_df.columns:
            cpt = {}
            for _, row in weather_probs_df.iterrows():
                key = (str(row["province"]), int(row["month"]))
                cpt[key] = float(row["p_hot"])
            hot_node.set_cpt(cpt)
        self.add_node(hot_node)

        # --- Node 4: Humid ---
        humid_node = BayesNode("humid", parents=["province", "month"])
        humid_node.set_default(DEFAULT_P_HUMID)
        if weather_probs_df is not None and "p_humid" in weather_probs_df.columns:
            cpt = {}
            for _, row in weather_probs_df.iterrows():
                key = (str(row["province"]), int(row["month"]))
                cpt[key] = float(row["p_humid"])
            humid_node.set_cpt(cpt)
        self.add_node(humid_node)

        # --- Node 5: User Preference (category affinity) ---
        user_pref_node = BayesNode("user_like", parents=["category", "group_type", "is_rain"])
        user_pref_node.set_default(0.5)
        # Build CPT from category affinity
        cpt = {}
        for group_type, cats in CATEGORY_AFFINITY.items():
            for cat, base_prob in cats.items():
                # Không mưa → giữ nguyên base_prob
                cpt[(cat, group_type, False)] = base_prob
                # Mưa → giảm xác suất tất cả (outdoor giảm mạnh, indoor giảm nhẹ)
                # Rain makes ALL activities less enjoyable; outdoor more severely.
                if cat in {"nature", "beach", "adventure"}:
                    cpt[(cat, group_type, True)] = base_prob * 0.5
                else:
                    cpt[(cat, group_type, True)] = base_prob * 0.9
        user_pref_node.set_cpt(cpt)
        self.add_node(user_pref_node)

        self._built = True
        print(f"[BAYES NET] Xây dựng thành công: {len(self.nodes)} nodes")
        for name, node in self.nodes.items():
            print(f"  → {name}: parents={node.parents}, CPT size={len(node.cpt)}")

    def _load_weather_probs(self) -> Optional[pd.DataFrame]:
        """Tải bảng xác suất thời tiết từ file."""
        path = os.path.join(FEATURES_DIR, "weather_probabilities.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[LOAD] Weather probabilities: {df.shape[0]} rows — {path}")
            return df
        print(f"[WARNING] Không tìm thấy {path}. Sẽ dùng prior mặc định.")
        return None

    def _load_weather_data(self) -> Optional[pd.DataFrame]:
        """Tải dữ liệu thời tiết đã clean."""
        path = os.path.join(CLEANED_DIR, "vietnam_weather.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[LOAD] Vietnam weather: {df.shape[0]:,} rows — {path}")
            return df
        return None

    # ============================================================
    # 3. QUERY / INFERENCE
    # ============================================================

    def _find_weather_rows(self, province: str, month: int) -> "pd.DataFrame":
        """
        Tìm hàng thời tiết theo tỉnh và tháng.
        Ưu tiên: (1) khớp chính xác, (2) mapping trạm, (3) substring, (4) rỗng.
        """
        if self.weather_probs_df is None:
            return pd.DataFrame()
        p_lower = province.lower().strip()
        month_mask = self.weather_probs_df["month"] == month
        # 1. Khớp chính xác trước
        exact = self.weather_probs_df[
            (self.weather_probs_df["province"].str.lower() == p_lower) & month_mask
        ]
        if len(exact) > 0:
            return exact
        # 2. Dùng mapping tỉnh → trạm khí tượng
        station = PROVINCE_TO_STATION.get(p_lower)
        if station:
            mapped = self.weather_probs_df[
                (self.weather_probs_df["province"].str.lower() == station) & month_mask
            ]
            if len(mapped) > 0:
                return mapped
        # 3. Fallback: khớp substring (regex=False để tránh lỗi ký tự đặc biệt)
        substr = self.weather_probs_df[
            self.weather_probs_df["province"].str.lower().str.contains(
                p_lower, na=False, regex=False
            ) & month_mask
        ]
        if len(substr) > 0:
            return substr
        return pd.DataFrame()

    def query_rain(self, province: str, month: int) -> float:
        """
        Truy vấn P(rain | province, month).

        Args:
            province: Tên tỉnh (e.g., "Ha Noi", "Da Nang")
            month: Tháng (1-12)

        Returns:
            float: Xác suất mưa (0-1)
        """
        if "rain" not in self.nodes:
            return DEFAULT_P_RAIN

        # Thử tìm chính xác
        prob = self.nodes["rain"].get_probability(province, month)

        # Nếu không tìm thấy, thử tìm gần đúng
        if prob == self.nodes["rain"]._default_prob:
            matches = self._find_weather_rows(province, month)
            if len(matches) > 0:
                prob = float(matches.iloc[0]["p_rain"])

        return prob

    def query_outdoor(self, province: str, month: int) -> float:
        """Truy vấn P(outdoor_suitable | province, month)."""
        if "outdoor_suitable" not in self.nodes:
            return DEFAULT_P_OUTDOOR

        prob = self.nodes["outdoor_suitable"].get_probability(province, month)

        if prob == self.nodes["outdoor_suitable"]._default_prob:
            matches = self._find_weather_rows(province, month)
            if len(matches) > 0 and "p_outdoor_ok" in matches.columns:
                prob = float(matches.iloc[0]["p_outdoor_ok"])

        return prob

    def query_hot(self, province: str, month: int) -> float:
        """Truy vấn P(hot | province, month)."""
        if "hot" not in self.nodes:
            return DEFAULT_P_HOT
        prob = self.nodes["hot"].get_probability(province, month)
        if prob == self.nodes["hot"]._default_prob:
            matches = self._find_weather_rows(province, month)
            if len(matches) > 0 and "p_hot" in matches.columns:
                prob = float(matches.iloc[0]["p_hot"])
        return prob

    def query_humid(self, province: str, month: int) -> float:
        """Truy vấn P(humid | province, month)."""
        if "humid" not in self.nodes:
            return DEFAULT_P_HUMID
        prob = self.nodes["humid"].get_probability(province, month)
        if prob == self.nodes["humid"]._default_prob:
            matches = self._find_weather_rows(province, month)
            if len(matches) > 0 and "p_humid" in matches.columns:
                prob = float(matches.iloc[0]["p_humid"])
        return prob

    def query_weather_full(self, province: str, month: int) -> Dict[str, float]:
        """
        Truy vấn đầy đủ tất cả biến thời tiết.

        Returns:
            Dict chứa: p_rain, p_outdoor, p_hot, p_humid
        """
        return {
            "p_rain": self.query_rain(province, month),
            "p_outdoor": self.query_outdoor(province, month),
            "p_hot": self.query_hot(province, month),
            "p_humid": self.query_humid(province, month),
        }

    def query_user_preference(self, category: str, group_type: str,
                               is_rain: bool) -> float:
        """
        Truy vấn P(user_like | category, group_type, is_rain).

        Args:
            category: Loại hình địa điểm ("culture", "nature", "beach", "adventure", "entertainment")
            group_type: Nhóm du khách ("solo", "couple", "family", "friends")
            is_rain: Có mưa không

        Returns:
            float: Xác suất user thích (0-1)
        """
        if "user_like" not in self.nodes:
            return CATEGORY_AFFINITY.get(group_type, {}).get(category, 0.5)
        return self.nodes["user_like"].get_probability(category, group_type, is_rain)

    # ============================================================
    # 4. PLACE SCORING
    # ============================================================

    def score_places(self, places_df: pd.DataFrame,
                     month: int, group_type: str = "solo",
                     verbose: bool = True) -> pd.DataFrame:
        """
        Tính điểm Bayesian cho từng địa điểm du lịch.

        Điểm = P(outdoor_ok | province, month) × P(user_like | category, group, rain)

        Args:
            places_df: DataFrame các điểm du lịch (cần cột: province, category)
            month: Tháng du lịch (1-12)
            group_type: Nhóm du khách

        Returns:
            DataFrame với thêm cột bayesian_score, p_rain, p_outdoor, p_user_like
        """
        df = places_df.copy()

        scores = []
        p_rains = []
        p_outdoors = []
        p_user_likes = []

        for _, row in df.iterrows():
            province = row.get("province", "Ha Noi")
            category = row.get("category", "culture")

            # Truy vấn thời tiết
            p_rain = self.query_rain(province, month)
            p_outdoor = self.query_outdoor(province, month)

            # Truy vấn sở thích user
            # Dùng expected value: P(like) = P(like|rain)P(rain) + P(like|no_rain)P(no_rain)
            p_like_rain = self.query_user_preference(category, group_type, True)
            p_like_no_rain = self.query_user_preference(category, group_type, False)
            p_user_like = p_like_rain * p_rain + p_like_no_rain * (1 - p_rain)

            # Tính Bayesian score
            # Score = P(outdoor_ok) × P(user_like) (cho outdoor)
            # Score = P(user_like) (cho indoor — ít bị ảnh hưởng bởi thời tiết)
            if category in {"nature", "beach", "adventure"}:
                bayesian_score = p_outdoor * p_user_like
            else:
                bayesian_score = p_user_like * 0.9 + 0.1  # Indoor luôn có bonus

            p_rains.append(round(p_rain, 3))
            p_outdoors.append(round(p_outdoor, 3))
            p_user_likes.append(round(p_user_like, 3))
            scores.append(round(bayesian_score, 3))

        df["p_rain"] = p_rains
        df["p_outdoor"] = p_outdoors
        df["p_user_like"] = p_user_likes
        df["bayesian_score"] = scores

        # Sắp xếp theo score giảm dần
        df = df.sort_values("bayesian_score", ascending=False).reset_index(drop=True)

        if verbose:
            print(f"\n📊 BAYESIAN SCORING — Tháng {month}, Nhóm: {group_type}")
            print(f"{'=' * 80}")
            display_cols = ["place_name", "category", "province",
                           "p_rain", "p_outdoor", "p_user_like", "bayesian_score"]
            display_cols = [c for c in display_cols if c in df.columns]
            print(df[display_cols].head(15).to_string(index=False))

        return df

    def predict_best_month(self, province: str, category: str = "nature",
                            group_type: str = "solo") -> pd.DataFrame:
        """
        Dự đoán tháng tốt nhất để du lịch ở một tỉnh.

        Returns:
            DataFrame: month, p_rain, p_outdoor, p_user_like, score
        """
        results = []
        for month in range(1, 13):
            weather = self.query_weather_full(province, month)
            p_like_rain = self.query_user_preference(category, group_type, True)
            p_like_no_rain = self.query_user_preference(category, group_type, False)
            p_user_like = (p_like_rain * weather["p_rain"] +
                          p_like_no_rain * (1 - weather["p_rain"]))

            if category in {"nature", "beach", "adventure"}:
                score = weather["p_outdoor"] * p_user_like
            else:
                score = p_user_like * 0.9 + 0.1

            results.append({
                "month": month,
                "p_rain": round(weather["p_rain"], 3),
                "p_outdoor": round(weather["p_outdoor"], 3),
                "p_hot": round(weather["p_hot"], 3),
                "p_humid": round(weather["p_humid"], 3),
                "p_user_like": round(p_user_like, 3),
                "score": round(score, 3),
            })

        df = pd.DataFrame(results)
        best = df.loc[df["score"].idxmax()]
        print(f"\n🏆 Tháng tốt nhất cho {category} tại {province}: "
              f"Tháng {int(best['month'])} (score={best['score']:.3f})")

        return df

    # ============================================================
    # 5. NETWORK VISUALIZATION
    # ============================================================

    def get_network_structure(self) -> Dict[str, List[str]]:
        """Trả về cấu trúc mạng dưới dạng adjacency list."""
        structure = {}
        for name, node in self.nodes.items():
            structure[name] = node.parents
        return structure

    def print_network(self):
        """In cấu trúc mạng Bayes."""
        print("\n🕸️ CẤU TRÚC MẠNG BAYES:")
        print("=" * 50)
        for name, node in self.nodes.items():
            if node.parents:
                parents_str = ", ".join(node.parents)
                print(f"  {parents_str} → {name}")
            else:
                print(f"  {name} (root node)")
        print()

        # Print summary
        print("📊 Tóm tắt:")
        print(f"  Số nodes: {len(self.nodes)}")
        total_cpt = sum(len(n.cpt) for n in self.nodes.values())
        print(f"  Tổng CPT entries: {total_cpt}")


# ============================================================
# 6. INTEGRATION WITH KNOWLEDGE BASE
# ============================================================

def integrate_bayes_kb(
    places_df: pd.DataFrame,
    province: str,
    month: int,
    group_type: str = "solo",
    budget_vnd: float = 2_000_000,
    user_preferences: Optional[List[str]] = None,
    num_days: int = 3,
    current_hour: Optional[int] = None,
    weather_probs_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tích hợp Bayesian Network (D) + Knowledge Base (C) để lọc và xếp hạng địa điểm.

    Pipeline:
        1. Bayes dự đoán thời tiết → P(rain|province,month)
        2. IF-THEN rules lọc → loại bỏ điểm không phù hợp
        3. Bayes scoring → xếp hạng điểm còn lại

    Args:
        places_df: DataFrame các điểm du lịch
        province: Tỉnh du lịch chính
        month: Tháng (1-12)
        group_type: "solo" | "couple" | "family" | "friends"
        budget_vnd: Ngân sách (VND)
        user_preferences: Danh sách sở thích
        num_days: Số ngày
        current_hour: Giờ hiện tại
        weather_probs_df: Bảng xác suất thời tiết (optional)

    Returns:
        (scored_places_df, metadata) — DataFrame + metadata
    """
    from modules.knowledge_base import KnowledgeBase, create_context

    if verbose:
        print("=" * 70)
        print("🔗 TÍCH HỢP BAYESIAN NETWORK (D) + KNOWLEDGE BASE (C)")
        print("=" * 70)

    # --- Lọc địa điểm theo tỉnh được chọn ---
    if "province" in places_df.columns:
        p_lower = province.lower()
        province_mask = places_df["province"].str.lower() == p_lower
        province_places = places_df[province_mask]
        if len(province_places) >= 2:
            places_df = province_places.reset_index(drop=True)
            if verbose:
                print(f"\n  📍 Lọc theo tỉnh '{province}': {len(places_df)} địa điểm")
        else:
            if verbose:
                print(
                    f"\n  ⚠️ Chỉ có {len(province_places)} địa điểm tại '{province}' "
                    f"— dùng toàn bộ {len(places_df)} điểm"
                )

    # --- Bước 1: Xây dựng Bayesian Network & dự đoán thời tiết ---
    if verbose:
        print("\n📊 Bước 1: Bayesian Network — Dự đoán thời tiết")
        print("-" * 50)

    bn = BayesianNetwork()
    bn.build_from_data(weather_probs_df=weather_probs_df)
    weather = bn.query_weather_full(province, month)

    if verbose:
        print(f"\n  Tỉnh: {province}, Tháng: {month}")
        print(f"  P(rain)    = {weather['p_rain']:.3f}")
        print(f"  P(outdoor) = {weather['p_outdoor']:.3f}")
        print(f"  P(hot)     = {weather['p_hot']:.3f}")
        print(f"  P(humid)   = {weather['p_humid']:.3f}")

    # Chuyển đổi xác suất thành weather context cho KB
    # Sử dụng expected values từ Bayes
    is_likely_rain = weather["p_rain"] > 0.5
    expected_rain_mm = weather["p_rain"] * 15  # Ước lượng lượng mưa trung bình
    outdoor_suitable = weather["p_outdoor"] > 0.5

    # Quyết định season dựa trên month
    if month in [2, 3, 4]:
        season = "spring"
    elif month in [5, 6, 7]:
        season = "summer"
    elif month in [8, 9, 10]:
        season = "autumn"
    else:
        season = "winter"

    # --- Bước 2: Knowledge Base — Áp dụng luật IF-THEN ---
    if verbose:
        print(f"\n🧠 Bước 2: Knowledge Base — Áp dụng luật IF-THEN")
        print("-" * 50)

    ctx = create_context(
        rain_mm=expected_rain_mm,
        temp_max=35 if weather["p_hot"] > 0.5 else 30,
        temp_min=12 if season == "winter" and province in ["Lao Cai", "Ha Noi"] else 22,
        humidity=90 if weather["p_humid"] > 0.5 else 70,
        wind_speed=10,
        budget_vnd=budget_vnd,
        group_type=group_type,
        season=season,
        outdoor_suitable=outdoor_suitable,
        user_preferences=user_preferences or [],
        num_days=num_days,
        current_hour=current_hour,
        current_province=province,
    )

    kb = KnowledgeBase()
    filtered_df, firing_log = kb.infer(ctx, places_df, verbose=verbose)

    # --- Bước 3: Bayesian Scoring — Xếp hạng ---
    if verbose:
        print(f"\n📈 Bước 3: Bayesian Scoring — Xếp hạng")
        print("-" * 50)

    if len(filtered_df) > 0:
        scored_df = bn.score_places(filtered_df, month, group_type, verbose=verbose)

        # Kết hợp score: bayesian_score × weather_score × preference_score
        if "weather_score" in scored_df.columns:
            scored_df["final_score"] = (
                scored_df["bayesian_score"] *
                scored_df["weather_score"]
            )
        else:
            scored_df["final_score"] = scored_df["bayesian_score"]

        if "preference_score" in scored_df.columns:
            scored_df["final_score"] *= scored_df["preference_score"]

        scored_df = scored_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    else:
        scored_df = filtered_df

    # Metadata
    metadata = {
        "province": province,
        "month": month,
        "group_type": group_type,
        "budget_vnd": budget_vnd,
        "weather_prediction": weather,
        "rules_fired": len(firing_log),
        "places_initial": len(places_df),
        "places_after_rules": len(filtered_df),
        "places_final": len(scored_df),
        "firing_log": firing_log,
        "season": season,
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"📋 TỔNG KẾT:")
        print(f"  Tỉnh: {province} | Tháng: {month} | Nhóm: {group_type}")
        print(f"  Thời tiết: P(mưa)={weather['p_rain']:.1%}, "
              f"P(outdoor)={weather['p_outdoor']:.1%}")
        print(f"  Điểm: {len(places_df)} → {len(filtered_df)} (sau rules) "
              f"→ {len(scored_df)} (xếp hạng)")
        print(f"  Luật kích hoạt: {len(firing_log)}")
        if len(scored_df) > 0:
            display_cols = ["place_name", "category", "province", "bayesian_score"]
            if "final_score" in scored_df.columns:
                display_cols.append("final_score")
            display_cols = [c for c in display_cols if c in scored_df.columns]
            print(f"\n🏆 Top 5 địa điểm gợi ý:")
            print(scored_df[display_cols].head(5).to_string(index=False))
        print(f"{'=' * 70}")

    return scored_df, metadata


# ============================================================
# 7. DEMO / TEST
# ============================================================

def demo_bayesian_network():
    """Demo Bayesian Network với các truy vấn mẫu."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from modules.data_pipeline import build_places_dataframe, build_weather_probability_table
    except ImportError:
        print("Cần chạy data_pipeline trước.")
        return

    # Xây dựng mạng
    bn = BayesianNetwork()

    # Thử load từ file, nếu không có thì tạo mới
    weather_probs_path = os.path.join(FEATURES_DIR, "weather_probabilities.csv")
    if os.path.exists(weather_probs_path):
        weather_probs = pd.read_csv(weather_probs_path)
        bn.build_from_data(weather_probs_df=weather_probs)
    else:
        print("⚠️ Không có file weather_probabilities.csv.")
        print("  Đang dùng dữ liệu thời tiết raw để tính...")
        try:
            from modules.data_pipeline import load_vietnam_weather, clean_vietnam_weather
            df_weather = clean_vietnam_weather(load_vietnam_weather())
            weather_probs = build_weather_probability_table(df_weather)
            bn.build_from_data(weather_probs_df=weather_probs)
        except Exception as e:
            print(f"Không thể load dữ liệu: {e}")
            print("  → Sử dụng prior mặc định.")
            bn.build_from_data()

    # In cấu trúc mạng
    bn.print_network()

    # --- Truy vấn 1: Thời tiết TP.HCM ---
    print("\n" + "🔸" * 30)
    print("TRUY VẤN 1: Thời tiết Ho Chi Minh theo tháng")
    print("🔸" * 30)

    for month in [1, 5, 8, 12]:
        weather = bn.query_weather_full("Ho Chi Minh", month)
        print(f"  Tháng {month:2d}: "
              f"P(rain)={weather['p_rain']:.3f}, "
              f"P(outdoor)={weather['p_outdoor']:.3f}, "
              f"P(hot)={weather['p_hot']:.3f}, "
              f"P(humid)={weather['p_humid']:.3f}")

    # --- Truy vấn 2: So sánh các tỉnh du lịch tháng 7 ---
    print("\n" + "🔸" * 30)
    print("TRUY VẤN 2: So sánh thời tiết tháng 7 (mùa mưa)")
    print("🔸" * 30)

    provinces = ["Ha Noi", "Da Nang", "Ho Chi Minh", "Lam Dong", "Khanh Hoa"]
    for prov in provinces:
        weather = bn.query_weather_full(prov, 7)
        print(f"  {prov:15s}: "
              f"P(rain)={weather['p_rain']:.3f}, "
              f"P(outdoor)={weather['p_outdoor']:.3f}")

    # --- Truy vấn 3: Tháng tốt nhất để du lịch ---
    print("\n" + "🔸" * 30)
    print("TRUY VẤN 3: Tháng tốt nhất để du lịch biển Nha Trang")
    print("🔸" * 30)
    best_months = bn.predict_best_month("Khanh Hoa", "beach", "couple")
    print(best_months.to_string(index=False))

    # --- Scoring 4: Xếp hạng địa điểm ---
    print("\n" + "🔸" * 30)
    print("TRUY VẤN 4: Xếp hạng 30 điểm du lịch (tháng 3, cặp đôi)")
    print("🔸" * 30)
    df_places = build_places_dataframe()
    scored = bn.score_places(df_places, month=3, group_type="couple")

    # --- Tích hợp 5: Full Pipeline C+D ---
    print("\n" + "🔸" * 30)
    print("TRUY VẤN 5: Tích hợp C+D (Đà Nẵng, tháng 8, gia đình)")
    print("🔸" * 30)
    try:
        result, meta = integrate_bayes_kb(
            df_places,
            province="Da Nang",
            month=8,
            group_type="family",
            budget_vnd=3_000_000,
            user_preferences=["beach", "culture"],
            num_days=3,
            weather_probs_df=weather_probs if 'weather_probs' in dir() else None,
        )
    except ImportError as e:
        print(f"Cần import knowledge_base: {e}")

    return bn


if __name__ == "__main__":
    demo_bayesian_network()
