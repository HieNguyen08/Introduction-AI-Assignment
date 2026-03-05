"""
planner.py — Module tích hợp tất cả thành phần AI cho hệ thống lập kế hoạch du lịch.

Pipeline tổng:
    Input (user preferences) 
      → (E) ML phân loại user
      → (D) Bayes dự đoán thời tiết
      → (C) IF-THEN rules lọc địa điểm
      → (B) CSP ràng buộc lịch trình
      → (A) A* tìm route tối ưu
      → Output (lịch trình tối ưu + giải thích)

Module này tích hợp:
  - (C) Knowledge Base IF-THEN rules (knowledge_base.py)
  - (D) Bayesian Network (bayesian_net.py)
  - (A) A* Search (search.py) — TODO: TV2
  - (B) CSP Solver (csp_solver.py) — TODO: TV2
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Import modules
from modules.knowledge_base import (
    KnowledgeBase, create_context,
    filter_places_full, filter_places_by_weather,
)
from modules.bayesian_net import (
    BayesianNetwork, integrate_bayes_kb,
)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")


# ============================================================
# 1. DATA LOADING HELPERS
# ============================================================

def load_places() -> pd.DataFrame:
    """Load DataFrame các điểm du lịch Việt Nam."""
    path = os.path.join(FEATURES_DIR, "vn_tourist_places.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback: build from data_pipeline
    from modules.data_pipeline import build_places_dataframe
    return build_places_dataframe()


def load_weather_probs() -> Optional[pd.DataFrame]:
    """Load bảng xác suất thời tiết."""
    path = os.path.join(FEATURES_DIR, "weather_probabilities.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_distance_matrix() -> Optional[np.ndarray]:
    """Load ma trận khoảng cách."""
    path = os.path.join(FEATURES_DIR, "distance_matrix.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def load_cost_matrix() -> Optional[np.ndarray]:
    """Load ma trận chi phí."""
    path = os.path.join(FEATURES_DIR, "cost_matrix.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def load_time_matrix() -> Optional[np.ndarray]:
    """Load ma trận thời gian."""
    path = os.path.join(FEATURES_DIR, "travel_time_matrix.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


# ============================================================
# 2. COMPONENT C+D: WEATHER-AWARE PLACE FILTERING
# ============================================================

def filter_and_rank_places(
    province: str,
    month: int,
    group_type: str = "solo",
    budget_vnd: float = 2_000_000,
    user_preferences: Optional[List[str]] = None,
    num_days: int = 3,
    current_hour: Optional[int] = None,
    places_df: Optional[pd.DataFrame] = None,
    weather_probs_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Lọc và xếp hạng địa điểm du lịch — Tích hợp (C) + (D).

    Pipeline:
        1. (D) Bayesian Network dự đoán thời tiết P(rain | province, month)
        2. (C) Knowledge Base IF-THEN rules lọc địa điểm không phù hợp
        3. (D) Bayesian scoring xếp hạng địa điểm còn lại

    Args:
        province: Tỉnh du lịch chính (e.g., "Da Nang", "Ha Noi")
        month: Tháng du lịch (1-12)
        group_type: "solo" | "couple" | "family" | "friends"
        budget_vnd: Ngân sách (VND)
        user_preferences: Danh sách sở thích ["culture", "beach", ...]
        num_days: Số ngày du lịch
        current_hour: Giờ hiện tại (0-23), None = không kiểm tra
        places_df: DataFrame địa điểm (None = auto load)
        weather_probs_df: Bảng xác suất thời tiết (None = auto load)
        verbose: In chi tiết

    Returns:
        (ranked_places_df, metadata)
    """
    # Load data nếu chưa cung cấp
    if places_df is None:
        places_df = load_places()
    if weather_probs_df is None:
        weather_probs_df = load_weather_probs()

    # Gọi hàm tích hợp C+D từ bayesian_net module
    ranked_df, metadata = integrate_bayes_kb(
        places_df=places_df,
        province=province,
        month=month,
        group_type=group_type,
        budget_vnd=budget_vnd,
        user_preferences=user_preferences,
        num_days=num_days,
        current_hour=current_hour,
        weather_probs_df=weather_probs_df,
        verbose=verbose,
    )

    return ranked_df, metadata


def get_weather_recommendation(
    province: str,
    month: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Lấy khuyến nghị thời tiết cho một tỉnh tại một tháng nhất định.

    Returns:
        Dict chứa:
        - weather: Dict xác suất thời tiết
        - recommendation: str khuyến nghị
        - outdoor_ok: bool có nên outdoor không
        - warnings: List[str] cảnh báo
    """
    # Xây dựng Bayesian Network
    bn = BayesianNetwork()
    weather_probs = load_weather_probs()
    bn.build_from_data(weather_probs_df=weather_probs)

    # Truy vấn thời tiết
    weather = bn.query_weather_full(province, month)

    # Phân tích và đưa ra khuyến nghị
    warnings = []
    recommendations = []

    if weather["p_rain"] > 0.6:
        warnings.append(f"⛈️ Xác suất mưa cao ({weather['p_rain']:.0%})")
        recommendations.append("Nên chuẩn bị áo mưa và ưu tiên hoạt động indoor")
    elif weather["p_rain"] > 0.4:
        warnings.append(f"🌧️ Có khả năng mưa ({weather['p_rain']:.0%})")
        recommendations.append("Nên mang theo áo mưa phòng")

    if weather["p_hot"] > 0.5:
        warnings.append(f"🌡️ Khả năng nóng cao ({weather['p_hot']:.0%})")
        recommendations.append("Nên tránh hoạt động ngoài trời giữa trưa")

    if weather["p_humid"] > 0.6:
        warnings.append(f"💧 Độ ẩm có thể cao ({weather['p_humid']:.0%})")

    outdoor_ok = weather["p_outdoor"] > 0.5

    if outdoor_ok:
        recommendations.append("☀️ Thời tiết khá thuận lợi cho hoạt động ngoài trời")
    else:
        recommendations.append("🏠 Nên ưu tiên hoạt động indoor")

    recommendation_text = ". ".join(recommendations) if recommendations else "Thời tiết bình thường"

    result = {
        "weather": weather,
        "recommendation": recommendation_text,
        "outdoor_ok": outdoor_ok,
        "warnings": warnings,
    }

    if verbose:
        print(f"\n🌤️ KHUYẾN NGHỊ THỜI TIẾT — {province}, Tháng {month}")
        print(f"{'=' * 50}")
        print(f"  P(mưa)    = {weather['p_rain']:.1%}")
        print(f"  P(outdoor) = {weather['p_outdoor']:.1%}")
        print(f"  P(nóng)   = {weather['p_hot']:.1%}")
        print(f"  P(ẩm)     = {weather['p_humid']:.1%}")
        if warnings:
            print(f"\n  ⚠️ Cảnh báo:")
            for w in warnings:
                print(f"    {w}")
        print(f"\n  📋 Khuyến nghị: {recommendation_text}")

    return result


def find_best_travel_month(
    province: str,
    category: str = "nature",
    group_type: str = "solo",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Tìm tháng tốt nhất để du lịch một tỉnh (theo loại hình).

    Returns:
        DataFrame 12 tháng với score
    """
    bn = BayesianNetwork()
    weather_probs = load_weather_probs()
    bn.build_from_data(weather_probs_df=weather_probs)
    return bn.predict_best_month(province, category, group_type)


# ============================================================
# 3. FULL TRAVEL PLANNER (C+D, A+B sẽ thêm bởi TV2)
# ============================================================

def plan_trip(
    province: str,
    month: int,
    group_type: str = "solo",
    budget_vnd: float = 2_000_000,
    user_preferences: Optional[List[str]] = None,
    num_days: int = 3,
    max_places_per_day: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Lập kế hoạch du lịch đầy đủ.

    Pipeline:
        1. (D) Dự đoán thời tiết → khuyến nghị
        2. (C) Lọc địa điểm theo rules
        3. (D) Xếp hạng bằng Bayesian scoring
        4. (A) Tìm route tối ưu — TODO: TV2 sẽ thêm A* search
        5. (B) Kiểm tra ràng buộc — TODO: TV2 sẽ thêm CSP

    Returns:
        Dict chứa kết quả lập kế hoạch
    """
    if verbose:
        print("=" * 70)
        print("🗺️  AI TRAVEL PLANNER")
        print("=" * 70)
        print(f"  📍 Tỉnh: {province}")
        print(f"  📅 Tháng: {month} | Số ngày: {num_days}")
        print(f"  👥 Nhóm: {group_type}")
        print(f"  💰 Ngân sách: {budget_vnd:,.0f} VND")
        print(f"  ❤️ Sở thích: {user_preferences or 'Không chỉ định'}")
        print("=" * 70)

    # Step 1-3: Filter & Rank (C+D)
    ranked_places, metadata = filter_and_rank_places(
        province=province,
        month=month,
        group_type=group_type,
        budget_vnd=budget_vnd,
        user_preferences=user_preferences,
        num_days=num_days,
        verbose=verbose,
    )

    # Step 4: Phân chia địa điểm theo ngày (simple greedy — TV2 sẽ thay bằng A* + CSP)
    selected = ranked_places.head(max_places_per_day * num_days)
    daily_plan = {}
    for day in range(1, num_days + 1):
        start_idx = (day - 1) * max_places_per_day
        end_idx = min(day * max_places_per_day, len(selected))
        day_places = selected.iloc[start_idx:end_idx]
        daily_plan[f"Ngày {day}"] = day_places

    # Weather recommendation
    weather_rec = get_weather_recommendation(province, month, verbose=False)

    # Build result
    result = {
        "province": province,
        "month": month,
        "group_type": group_type,
        "budget_vnd": budget_vnd,
        "num_days": num_days,
        "weather": weather_rec,
        "ranked_places": ranked_places,
        "daily_plan": daily_plan,
        "metadata": metadata,
    }

    # Print plan
    if verbose:
        print(f"\n{'=' * 70}")
        print("📋 KẾ HOẠCH DU LỊCH")
        print(f"{'=' * 70}")

        # Weather
        print(f"\n🌤️ Thời tiết: {weather_rec['recommendation']}")
        if weather_rec["warnings"]:
            for w in weather_rec["warnings"]:
                print(f"  {w}")

        # Daily plan
        for day_name, day_places in daily_plan.items():
            print(f"\n📅 {day_name}:")
            if len(day_places) == 0:
                print("  (Không có địa điểm phù hợp)")
                continue
            for _, place in day_places.iterrows():
                fee = place.get("entry_fee_vnd", 0)
                fee_str = f"{fee:,.0f} VND" if fee > 0 else "Miễn phí"
                score = place.get("bayesian_score", place.get("final_score", "N/A"))
                if isinstance(score, float):
                    score = f"{score:.3f}"
                print(f"  📍 {place['place_name']} ({place['category']}) "
                      f"— {place['province']} | {fee_str} | score={score}")

        print(f"\n{'=' * 70}")
        print(f"✅ Tổng: {len(selected)} địa điểm / {num_days} ngày")
        print(f"   Luật IF-THEN đã áp dụng: {metadata.get('rules_fired', 'N/A')}")
        print(f"{'=' * 70}")

    return result


# ============================================================
# 4. DEMO
# ============================================================

def demo_planner():
    """Demo toàn bộ planner."""
    print("\n" + "🔸" * 35)
    print("DEMO 1: Gia đình du lịch Đà Nẵng tháng 8")
    print("🔸" * 35)
    result1 = plan_trip(
        province="Da Nang", month=8, group_type="family",
        budget_vnd=3_000_000, user_preferences=["beach", "culture"],
        num_days=3,
    )

    print("\n\n" + "🔸" * 35)
    print("DEMO 2: Cặp đôi du lịch Hà Nội tháng 3")
    print("🔸" * 35)
    result2 = plan_trip(
        province="Ha Noi", month=3, group_type="couple",
        budget_vnd=2_000_000, user_preferences=["culture"],
        num_days=2,
    )

    print("\n\n" + "🔸" * 35)
    print("DEMO 3: Solo adventure Lâm Đồng tháng 12")
    print("🔸" * 35)
    result3 = plan_trip(
        province="Lam Dong", month=12, group_type="solo",
        budget_vnd=5_000_000, user_preferences=["adventure", "nature"],
        num_days=4,
    )

    return result1, result2, result3


if __name__ == "__main__":
    demo_planner()
