"""
knowledge_base.py — Thành phần (C): Hệ luật IF-THEN cho suy luận tri thức.

Module này xây dựng hệ thống luật suy luận (Rule-Based Expert System) để:
  - Lọc hoạt động theo điều kiện thời tiết (mưa, nóng, ẩm)
  - Lọc theo ngân sách (ưu tiên địa điểm phù hợp mức chi tiêu)
  - Lọc theo nhóm du khách (gia đình, cặp đôi, đi một mình, nhóm bạn)
  - Lọc theo mùa du lịch
  - Gợi ý hoạt động phù hợp dựa trên sở thích user
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


# ============================================================
# 0. CONSTANTS & CATEGORY MAPPINGS
# ============================================================

# Phân loại hoạt động outdoor/indoor
OUTDOOR_CATEGORIES = {"nature", "beach", "adventure"}
INDOOR_CATEGORIES = {"culture", "entertainment"}

# Phân loại hoạt động theo nhóm người dùng
FAMILY_FRIENDLY = {"nature", "culture", "beach", "entertainment"}
FAMILY_EXCLUDE = set()  # không loại trừ gì đặc biệt

COUPLE_FRIENDLY = {"nature", "beach", "culture", "entertainment"}
COUPLE_EXCLUDE = set()

SOLO_FRIENDLY = {"adventure", "nature", "culture", "beach"}
SOLO_EXCLUDE = set()

GROUP_FRIENDLY = {"adventure", "nature", "beach", "entertainment"}
GROUP_EXCLUDE = set()

# Ngưỡng ngân sách (VND)
BUDGET_THRESHOLDS = {
    "low": 500_000,        # < 500K: budget thấp
    "medium": 2_000_000,   # 500K - 2M: trung bình
    "high": 5_000_000,     # 2M - 5M: cao
    # > 5M: luxury
}

# Ngưỡng thời tiết
WEATHER_THRESHOLDS = {
    "rain_heavy_mm": 10.0,     # Mưa > 10mm → loại hoạt động outdoor
    "rain_moderate_mm": 5.0,   # Mưa > 5mm → cảnh báo outdoor
    "temp_hot_c": 35.0,        # Nhiệt độ > 35°C → cảnh báo
    "temp_cold_c": 15.0,       # Nhiệt độ < 15°C → cần chuẩn bị
    "humidity_high": 80.0,     # Độ ẩm > 80% → khó chịu (theo KTTV Việt Nam)
    "wind_strong_kmh": 40.0,   # Gió > 40km/h → nguy hiểm
}


# ============================================================
# 1. RULE DEFINITIONS
# ============================================================

class Rule:
    """Một luật IF-THEN trong hệ tri thức."""

    def __init__(self, rule_id: str, name: str, description: str,
                 condition_fn, action_fn, priority: int = 5):
        """
        Args:
            rule_id: ID duy nhất của luật
            name: Tên luật (ngắn gọn)
            description: Mô tả chi tiết
            condition_fn: Hàm (context) -> bool, kiểm tra điều kiện IF
            action_fn: Hàm (context, places_df) -> (places_df, explanation),
                       thực thi hành động THEN
            priority: Mức ưu tiên (1=cao nhất, 10=thấp nhất)
        """
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.condition = condition_fn
        self.action = action_fn
        self.priority = priority

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Đánh giá điều kiện IF."""
        try:
            return self.condition(context)
        except Exception:
            return False

    def execute(self, context: Dict[str, Any],
                places_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Thực thi hành động THEN."""
        return self.action(context, places_df)

    def __repr__(self):
        return f"Rule({self.rule_id}: {self.name}, priority={self.priority})"


# ============================================================
# 2. KNOWLEDGE BASE
# ============================================================

class KnowledgeBase:
    """
    Hệ tri thức (Knowledge Base) chứa tập luật IF-THEN.
    Sử dụng Forward Chaining (suy luận tiến) để áp dụng luật.
    """

    def __init__(self):
        self.rules: List[Rule] = []
        self.firing_log: List[Dict] = []  # Log các luật đã kích hoạt
        self._build_default_rules()

    def add_rule(self, rule: Rule):
        """Thêm luật vào knowledge base."""
        self.rules.append(rule)
        # Sắp xếp theo priority
        self.rules.sort(key=lambda r: r.priority)

    def _build_default_rules(self):
        """Xây dựng tập luật mặc định."""

        # ------- LUẬT THỜI TIẾT -------

        # R1: Mưa to → loại hoạt động outdoor
        self.add_rule(Rule(
            rule_id="R1_HEAVY_RAIN",
            name="Mưa to → Loại outdoor",
            description="IF mưa > 10mm THEN loại bỏ hoạt động outdoor (nature, beach, adventure)",
            condition_fn=lambda ctx: ctx.get("rain_mm", 0) > WEATHER_THRESHOLDS["rain_heavy_mm"],
            action_fn=self._action_remove_outdoor,
            priority=1
        ))

        # R2: Mưa vừa → cảnh báo outdoor
        self.add_rule(Rule(
            rule_id="R2_MODERATE_RAIN",
            name="Mưa vừa → Cảnh báo outdoor",
            description="IF 5mm < mưa <= 10mm THEN giảm ưu tiên outdoor, gợi ý mang áo mưa",
            condition_fn=lambda ctx: (
                WEATHER_THRESHOLDS["rain_moderate_mm"]
                < ctx.get("rain_mm", 0)
                <= WEATHER_THRESHOLDS["rain_heavy_mm"]
            ),
            action_fn=self._action_warn_outdoor,
            priority=2
        ))

        # R3: Nóng quá → ưu tiên indoor/beach
        self.add_rule(Rule(
            rule_id="R3_HOT_WEATHER",
            name="Nóng > 35°C → Ưu tiên indoor/beach",
            description="IF temp_max > 35°C THEN ưu tiên indoor và beach, giảm adventure",
            condition_fn=lambda ctx: ctx.get("temp_max", 25) > WEATHER_THRESHOLDS["temp_hot_c"],
            action_fn=self._action_prefer_indoor_beach,
            priority=3
        ))

        # R4: Lạnh → cảnh báo
        self.add_rule(Rule(
            rule_id="R4_COLD_WEATHER",
            name="Lạnh < 15°C → Cảnh báo",
            description="IF temp_min < 15°C THEN gợi ý mang áo ấm, ưu tiên văn hóa/entertainment",
            condition_fn=lambda ctx: ctx.get("temp_min", 20) < WEATHER_THRESHOLDS["temp_cold_c"],
            action_fn=self._action_warn_cold,
            priority=4
        ))

        # R5: Độ ẩm cao → giảm outdoor
        self.add_rule(Rule(
            rule_id="R5_HIGH_HUMIDITY",
            name="Ẩm > 85% → Giảm outdoor",
            description="IF humidity > 85% THEN giảm ưu tiên hoạt động ngoài trời",
            condition_fn=lambda ctx: ctx.get("humidity", 70) > WEATHER_THRESHOLDS["humidity_high"],
            action_fn=self._action_reduce_outdoor_priority,
            priority=5
        ))

        # R6: Gió mạnh → loại adventure nguy hiểm
        self.add_rule(Rule(
            rule_id="R6_STRONG_WIND",
            name="Gió > 40km/h → Loại adventure",
            description="IF wind_speed > 40km/h THEN loại bỏ adventure (leo núi, hang động)",
            condition_fn=lambda ctx: ctx.get("wind_speed", 10) > WEATHER_THRESHOLDS["wind_strong_kmh"],
            action_fn=self._action_remove_adventure,
            priority=1
        ))

        # ------- LUẬT NGÂN SÁCH -------

        # R7: Ngân sách thấp → ưu tiên miễn phí
        self.add_rule(Rule(
            rule_id="R7_LOW_BUDGET",
            name="Ngân sách thấp → Ưu tiên miễn phí",
            description="IF budget < 500K VND THEN ưu tiên địa điểm miễn phí hoặc phí vào cổng thấp",
            condition_fn=lambda ctx: ctx.get("budget_vnd", float("inf")) < BUDGET_THRESHOLDS["low"],
            action_fn=self._action_budget_low,
            priority=2
        ))

        # R8: Ngân sách trung bình → lọc phí cao
        self.add_rule(Rule(
            rule_id="R8_MEDIUM_BUDGET",
            name="Ngân sách TB → Loại phí cao",
            description="IF 500K <= budget < 2M VND THEN loại bỏ địa điểm phí > 500K",
            condition_fn=lambda ctx: (
                BUDGET_THRESHOLDS["low"]
                <= ctx.get("budget_vnd", float("inf"))
                < BUDGET_THRESHOLDS["medium"]
            ),
            action_fn=self._action_budget_medium,
            priority=3
        ))

        # ------- LUẬT NHÓM DU KHÁCH -------

        # R9: Gia đình → loại nightlife/adventure nguy hiểm
        self.add_rule(Rule(
            rule_id="R9_FAMILY",
            name="Gia đình → Loại adventure nguy hiểm",
            description="IF group_type == 'family' THEN loại bỏ adventure nguy hiểm (Son Doong, Fansipan)",
            condition_fn=lambda ctx: ctx.get("group_type", "").lower() == "family",
            action_fn=self._action_family_filter,
            priority=3
        ))

        # R10: Cặp đôi → ưu tiên romantic
        self.add_rule(Rule(
            rule_id="R10_COUPLE",
            name="Cặp đôi → Ưu tiên romantic",
            description="IF group_type == 'couple' THEN ưu tiên beach, nature, culture",
            condition_fn=lambda ctx: ctx.get("group_type", "").lower() == "couple",
            action_fn=self._action_couple_filter,
            priority=4
        ))

        # R11: Nhóm bạn → ưu tiên adventure/entertainment
        self.add_rule(Rule(
            rule_id="R11_GROUP",
            name="Nhóm bạn → Ưu tiên vui chơi",
            description="IF group_type == 'friends' THEN ưu tiên adventure, entertainment, beach",
            condition_fn=lambda ctx: ctx.get("group_type", "").lower() == "friends",
            action_fn=self._action_group_filter,
            priority=4
        ))

        # ------- LUẬT MÙA DU LỊCH -------

        # R12: Mùa mưa → ưu tiên indoor
        self.add_rule(Rule(
            rule_id="R12_RAINY_SEASON",
            name="Mùa mưa → Ưu tiên indoor",
            description="IF season == 'summer' (mùa mưa VN) THEN tăng ưu tiên indoor",
            condition_fn=lambda ctx: ctx.get("season", "").lower() in ["summer", "autumn"],
            action_fn=self._action_rainy_season,
            priority=5
        ))

        # R13: Thời tiết đẹp → khuyến khích outdoor
        self.add_rule(Rule(
            rule_id="R13_GOOD_WEATHER",
            name="Thời tiết đẹp → Khuyến khích outdoor",
            description="IF outdoor_suitable == True THEN tăng ưu tiên outdoor",
            condition_fn=lambda ctx: ctx.get("outdoor_suitable", False) is True,
            action_fn=self._action_good_weather,
            priority=6
        ))

        # ------- LUẬT SỞ THÍCH USER -------

        # R14: User thích văn hóa
        self.add_rule(Rule(
            rule_id="R14_CULTURE_LOVER",
            name="User thích văn hóa → Ưu tiên culture",
            description="IF user_preference == 'culture' THEN tăng ưu tiên culture, museums",
            condition_fn=lambda ctx: "culture" in ctx.get("user_preferences", []),
            action_fn=self._action_prefer_culture,
            priority=6
        ))

        # R15: User thích phiêu lưu
        self.add_rule(Rule(
            rule_id="R15_ADVENTURE_SEEKER",
            name="User thích phiêu lưu → Ưu tiên adventure",
            description="IF user_preference == 'adventure' THEN tăng ưu tiên adventure, nature",
            condition_fn=lambda ctx: "adventure" in ctx.get("user_preferences", []),
            action_fn=self._action_prefer_adventure,
            priority=6
        ))

        # R16: User thích biển
        self.add_rule(Rule(
            rule_id="R16_BEACH_LOVER",
            name="User thích biển → Ưu tiên beach",
            description="IF user_preference == 'beach' THEN tăng ưu tiên beach",
            condition_fn=lambda ctx: "beach" in ctx.get("user_preferences", []),
            action_fn=self._action_prefer_beach,
            priority=6
        ))

        # ------- LUẬT THỜI GIAN -------

        # R17: Ít ngày → ưu tiên gần nhau
        self.add_rule(Rule(
            rule_id="R17_SHORT_TRIP",
            name="Ít ngày → Ưu tiên gần nhau",
            description="IF num_days <= 2 THEN ưu tiên các điểm gần nhau (cùng tỉnh)",
            condition_fn=lambda ctx: ctx.get("num_days", 7) <= 2,
            action_fn=self._action_short_trip,
            priority=4
        ))

        # R18: Giờ muộn → loại địa điểm đã đóng
        self.add_rule(Rule(
            rule_id="R18_LATE_HOUR",
            name="Quá giờ → Loại địa điểm đã đóng",
            description="IF current_hour >= closing_hour THEN loại bỏ địa điểm đã đóng cửa",
            condition_fn=lambda ctx: ctx.get("current_hour", 8) is not None,
            action_fn=self._action_check_opening_hours,
            priority=2
        ))

    # ============================================================
    # 3. ACTION FUNCTIONS (THEN)
    # ============================================================

    def _action_remove_outdoor(self, ctx, df):
        """Loại bỏ hoạt động outdoor."""
        if "category" in df.columns:
            removed = df[df["category"].isin(OUTDOOR_CATEGORIES)]
            df = df[~df["category"].isin(OUTDOOR_CATEGORIES)]
            explanation = (
                f"⛈️ Mưa to ({ctx.get('rain_mm', '?')}mm) → "
                f"Loại {len(removed)} điểm outdoor: "
                f"{', '.join(removed['place_name'].tolist()[:5])}"
            )
        else:
            explanation = "⛈️ Mưa to → Nên tránh hoạt động ngoài trời"
        return df, explanation

    def _action_warn_outdoor(self, ctx, df):
        """Cảnh báo outdoor nhưng không loại bỏ."""
        if "category" in df.columns and "weather_score" not in df.columns:
            df = df.copy()
            df["weather_score"] = df["category"].apply(
                lambda c: 0.5 if c in OUTDOOR_CATEGORIES else 1.0
            )
        explanation = (
            f"🌧️ Mưa vừa ({ctx.get('rain_mm', '?')}mm) → "
            f"Cảnh báo: hoạt động outdoor có thể bị ảnh hưởng. Nên mang áo mưa."
        )
        return df, explanation

    def _action_prefer_indoor_beach(self, ctx, df):
        """Ưu tiên indoor và beach khi nóng."""
        if "category" in df.columns:
            df = df.copy()
            if "weather_score" not in df.columns:
                df["weather_score"] = 1.0
            df.loc[df["category"].isin({"culture", "entertainment", "beach"}),
                   "weather_score"] *= 1.3
            df.loc[df["category"] == "adventure", "weather_score"] *= 0.6
        explanation = (
            f"🌡️ Nóng ({ctx.get('temp_max', '?')}°C) → "
            f"Ưu tiên hoạt động indoor/beach, giảm adventure"
        )
        return df, explanation

    def _action_warn_cold(self, ctx, df):
        """Cảnh báo lạnh."""
        explanation = (
            f"❄️ Lạnh ({ctx.get('temp_min', '?')}°C) → "
            f"Nên mang áo ấm. Ưu tiên hoạt động văn hóa/entertainment."
        )
        if "category" in df.columns:
            df = df.copy()
            if "weather_score" not in df.columns:
                df["weather_score"] = 1.0
            df.loc[df["category"].isin({"culture", "entertainment"}),
                   "weather_score"] *= 1.2
        return df, explanation

    def _action_reduce_outdoor_priority(self, ctx, df):
        """Giảm ưu tiên outdoor khi ẩm cao."""
        if "category" in df.columns:
            df = df.copy()
            if "weather_score" not in df.columns:
                df["weather_score"] = 1.0
            df.loc[df["category"].isin(OUTDOOR_CATEGORIES),
                   "weather_score"] *= 0.8
        explanation = (
            f"💧 Độ ẩm cao ({ctx.get('humidity', '?')}%) → "
            f"Giảm ưu tiên hoạt động ngoài trời"
        )
        return df, explanation

    def _action_remove_adventure(self, ctx, df):
        """Loại bỏ adventure khi gió mạnh."""
        if "category" in df.columns:
            removed = df[df["category"] == "adventure"]
            df = df[df["category"] != "adventure"]
            explanation = (
                f"💨 Gió mạnh ({ctx.get('wind_speed', '?')}km/h) → "
                f"Loại {len(removed)} điểm adventure nguy hiểm"
            )
        else:
            explanation = "💨 Gió mạnh → Nên tránh hoạt động phiêu lưu"
        return df, explanation

    def _action_budget_low(self, ctx, df):
        """Ưu tiên miễn phí khi ngân sách thấp."""
        if "entry_fee_vnd" in df.columns:
            budget = ctx.get("budget_vnd", 500000)
            max_fee = budget * 0.3  # Tối đa 30% ngân sách cho phí vào cổng
            removed = df[df["entry_fee_vnd"] > max_fee]
            df = df[df["entry_fee_vnd"] <= max_fee]
            explanation = (
                f"💰 Ngân sách thấp ({budget:,.0f} VND) → "
                f"Ưu tiên địa điểm miễn phí/phí thấp. "
                f"Loại {len(removed)} điểm phí cao"
            )
        else:
            explanation = f"💰 Ngân sách thấp → Ưu tiên địa điểm miễn phí"
        return df, explanation

    def _action_budget_medium(self, ctx, df):
        """Loại bỏ địa điểm phí quá cao."""
        if "entry_fee_vnd" in df.columns:
            max_fee = 500_000
            removed = df[df["entry_fee_vnd"] > max_fee]
            df = df[df["entry_fee_vnd"] <= max_fee]
            explanation = (
                f"💵 Ngân sách TB → Loại {len(removed)} điểm "
                f"phí vào cổng > {max_fee:,.0f} VND"
            )
        else:
            explanation = "💵 Ngân sách trung bình → Loại địa điểm phí quá cao"
        return df, explanation

    def _action_family_filter(self, ctx, df):
        """Lọc cho gia đình."""
        if "category" in df.columns:
            # Loại adventure nguy hiểm (phí > 1M thường là tour chuyên nghiệp)
            dangerous = df[
                (df["category"] == "adventure") &
                (df.get("entry_fee_vnd", pd.Series(dtype=float)).fillna(0) > 1_000_000)
            ] if "entry_fee_vnd" in df.columns else pd.DataFrame()

            if len(dangerous) > 0:
                df = df.drop(dangerous.index)

            explanation = (
                f"👨‍👩‍👧‍👦 Gia đình → Loại {len(dangerous)} điểm adventure nguy hiểm. "
                f"Ưu tiên: văn hóa, thiên nhiên, biển"
            )
        else:
            explanation = "👨‍👩‍👧‍👦 Gia đình → Ưu tiên hoạt động an toàn"
        return df, explanation

    def _action_couple_filter(self, ctx, df):
        """Ưu tiên cho cặp đôi."""
        if "category" in df.columns:
            df = df.copy()
            if "preference_score" not in df.columns:
                df["preference_score"] = 1.0
            df.loc[df["category"].isin({"beach", "nature"}),
                   "preference_score"] *= 1.3
        explanation = "💑 Cặp đôi → Ưu tiên beach, nature (romantic)"
        return df, explanation

    def _action_group_filter(self, ctx, df):
        """Ưu tiên cho nhóm bạn."""
        if "category" in df.columns:
            df = df.copy()
            if "preference_score" not in df.columns:
                df["preference_score"] = 1.0
            df.loc[df["category"].isin({"adventure", "entertainment", "beach"}),
                   "preference_score"] *= 1.3
        explanation = "👫 Nhóm bạn → Ưu tiên adventure, entertainment, beach"
        return df, explanation

    def _action_rainy_season(self, ctx, df):
        """Xử lý mùa mưa."""
        if "category" in df.columns:
            df = df.copy()
            if "weather_score" not in df.columns:
                df["weather_score"] = 1.0
            df.loc[df["category"].isin(INDOOR_CATEGORIES), "weather_score"] *= 1.2
            df.loc[df["category"].isin(OUTDOOR_CATEGORIES), "weather_score"] *= 0.8
        explanation = (
            f"🌦️ Mùa mưa ({ctx.get('season', '?')}) → "
            f"Tăng ưu tiên indoor, giảm outdoor"
        )
        return df, explanation

    def _action_good_weather(self, ctx, df):
        """Khuyến khích outdoor khi thời tiết đẹp."""
        if "category" in df.columns:
            df = df.copy()
            if "weather_score" not in df.columns:
                df["weather_score"] = 1.0
            df.loc[df["category"].isin(OUTDOOR_CATEGORIES), "weather_score"] *= 1.3
        explanation = "☀️ Thời tiết đẹp → Khuyến khích hoạt động ngoài trời!"
        return df, explanation

    def _action_prefer_culture(self, ctx, df):
        """Ưu tiên văn hóa."""
        if "category" in df.columns:
            df = df.copy()
            if "preference_score" not in df.columns:
                df["preference_score"] = 1.0
            df.loc[df["category"] == "culture", "preference_score"] *= 1.4
        explanation = "🏛️ User thích văn hóa → Ưu tiên điểm văn hóa, lịch sử"
        return df, explanation

    def _action_prefer_adventure(self, ctx, df):
        """Ưu tiên phiêu lưu."""
        if "category" in df.columns:
            df = df.copy()
            if "preference_score" not in df.columns:
                df["preference_score"] = 1.0
            df.loc[df["category"].isin({"adventure", "nature"}),
                   "preference_score"] *= 1.4
        explanation = "🧗 User thích phiêu lưu → Ưu tiên adventure, nature"
        return df, explanation

    def _action_prefer_beach(self, ctx, df):
        """Ưu tiên biển."""
        if "category" in df.columns:
            df = df.copy()
            if "preference_score" not in df.columns:
                df["preference_score"] = 1.0
            df.loc[df["category"] == "beach", "preference_score"] *= 1.5
        explanation = "🏖️ User thích biển → Ưu tiên điểm biển"
        return df, explanation

    def _action_short_trip(self, ctx, df):
        """Lọc điểm gần nhau cho trip ngắn."""
        # Nếu có thông tin tỉnh hiện tại, ưu tiên cùng tỉnh
        current_province = ctx.get("current_province", None)
        if "province" in df.columns and current_province:
            df = df.copy()
            if "preference_score" not in df.columns:
                df["preference_score"] = 1.0
            df.loc[df["province"] == current_province, "preference_score"] *= 1.5
        explanation = (
            f"📅 Trip ngắn ({ctx.get('num_days', '?')} ngày) → "
            f"Ưu tiên điểm gần nhau"
        )
        return df, explanation

    def _action_check_opening_hours(self, ctx, df):
        """Loại điểm đã đóng cửa."""
        current_hour = ctx.get("current_hour")
        if current_hour is not None and "opening_hour" in df.columns and "closing_hour" in df.columns:
            closed = df[
                (df["opening_hour"] > current_hour) |
                (df["closing_hour"] <= current_hour)
            ]
            df = df[
                (df["opening_hour"] <= current_hour) &
                (df["closing_hour"] > current_hour)
            ]
            explanation = (
                f"🕐 Giờ hiện tại: {current_hour}h → "
                f"Loại {len(closed)} điểm đã đóng/chưa mở cửa"
            )
        else:
            explanation = "🕐 Kiểm tra giờ mở cửa"
        return df, explanation

    # ============================================================
    # 4. INFERENCE ENGINE
    # ============================================================

    def infer(self, context: Dict[str, Any],
              places_df: pd.DataFrame,
              verbose: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Thực hiện suy luận tiến (Forward Chaining).

        Args:
            context: Dict chứa thông tin ngữ cảnh:
                - rain_mm: Lượng mưa (mm)
                - temp_max: Nhiệt độ cao nhất (°C)
                - temp_min: Nhiệt độ thấp nhất (°C)
                - humidity: Độ ẩm (%)
                - wind_speed: Tốc độ gió (km/h)
                - budget_vnd: Ngân sách (VND)
                - group_type: "family" | "couple" | "solo" | "friends"
                - season: "spring" | "summer" | "autumn" | "winter"
                - outdoor_suitable: bool
                - user_preferences: list of str ("culture", "adventure", "beach", ...)
                - num_days: Số ngày du lịch
                - current_hour: Giờ hiện tại (0-23)
                - current_province: Tỉnh hiện tại
            places_df: DataFrame các điểm du lịch

        Returns:
            (filtered_places_df, firing_log) — DataFrame đã lọc + log các luật kích hoạt
        """
        self.firing_log = []
        current_df = places_df.copy()
        initial_count = len(current_df)

        if verbose:
            print("=" * 60)
            print("🧠 KNOWLEDGE BASE — FORWARD CHAINING INFERENCE")
            print("=" * 60)
            print(f"\n📋 Context: {context}")
            print(f"📍 Số điểm ban đầu: {initial_count}")

        # Áp dụng từng luật theo thứ tự priority
        for rule in self.rules:
            if rule.evaluate(context):
                before_count = len(current_df)
                current_df, explanation = rule.execute(context, current_df)
                after_count = len(current_df)

                log_entry = {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "explanation": explanation,
                    "places_before": before_count,
                    "places_after": after_count,
                    "places_removed": before_count - after_count,
                }
                self.firing_log.append(log_entry)

                if verbose:
                    print(f"\n🔥 {rule.rule_id}: {rule.name}")
                    print(f"   → {explanation}")
                    if before_count != after_count:
                        print(f"   → Còn lại: {after_count}/{before_count} điểm")

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"📊 KẾT QUẢ: {len(current_df)}/{initial_count} điểm phù hợp")
            print(f"📜 Số luật kích hoạt: {len(self.firing_log)}")
            print(f"{'=' * 60}")

        return current_df, self.firing_log

    def explain(self) -> str:
        """Giải thích các luật đã kích hoạt (cho báo cáo)."""
        if not self.firing_log:
            return "Chưa có phiên suy luận nào được thực hiện."

        lines = ["📜 GIẢI THÍCH SUY LUẬN (IF-THEN Rules)", "=" * 50]
        for i, log in enumerate(self.firing_log, 1):
            lines.append(f"\n{i}. [{log['rule_id']}] {log['rule_name']}")
            lines.append(f"   {log['explanation']}")
            if log['places_removed'] > 0:
                lines.append(
                    f"   → Loại {log['places_removed']} điểm "
                    f"({log['places_before']} → {log['places_after']})"
                )
        return "\n".join(lines)

    def get_all_rules_info(self) -> pd.DataFrame:
        """Trả về DataFrame mô tả tất cả luật."""
        rows = []
        for r in self.rules:
            rows.append({
                "rule_id": r.rule_id,
                "name": r.name,
                "description": r.description,
                "priority": r.priority,
            })
        return pd.DataFrame(rows)


# ============================================================
# 5. CONVENIENCE FUNCTIONS
# ============================================================

def create_context(
    rain_mm: float = 0,
    temp_max: float = 30,
    temp_min: float = 22,
    humidity: float = 70,
    wind_speed: float = 10,
    budget_vnd: float = 2_000_000,
    group_type: str = "solo",
    season: str = "spring",
    outdoor_suitable: bool = True,
    user_preferences: Optional[List[str]] = None,
    num_days: int = 3,
    current_hour: Optional[int] = None,
    current_province: Optional[str] = None,
) -> Dict[str, Any]:
    """Tạo context dict chuẩn cho inference."""
    return {
        "rain_mm": rain_mm,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "budget_vnd": budget_vnd,
        "group_type": group_type,
        "season": season,
        "outdoor_suitable": outdoor_suitable,
        "user_preferences": user_preferences or [],
        "num_days": num_days,
        "current_hour": current_hour,
        "current_province": current_province,
    }


def filter_places_by_weather(
    places_df: pd.DataFrame,
    weather_data: Dict[str, float],
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Lọc địa điểm chỉ dựa trên thời tiết.
    Tiện dùng khi chỉ cần check thời tiết mà không cần full context.
    """
    kb = KnowledgeBase()
    ctx = create_context(
        rain_mm=weather_data.get("rain_mm", 0),
        temp_max=weather_data.get("temp_max", 30),
        temp_min=weather_data.get("temp_min", 22),
        humidity=weather_data.get("humidity", 70),
        wind_speed=weather_data.get("wind_speed", 10),
        outdoor_suitable=weather_data.get("outdoor_suitable", True),
    )
    return kb.infer(ctx, places_df, verbose=verbose)


def filter_places_full(
    places_df: pd.DataFrame,
    weather_data: Dict[str, float],
    budget_vnd: float = 2_000_000,
    group_type: str = "solo",
    season: str = "spring",
    user_preferences: Optional[List[str]] = None,
    num_days: int = 3,
    current_hour: Optional[int] = None,
    current_province: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Lọc địa điểm đầy đủ (thời tiết + ngân sách + nhóm + sở thích).
    Đây là hàm chính gọi từ planner.
    """
    kb = KnowledgeBase()
    ctx = create_context(
        rain_mm=weather_data.get("rain_mm", 0),
        temp_max=weather_data.get("temp_max", 30),
        temp_min=weather_data.get("temp_min", 22),
        humidity=weather_data.get("humidity", 70),
        wind_speed=weather_data.get("wind_speed", 10),
        budget_vnd=budget_vnd,
        group_type=group_type,
        season=season,
        outdoor_suitable=weather_data.get("outdoor_suitable", True),
        user_preferences=user_preferences,
        num_days=num_days,
        current_hour=current_hour,
        current_province=current_province,
    )
    return kb.infer(ctx, places_df, verbose=verbose)


# ============================================================
# 6. DEMO / TEST
# ============================================================

def demo_knowledge_base():
    """Demo hệ tri thức với các kịch bản khác nhau."""
    # Import places data
    try:
        from modules.data_pipeline import build_places_dataframe
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from modules.data_pipeline import build_places_dataframe

    df_places = build_places_dataframe()
    kb = KnowledgeBase()

    # Hiển thị tất cả luật
    print("\n📚 TẤT CẢ LUẬT TRONG KNOWLEDGE BASE:")
    print(kb.get_all_rules_info().to_string(index=False))

    # ------- Kịch bản 1: Mưa to, gia đình, ngân sách thấp -------
    print("\n\n" + "🔸" * 30)
    print("KỊCH BẢN 1: Mưa to + Gia đình + Ngân sách thấp")
    print("🔸" * 30)
    ctx1 = create_context(
        rain_mm=15, temp_max=28, temp_min=22, humidity=90,
        budget_vnd=400_000, group_type="family",
        season="summer", outdoor_suitable=False,
        user_preferences=["culture"],
    )
    result1, log1 = kb.infer(ctx1, df_places)
    print(f"\nĐịa điểm gợi ý:")
    if len(result1) > 0:
        print(result1[["place_name", "category", "province", "entry_fee_vnd"]].to_string(index=False))

    # ------- Kịch bản 2: Thời tiết đẹp, cặp đôi, ngân sách cao -------
    print("\n\n" + "🔸" * 30)
    print("KỊCH BẢN 2: Thời tiết đẹp + Cặp đôi + Ngân sách cao")
    print("🔸" * 30)
    ctx2 = create_context(
        rain_mm=0, temp_max=28, temp_min=22, humidity=65,
        budget_vnd=5_000_000, group_type="couple",
        season="spring", outdoor_suitable=True,
        user_preferences=["beach", "nature"],
    )
    result2, log2 = kb.infer(ctx2, df_places)
    print(f"\nĐịa điểm gợi ý:")
    if len(result2) > 0:
        cols = ["place_name", "category", "province", "entry_fee_vnd"]
        extra = [c for c in ["weather_score", "preference_score"] if c in result2.columns]
        print(result2[cols + extra].to_string(index=False))

    # ------- Kịch bản 3: Nhóm bạn, phiêu lưu, gió mạnh -------
    print("\n\n" + "🔸" * 30)
    print("KỊCH BẢN 3: Nhóm bạn + Phiêu lưu + Gió mạnh")
    print("🔸" * 30)
    ctx3 = create_context(
        rain_mm=0, temp_max=25, temp_min=18, humidity=60,
        wind_speed=45, budget_vnd=10_000_000,
        group_type="friends", season="winter",
        outdoor_suitable=True, user_preferences=["adventure"],
    )
    result3, log3 = kb.infer(ctx3, df_places)
    print(f"\nĐịa điểm gợi ý:")
    if len(result3) > 0:
        cols = ["place_name", "category", "province", "entry_fee_vnd"]
        extra = [c for c in ["weather_score", "preference_score"] if c in result3.columns]
        print(result3[cols + extra].to_string(index=False))

    return kb


if __name__ == "__main__":
    demo_knowledge_base()
