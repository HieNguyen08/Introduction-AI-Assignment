"""
csp_solver.py — CSP Backtracking + Forward Checking cho lịch trình du lịch.

TV2: Trần Ngọc Khánh Huy
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")


# ============================================================
# LOAD DATA
# ============================================================

def load_hotel_price_stats() -> pd.DataFrame:
    """Load bảng giá khách sạn theo tỉnh."""
    path = os.path.join(FEATURES_DIR, "hotel_price_stats.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def get_hotel_cost(province: str, budget_level: str,
                   num_nights: int, stats_df: pd.DataFrame) -> float:
    """
    Ước tính chi phí khách sạn.

    Args:
        province: Tên tỉnh (e.g., "Da Nang", "Ha Noi")
        budget_level: "budget" | "mid_range" | "premium"
        num_nights: Số đêm ở
        stats_df: DataFrame từ hotel_price_stats.csv

    Returns:
        Tổng chi phí VND
    """
    if stats_df.empty:
        # Fallback: ước tính mặc định
        defaults = {"budget": 300_000, "mid_range": 800_000, "premium": 2_500_000}
        return defaults.get(budget_level, 500_000) * num_nights

    mask = (stats_df["province"].str.lower() == province.lower()) & \
           (stats_df["price_tier"] == budget_level)
    row = stats_df[mask]
    if row.empty:
        # Dùng median toàn quốc cho tier đó
        fallback = stats_df[stats_df["price_tier"] == budget_level]["median_price"]
        price = fallback.median() if not fallback.empty else 500_000
    else:
        price = row.iloc[0]["median_price"]
    return price * num_nights


# ============================================================
# CONSTRAINT CHECKERS
# ============================================================

def check_time_constraint(
    places: List[pd.Series],
    time_matrix: np.ndarray,
    places_df: pd.DataFrame,
    time_limit_hours: float = 10.0,
) -> Tuple[bool, float]:
    """
    Kiểm tra ràng buộc thời gian: tổng (di chuyển + tham quan) ≤ time_limit_hours.

    Returns:
        (feasible: bool, total_time: float)
    """
    if not places:
        return True, 0.0

    place_names = list(places_df["place_name"])
    total_time = 0.0

    # Thời gian tham quan
    for p in places:
        total_time += p.get("visit_duration_hours", 1.0)

    # Thời gian di chuyển (theo thứ tự tham quan hiện tại — sẽ tối ưu bởi A*)
    for i in range(len(places) - 1):
        name_a = places[i]["place_name"]
        name_b = places[i + 1]["place_name"]
        if name_a in place_names and name_b in place_names:
            idx_a = place_names.index(name_a)
            idx_b = place_names.index(name_b)
            if idx_a < time_matrix.shape[0] and idx_b < time_matrix.shape[1]:
                total_time += time_matrix[idx_a][idx_b]

    return total_time <= time_limit_hours, total_time


def check_budget_constraint(
    all_assigned: List[pd.Series],
    province: str,
    num_days: int,
    budget_vnd: float,
    budget_level: str,
    stats_df: pd.DataFrame,
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    places_df: pd.DataFrame,
) -> Tuple[bool, float]:
    """
    Kiểm tra ràng buộc ngân sách:
        Σ(vé vào cổng) + Σ(chi phí di chuyển) + chi phí khách sạn ≤ budget_vnd

    Returns:
        (feasible: bool, total_cost: float)
    """
    # Vé vào cổng
    entry_fees = sum(p.get("entry_fee_vnd", 0) for p in all_assigned)

    # Chi phí di chuyển (ước tính)
    place_names = list(places_df["place_name"])
    travel_cost = 0.0
    for i in range(len(all_assigned) - 1):
        na = all_assigned[i]["place_name"]
        nb = all_assigned[i + 1]["place_name"]
        if na in place_names and nb in place_names:
            ia, ib = place_names.index(na), place_names.index(nb)
            if ia < cost_matrix.shape[0] and ib < cost_matrix.shape[1]:
                travel_cost += cost_matrix[ia][ib]

    # Chi phí khách sạn
    hotel_cost = get_hotel_cost(province, budget_level, num_days, stats_df)

    total_cost = entry_fees + travel_cost + hotel_cost
    return total_cost <= budget_vnd, total_cost


def check_opening_hours(place: pd.Series, start_hour: int = 8) -> bool:
    """
    Kiểm tra giờ mở cửa:
        opening_hour ≤ start_hour < closing_hour - visit_duration
    """
    opening = place.get("opening_hour", 0)
    closing = place.get("closing_hour", 24)
    duration = place.get("visit_duration_hours", 1.0)
    return opening <= start_hour and (start_hour + duration) <= closing


def forward_check(
    candidate: pd.Series,
    assigned_today: List[pd.Series],
    remaining_budget: float,
    remaining_time: float,
    places_df: pd.DataFrame,
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
) -> bool:
    """
    Forward Checking: kiểm tra xem thêm candidate có vi phạm ràng buộc không.

    Returns:
        True nếu có thể thêm candidate (không vi phạm), False nếu prune.
    """
    # Kiểm tra phí vào cổng
    fee = candidate.get("entry_fee_vnd", 0)
    if fee > remaining_budget:
        return False

    # Kiểm tra thời gian tham quan
    visit_t = candidate.get("visit_duration_hours", 1.0)
    if visit_t > remaining_time:
        return False

    # Kiểm tra thời gian di chuyển từ điểm cuối cùng
    if assigned_today:
        place_names = list(places_df["place_name"])
        last = assigned_today[-1]
        last_name = last["place_name"]
        cand_name = candidate["place_name"]
        if last_name in place_names and cand_name in place_names:
            ia = place_names.index(last_name)
            ib = place_names.index(cand_name)
            if ia < time_matrix.shape[0] and ib < time_matrix.shape[1]:
                travel_t = time_matrix[ia][ib]
                if visit_t + travel_t > remaining_time:
                    return False

    return True


# ============================================================
# BACKTRACKING CSP SOLVER
# ============================================================

def csp_backtrack(
    day: int,
    num_days: int,
    domain: List[pd.Series],
    schedule: Dict[int, List[pd.Series]],
    assigned_global: set,
    province: str,
    budget_vnd: float,
    budget_level: str,
    time_limit_per_day: float,
    max_places_per_day: int,
    stats_df: pd.DataFrame,
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    places_df: pd.DataFrame,
    best_solution: Dict,
) -> bool:
    """
    Backtracking với Forward Checking.
    Điền lịch trình từng ngày, tối đa hoá tổng bayesian_score.

    Args:
        day: Ngày đang xử lý (1-indexed)
        num_days: Tổng số ngày
        domain: Danh sách địa điểm có thể chọn (đã được filter + rank)
        schedule: Dict {day: [places]} — kết quả đang xây dựng
        assigned_global: Set tên các địa điểm đã được gán (tránh trùng)
        ...

    Returns:
        True nếu tìm được lịch trình hợp lệ
    """
    if day > num_days:
        # Tất cả ngày đã được gán → lưu nếu tốt hơn best
        all_places = [p for d in range(1, num_days + 1) for p in schedule.get(d, [])]
        total_score = sum(p.get("bayesian_score", p.get("final_score", 0)) for p in all_places)

        feasible, total_cost = check_budget_constraint(
            all_places, province, num_days, budget_vnd, budget_level,
            stats_df, time_matrix, cost_matrix, places_df
        )

        if feasible and total_score > best_solution.get("score", -1):
            best_solution["schedule"] = {k: list(v) for k, v in schedule.items()}
            best_solution["score"] = total_score
            best_solution["total_cost"] = total_cost
        return feasible

    schedule[day] = []
    remaining_time = time_limit_per_day
    # Ước tính ngân sách còn lại cho ngày này (chia đều)
    all_prev = [p for d in range(1, day) for p in schedule.get(d, [])]
    spent_so_far = sum(p.get("entry_fee_vnd", 0) for p in all_prev)
    remaining_budget = budget_vnd * 0.6 - spent_so_far  # 60% budget cho vé + di chuyển

    # Duyệt qua domain (đã sắp xếp theo bayesian_score giảm dần)
    for candidate in domain:
        cand_name = candidate["place_name"]

        # Ràng buộc: không thăm lại
        if cand_name in assigned_global:
            continue

        # Forward Checking
        if not forward_check(candidate, schedule[day], remaining_budget,
                             remaining_time, places_df, time_matrix, cost_matrix):
            continue

        # Kiểm tra giờ mở cửa
        start_hour = 8 + len(schedule[day])  # ước tính giờ đến
        if not check_opening_hours(candidate, start_hour):
            continue

        # Thêm vào schedule ngày hiện tại
        schedule[day].append(candidate)
        assigned_global.add(cand_name)

        visit_t = candidate.get("visit_duration_hours", 1.0)
        fee = candidate.get("entry_fee_vnd", 0)
        remaining_time -= visit_t
        remaining_budget -= fee

        if len(schedule[day]) >= max_places_per_day:
            # Ngày đầy → chuyển sang ngày kế
            if csp_backtrack(day + 1, num_days, domain, schedule,
                             assigned_global, province, budget_vnd, budget_level,
                             time_limit_per_day, max_places_per_day, stats_df,
                             time_matrix, cost_matrix, places_df, best_solution):
                return True

        remaining_time += visit_t
        remaining_budget += fee
        schedule[day].pop()
        assigned_global.discard(cand_name)

    # Chuyển sang ngày tiếp dù ngày hiện tại chưa đầy
    return csp_backtrack(day + 1, num_days, domain, schedule,
                         assigned_global, province, budget_vnd, budget_level,
                         time_limit_per_day, max_places_per_day, stats_df,
                         time_matrix, cost_matrix, places_df, best_solution)


# ============================================================
# PUBLIC API
# ============================================================

def solve_schedule(
    ranked_places_df: pd.DataFrame,
    province: str,
    num_days: int,
    budget_vnd: float,
    budget_level: str = "mid_range",
    time_limit_per_day: float = 10.0,
    max_places_per_day: int = 3,
    time_matrix: Optional[np.ndarray] = None,
    cost_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    API chính: giải CSP để tạo lịch trình thoả mãn ràng buộc.

    Args:
        ranked_places_df: DataFrame đã filter+rank từ filter_and_rank_places() (TV3)
        province: Tỉnh du lịch
        num_days: Số ngày
        budget_vnd: Tổng ngân sách (VND)
        budget_level: "budget" | "mid_range" | "premium"
        time_limit_per_day: Số giờ tối đa mỗi ngày (mặc định 10h)
        max_places_per_day: Số địa điểm tối đa mỗi ngày
        time_matrix: Ma trận thời gian (None = tự load)
        cost_matrix: Ma trận chi phí (None = tự load)

    Returns:
        Dict:
            "schedule": {1: [place_row, ...], 2: [...], ...}
            "score": float tổng bayesian_score
            "total_cost": float tổng chi phí ước tính
            "feasible": bool
    """
    from modules.search import load_matrices, load_places

    if time_matrix is None or cost_matrix is None:
        _, cost_matrix, time_matrix = load_matrices()

    places_df = load_places()
    stats_df = load_hotel_price_stats()

    # Sắp xếp domain theo score giảm dần (greedy ordering cho backtracking)
    score_col = "bayesian_score" if "bayesian_score" in ranked_places_df.columns else "final_score"
    domain_df = ranked_places_df.sort_values(score_col, ascending=False)
    domain = [row for _, row in domain_df.iterrows()]

    best_solution = {"schedule": {}, "score": -1, "total_cost": 0}

    csp_backtrack(
        day=1,
        num_days=num_days,
        domain=domain,
        schedule={},
        assigned_global=set(),
        province=province,
        budget_vnd=budget_vnd,
        budget_level=budget_level,
        time_limit_per_day=time_limit_per_day,
        max_places_per_day=max_places_per_day,
        stats_df=stats_df,
        time_matrix=time_matrix,
        cost_matrix=cost_matrix,
        places_df=places_df,
        best_solution=best_solution,
    )

    # Kiểm tra kết quả backtracking có thực sự có địa điểm không
    all_selected = [
        p for d in best_solution.get("schedule", {}).values() for p in d
    ]
    if not all_selected:
        # Backtracking không điền được lịch → dùng greedy với constraints
        return greedy_schedule_with_constraints(
            domain=domain,
            province=province,
            num_days=num_days,
            budget_vnd=budget_vnd,
            budget_level=budget_level,
            time_limit_per_day=time_limit_per_day,
            max_places_per_day=max_places_per_day,
            stats_df=stats_df,
            time_matrix=time_matrix,
            cost_matrix=cost_matrix,
            places_df=places_df,
        )

    return {
        "schedule": best_solution["schedule"],
        "score": best_solution["score"],
        "total_cost": best_solution["total_cost"],
        "feasible": True,
    }


def greedy_schedule_with_constraints(
    domain: List[pd.Series],
    province: str,
    num_days: int,
    budget_vnd: float,
    budget_level: str,
    time_limit_per_day: float,
    max_places_per_day: int,
    stats_df: pd.DataFrame,
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    places_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Greedy CSP: chọn địa điểm tốt nhất (theo score) qua từng ngày,
    áp dụng Forward Checking để đảm bảo ràng buộc thời gian + ngân sách.

    Đây là CSP greedy — đảm bảo luôn trả về lịch không rỗng.
    """
    hotel_cost = get_hotel_cost(province, budget_level, num_days, stats_df)
    remaining_total_budget = max(0, budget_vnd - hotel_cost)

    assigned_global = set()
    schedule = {}

    for day in range(1, num_days + 1):
        schedule[day] = []
        remaining_time = time_limit_per_day
        # Chia đều ngân sách còn lại theo số ngày còn lại
        days_left = num_days - day + 1
        day_budget = remaining_total_budget / max(1, days_left)
        remaining_day_budget = day_budget

        for candidate in domain:
            if len(schedule[day]) >= max_places_per_day:
                break

            cand_name = candidate["place_name"]
            if cand_name in assigned_global:
                continue

            # Kiểm tra giờ mở cửa dựa trên giờ dự kiến đến
            start_hour = 8 + int(sum(
                p.get("visit_duration_hours", 1.0) for p in schedule[day]
            ))
            if not check_opening_hours(candidate, start_hour):
                continue

            # Forward Checking
            if forward_check(candidate, schedule[day], remaining_day_budget,
                             remaining_time, places_df, time_matrix, cost_matrix):
                schedule[day].append(candidate)
                assigned_global.add(cand_name)
                remaining_time -= candidate.get("visit_duration_hours", 1.0)
                remaining_day_budget -= candidate.get("entry_fee_vnd", 0)

        # Cập nhật ngân sách tổng sau mỗi ngày
        day_spent = sum(p.get("entry_fee_vnd", 0) for p in schedule[day])
        remaining_total_budget -= day_spent

    all_places = [p for d in schedule.values() for p in d]
    score_col = next(
        (col for col in ["bayesian_score", "final_score"] if any(col in p for p in all_places)),
        "bayesian_score"
    )
    total_score = sum(p.get("bayesian_score", p.get("final_score", 0)) for p in all_places)
    feasible, total_cost = check_budget_constraint(
        all_places, province, num_days, budget_vnd, budget_level,
        stats_df, time_matrix, cost_matrix, places_df
    )

    return {
        "schedule": schedule,
        "score": total_score,
        "total_cost": total_cost,
        "feasible": feasible,
    }


def _greedy_schedule(domain_df: pd.DataFrame, num_days: int,
                     max_per_day: int) -> Dict[str, Any]:
    """Fallback đơn giản: chia đều địa điểm theo ngày (không kiểm tra ràng buộc)."""
    schedule = {}
    places = [row for _, row in domain_df.iterrows()]
    idx = 0
    for day in range(1, num_days + 1):
        schedule[day] = []
        for _ in range(max_per_day):
            if idx < len(places):
                schedule[day].append(places[idx])
                idx += 1
    return {"schedule": schedule, "score": 0, "total_cost": 0, "feasible": False}
