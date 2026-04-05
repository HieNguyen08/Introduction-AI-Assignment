"""
planner.py — Module tích hợp tất cả thành phần AI cho hệ thống lập kế hoạch du lịch.

Pipeline tổng:
    Input (user preferences)
      → from modules.csp_solver import solve_schedule(E) ML phân loại user
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
import heapq
import math
from modules.search import find_optimal_daily_route, load_matrices, load_places
from modules.csp_solver import solve_schedule
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

# ==========================================
# PHẦN 1: THUẬT TOÁN A* (TÌM ĐƯỜNG ĐI ĐA MỤC TIÊU)
# ==========================================

class AStarPlanner:
    def __init__(self, graph, nodes_data, weights=(1.0, 1.0, 1.0)):
        """
        graph: Cấu trúc dict of dicts, VD: { 'A': {'B': {'dist': 2.5, 'time': 10, 'cost': 50000}} }
        nodes_data: Chứa tọa độ lat/lon của các node VD: {'A': {'lat': 10.7, 'lon': 106.6}}
        weights: (w_dist, w_time, w_cost) - Trọng số do người dùng chọn
        """
        self.raw_graph = graph
        self.nodes_data = nodes_data

        # Chuẩn hóa trọng số để tổng = 1.0
        total_weight = sum(weights)
        self.w_dist = weights[0] / total_weight
        self.w_time = weights[1] / total_weight
        self.w_cost = weights[2] / total_weight

        self.normalized_graph = self._normalize_graph()

    def _normalize_graph(self):
        """Chuẩn hóa các giá trị dist, time, cost về khoảng [0, 1] bằng Min-Max Scaler"""
        norm_graph = {}
        all_dists, all_times, all_costs = [], [], []

        # Lấy tất cả giá trị để tìm Min, Max
        for u in self.raw_graph:
            for v in self.raw_graph[u]:
                edge = self.raw_graph[u][v]
                all_dists.append(edge.get('dist', 0))
                all_times.append(edge.get('time', 0))
                all_costs.append(edge.get('cost', 0))

        min_d, max_d = min(all_dists), max(all_dists)
        min_t, max_t = min(all_times), max(all_times)
        min_c, max_c = min(all_costs), max(all_costs)

        # Hàm lambda để tránh lỗi chia cho 0
        scale = lambda val, min_v, max_v: (val - min_v) / (max_v - min_v) if max_v > min_v else 0

        # Tạo đồ thị đã chuẩn hóa
        for u in self.raw_graph:
            norm_graph[u] = {}
            for v in self.raw_graph[u]:
                edge = self.raw_graph[u][v]
                norm_graph[u][v] = {
                    'dist_norm': scale(edge.get('dist', 0), min_d, max_d),
                    'time_norm': scale(edge.get('time', 0), min_t, max_t),
                    'cost_norm': scale(edge.get('cost', 0), min_c, max_c)
                }
        return norm_graph

    def haversine(self, lat1, lon1, lat2, lon2):
        """Tính khoảng cách đường chim bay (km)"""
        R = 6371  # Bán kính trái đất (km)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def heuristic(self, current, goal):
        """
        Hàm Heuristic h(n) có thể điều chỉnh để admissibility cao nhất.
        Ở mức độ cơ bản, ta dùng Euclidean/Haversine distance làm base.
        """
        lat1, lon1 = self.nodes_data[current]['lat'], self.nodes_data[current]['lon']
        lat2, lon2 = self.nodes_data[goal]['lat'], self.nodes_data[goal]['lon']

        # Để an toàn và đảm bảo thuật toán không ước lượng quá cao (overestimate),
        # ta cho h(n) dựa trên khoảng cách địa lý đơn thuần nhân với trọng số.
        dist = self.haversine(lat1, lon1, lat2, lon2)
        # Giả định: 1km tương đương 0.01 điểm penalty trên thang chuẩn hóa (có thể tinh chỉnh)
        return dist * 0.01 * self.w_dist

    def find_path(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}

        g_score = {node: float('inf') for node in self.raw_graph}
        g_score[start] = 0

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in self.normalized_graph.get(current, {}):
                edge = self.normalized_graph[current][neighbor]
                # Tính chi phí cạnh đã chuẩn hóa
                step_cost = (self.w_dist * edge['dist_norm'] +
                             self.w_time * edge['time_norm'] +
                             self.w_cost * edge['cost_norm'])

                tentative_g_score = g_score[current] + step_cost

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# ==========================================
# PHẦN 2: CSP BACKTRACKING + FORWARD CHECKING
# ==========================================

class CSPScheduler:
    def __init__(self, pois, budget, start_time_minutes, astar_planner):
        """
        pois: List các dictionary thông tin POI.
              VD: [{'id': 'P1', 'cost': 50000, 'duration': 120, 'open_time': 480, 'close_time': 1020}]
        budget: Ngân sách tối đa (VND)
        start_time_minutes: Thời gian bắt đầu lịch trình (tính bằng phút từ 0h, VD 8h sáng = 480)
        astar_planner: Instance của AStarPlanner để tính nhanh thời gian di chuyển giữa các POI
        """
        self.pois = {poi['id']: poi for poi in pois}
        self.budget = budget
        self.start_time = start_time_minutes
        self.astar_planner = astar_planner
        self.best_schedule = []
        self.max_pois_visited = 0

    def get_travel_time(self, node_a, node_b):
        """Lấy thời gian di chuyển thực tế từ Graph thông qua A* Planner (hoặc Lookup table)"""
        # Tránh lỗi nếu trùng node
        if node_a == node_b: return 0

        # Lookup nhanh từ raw_graph nếu có cạnh nối trực tiếp, nếu không trả về ước lượng
        graph = self.astar_planner.raw_graph
        if node_a in graph and node_b in graph[node_a]:
            return graph[node_a][node_b].get('time', 15) # Mặc định 15p nếu thiếu data
        return 20 # Mặc định 20p di chuyển nếu không liền kề

    def forward_checking(self, current_time, current_cost, remaining_poi_ids, last_visited_id=None):
        """
        Lọc bỏ các POI không còn khả thi dựa trên trạng thái hiện tại.
        Giúp Backtracking chạy nhanh hơn rất nhiều.
        """
        valid_pois = []
        for poi_id in remaining_poi_ids:
            poi = self.pois[poi_id]

            # Kiểm tra Ràng buộc 1: Ngân sách
            if current_cost + poi['cost'] > self.budget:
                continue

            # Kiểm tra Ràng buộc 2: Thời gian
            travel_time = self.get_travel_time(last_visited_id, poi_id) if last_visited_id else 0
            arrival_time = current_time + travel_time

            # Nếu đến nơi trước giờ mở cửa, phải đợi đến lúc mở cửa
            actual_start_time = max(arrival_time, poi['open_time'])
            finish_time = actual_start_time + poi['duration']

            # Nếu thời gian kết thúc vượt quá giờ đóng cửa -> Bỏ qua
            if finish_time > poi['close_time']:
                continue

            valid_pois.append(poi_id)

        return valid_pois

    def backtrack(self, current_schedule, current_time, current_cost, remaining_poi_ids):
        """Thuật toán Quay lui đệ quy"""
        # Lưu lại lịch trình tốt nhất (đi được nhiều điểm nhất hợp lệ)
        if len(current_schedule) > self.max_pois_visited:
            self.max_pois_visited = len(current_schedule)
            self.best_schedule = list(current_schedule)

        # Trích xuất ID của điểm cuối cùng vừa tham quan
        last_visited_id = current_schedule[-1] if current_schedule else None

        # Tỉa nhánh: Lấy danh sách POI hợp lệ tiếp theo
        valid_next_pois = self.forward_checking(current_time, current_cost, remaining_poi_ids, last_visited_id)

        for poi_id in valid_next_pois:
            poi = self.pois[poi_id]
            travel_time = self.get_travel_time(last_visited_id, poi_id) if last_visited_id else 0

            arrival_time = current_time + travel_time
            actual_start_time = max(arrival_time, poi['open_time'])
            finish_time = actual_start_time + poi['duration']
            new_cost = current_cost + poi['cost']

            # Backtrack Step
            current_schedule.append(poi_id)
            next_remaining = [p for p in remaining_poi_ids if p != poi_id]

            self.backtrack(current_schedule, finish_time, new_cost, next_remaining)

            # Undo Step (Quay lui)
            current_schedule.pop()

    def generate_schedule(self):
        """Hàm kích hoạt bộ lập lịch"""
        all_poi_ids = list(self.pois.keys())
        self.backtrack(current_schedule=[],
                       current_time=self.start_time,
                       current_cost=0,
                       remaining_poi_ids=all_poi_ids)
        return self.best_schedule


# ==========================================
# PHẦN 3: PIPELINE TÍCH HỢP
# (Thử nghiệm nhanh nghiệm logic của bạn)
# ==========================================
def run_member2_pipeline(graph, nodes_data, pois_list, user_budget, weights=(1.0, 1.0, 1.0)):
    # Bước 1: Khởi tạo A* Planner để xử lý logic không gian và trọng số
    planner = AStarPlanner(graph, nodes_data, weights)

    # Bước 2: Khởi chạy CSP để lọc và sắp xếp lịch trình POI
    scheduler = CSPScheduler(pois=pois_list,
                             budget=user_budget,
                             start_time_minutes=480, # 8:00 AM
                             astar_planner=planner)

    optimal_pois = scheduler.generate_schedule()
    print(f"[CSP] Lịch trình tối ưu đi qua các điểm: {optimal_pois}")

    # Bước 3: Định tuyến chi tiết (Routing) giữa các POI đã chọn bằng A*
    full_route = []
    for i in range(len(optimal_pois) - 1):
        start_node = optimal_pois[i]
        goal_node = optimal_pois[i+1]
        path_segment = planner.find_path(start_node, goal_node)

        if path_segment:
            # Bỏ node đầu tiên của segment sau để tránh lặp (A -> B, B -> C)
            if full_route:
                full_route.extend(path_segment[1:])
            else:
                full_route.extend(path_segment)

    print(f"[A*] Lộ trình chi tiết từng trạm (Nodes): {full_route}")
    return full_route
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

    # Step 4 (B): CSP phân chia địa điểm → thoả mãn ràng buộc ngân sách + giờ
    from modules.csp_solver import solve_schedule
    from modules.search import load_matrices, load_places, find_optimal_daily_route

    _, cost_mat, time_mat = load_matrices()
    places_ref_df = load_places()

    # Xác định budget_level từ budget_vnd
    if budget_vnd < 1_000_000:
        budget_level = "budget"
    elif budget_vnd < 3_000_000:
        budget_level = "mid_range"
    else:
        budget_level = "premium"

    csp_result = solve_schedule(
        ranked_places_df=ranked_places,
        province=province,
        num_days=num_days,
        budget_vnd=budget_vnd,
        budget_level=budget_level,
        time_limit_per_day=10.0,
        max_places_per_day=max_places_per_day,
        time_matrix=time_mat,
        cost_matrix=cost_mat,
    )

    # Step 5 (A): A* tối ưu thứ tự tham quan trong từng ngày
    daily_plan = {}
    for day, day_places_list in csp_result["schedule"].items():
        if not day_places_list:
            daily_plan[f"Ngày {day}"] = pd.DataFrame()
            continue

        day_df = pd.DataFrame(day_places_list)

        route_result = find_optimal_daily_route(
            day_places_df=day_df,
            places_df=places_ref_df,
            time_matrix=time_mat,
            cost_matrix=cost_mat,
            time_limit_hours=10.0,
            budget_vnd=budget_vnd * 0.3,  # 30% budget cho di chuyển/ngày
        )

        # Sắp xếp lại day_df theo thứ tự A* trả về
        if route_result["route"] and len(route_result["route"]) > 0:
            ordered_names = route_result["path_names"]
            day_df["_order"] = day_df["place_name"].apply(
                lambda n: ordered_names.index(n) if n in ordered_names else 999
            )
            day_df = day_df.sort_values("_order").drop(columns=["_order"])

        daily_plan[f"Ngày {day}"] = day_df

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
