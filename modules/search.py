"""
search.py — Thuật toán A* tìm lộ trình tối ưu giữa các điểm du lịch.

TV2: Trần Ngọc Khánh Huy
"""

import heapq
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, FrozenSet
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")


# ============================================================
# LOAD DATA
# ============================================================

def load_matrices():
    """Load ma trận khoảng cách, chi phí, thời gian."""
    dist  = np.load(os.path.join(FEATURES_DIR, "distance_matrix.npy"))
    cost  = np.load(os.path.join(FEATURES_DIR, "cost_matrix.npy"))
    ttime = np.load(os.path.join(FEATURES_DIR, "travel_time_matrix.npy"))
    return dist, cost, ttime

def load_places() -> pd.DataFrame:
    """Load 50 địa điểm đầu tiên (tương ứng với ma trận 50x50)."""
    df = pd.read_csv(os.path.join(FEATURES_DIR, "vn_tourist_places.csv"))
    return df.head(50).reset_index(drop=True)


# ============================================================
# STATE & NODE
# ============================================================

class SearchState:
    """
    State trong không gian tìm kiếm A*.

    Attributes:
        current_idx (int): Index địa điểm hiện tại (0-49)
        visited (FrozenSet[int]): Tập các index đã thăm
        time_used (float): Tổng giờ đã dùng (di chuyển + tham quan)
        cost_used (float): Tổng VND đã chi (vé + di chuyển)
    """
    def __init__(self, current_idx: int, visited: FrozenSet[int],
                 time_used: float, cost_used: float):
        self.current_idx = current_idx
        self.visited = visited
        self.time_used = time_used
        self.cost_used = cost_used

    def __eq__(self, other):
        return (self.current_idx == other.current_idx and
                self.visited == other.visited)

    def __hash__(self):
        return hash((self.current_idx, self.visited))

    def __lt__(self, other):
        # Dùng cho heapq (không cần so sánh state trực tiếp)
        return False


class SearchNode:
    """Node trong cây tìm kiếm A*."""
    def __init__(self, state: SearchState, parent=None,
                 g: float = 0.0, h: float = 0.0):
        self.state = state
        self.parent = parent
        self.g = g       # chi phí thực tế từ start
        self.h = h       # ước lượng heuristic đến goal
        self.f = g + h   # tổng chi phí ước lượng

    def __lt__(self, other):
        return self.f < other.f


# ============================================================
# HEURISTIC
# ============================================================

def heuristic_mst(
    current_idx: int,
    remaining: List[int],
    time_matrix: np.ndarray,
) -> float:
    """
    Heuristic admissible: MST (Minimum Spanning Tree) trên các đỉnh chưa thăm.

    Đây là lower bound của thời gian cần để thăm hết remaining,
    vì MST là cây khung nhỏ nhất — không thể đi ít hơn.

    Args:
        current_idx: Index địa điểm hiện tại
        remaining: Danh sách index chưa thăm
        time_matrix: Ma trận thời gian di chuyển (giờ)

    Returns:
        Lower bound thời gian di chuyển (giờ)
    """
    if not remaining:
        return 0.0

    # Tập đỉnh = current + remaining
    nodes = [current_idx] + remaining
    n = len(nodes)

    if n == 1:
        return 0.0

    # Prim's algorithm cho MST
    in_mst = [False] * n
    min_edge = [float('inf')] * n
    min_edge[0] = 0.0
    total_weight = 0.0

    for _ in range(n):
        # Tìm đỉnh chưa trong MST có min_edge nhỏ nhất
        u = -1
        for v in range(n):
            if not in_mst[v] and (u == -1 or min_edge[v] < min_edge[u]):
                u = v
        in_mst[u] = True
        total_weight += min_edge[u]

        # Cập nhật min_edge cho các đỉnh kề
        for v in range(n):
            if not in_mst[v]:
                w = time_matrix[nodes[u]][nodes[v]]
                if w < min_edge[v]:
                    min_edge[v] = w

    return total_weight


def heuristic_nearest(
    current_idx: int,
    remaining: List[int],
    time_matrix: np.ndarray,
) -> float:
    """
    Heuristic đơn giản hơn: khoảng cách đến điểm gần nhất trong remaining.
    Admissible nhưng kém chính xác hơn MST.
    Dùng khi n nhỏ hoặc cần tốc độ nhanh hơn.
    """
    if not remaining:
        return 0.0
    return min(time_matrix[current_idx][r] for r in remaining)


# ============================================================
# A* SEARCH
# ============================================================

def astar_route(
    start_idx: int,
    candidates: List[int],
    places_df: pd.DataFrame,
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    time_limit_hours: float = 10.0,
    budget_limit_vnd: float = float('inf'),
    use_mst_heuristic: bool = True,
) -> Dict[str, Any]:
    """
    Tìm lộ trình tối ưu (thứ tự tham quan) trong một ngày bằng A*.

    Args:
        start_idx: Index điểm xuất phát (trong places_df, cũng là index ma trận)
        candidates: Danh sách index các điểm cần thăm trong ngày
        places_df: DataFrame 50 điểm đầu (vn_tourist_places.csv)
        time_matrix: Ma trận thời gian di chuyển (giờ), shape (50, 50)
        cost_matrix: Ma trận chi phí di chuyển (VND), shape (50, 50)
        time_limit_hours: Giới hạn thời gian trong ngày (mặc định 10 giờ)
        budget_limit_vnd: Giới hạn ngân sách di chuyển (VND)
        use_mst_heuristic: True = MST (tốt hơn), False = nearest neighbor (nhanh hơn)

    Returns:
        Dict:
            "route": List[int] — thứ tự index địa điểm tối ưu
            "total_time": float — tổng giờ (di chuyển + tham quan)
            "total_travel_cost": float — tổng VND di chuyển
            "total_entry_fee": float — tổng VND vé vào cổng
            "found": bool — tìm được lộ trình hay không
            "path_names": List[str] — tên các địa điểm theo thứ tự
    """
    if not candidates:
        return {"route": [], "total_time": 0, "total_travel_cost": 0,
                "total_entry_fee": 0, "found": True, "path_names": []}

    # Thêm start_idx vào visited ban đầu nếu start không phải điểm tham quan
    all_targets = frozenset(candidates)
    initial_visited = frozenset({start_idx}) if start_idx not in candidates else frozenset()

    initial_state = SearchState(
        current_idx=start_idx,
        visited=initial_visited,
        time_used=0.0,
        cost_used=0.0,
    )

    def is_goal(state: SearchState) -> bool:
        return all_targets.issubset(state.visited)

    def get_remaining(visited: FrozenSet[int]) -> List[int]:
        return [c for c in candidates if c not in visited]

    def compute_h(state: SearchState) -> float:
        remaining = get_remaining(state.visited)
        if use_mst_heuristic:
            return heuristic_mst(state.current_idx, remaining, time_matrix)
        return heuristic_nearest(state.current_idx, remaining, time_matrix)

    h0 = compute_h(initial_state)
    start_node = SearchNode(initial_state, parent=None, g=0.0, h=h0)

    open_list = []
    heapq.heappush(open_list, start_node)
    closed = {}  # state -> g value

    while open_list:
        node = heapq.heappop(open_list)
        state = node.state

        # Kiểm tra goal
        if is_goal(state):
            # Truy vết đường đi
            route = []
            cur = node
            while cur.parent is not None:
                route.append(cur.state.current_idx)
                cur = cur.parent
            route.reverse()

            entry_fees = sum(
                places_df.iloc[idx]["entry_fee_vnd"]
                for idx in route if idx < len(places_df)
            )
            names = [places_df.iloc[idx]["place_name"] for idx in route if idx < len(places_df)]

            return {
                "route": route,
                "total_time": state.time_used,
                "total_travel_cost": state.cost_used,
                "total_entry_fee": entry_fees,
                "found": True,
                "path_names": names,
            }

        # Kiểm tra closed list
        state_key = (state.current_idx, state.visited)
        if state_key in closed and closed[state_key] <= node.g:
            continue
        closed[state_key] = node.g

        # Expand: thử đi đến các điểm chưa thăm
        remaining = get_remaining(state.visited)
        for next_idx in remaining:
            travel_t = time_matrix[state.current_idx][next_idx]
            travel_c = cost_matrix[state.current_idx][next_idx]
            visit_t  = places_df.iloc[next_idx]["visit_duration_hours"] if next_idx < len(places_df) else 1.0

            new_time = state.time_used + travel_t + visit_t
            new_cost = state.cost_used + travel_c

            # Pruning: vượt giới hạn thời gian hoặc ngân sách
            if new_time > time_limit_hours or new_cost > budget_limit_vnd:
                continue

            new_visited = state.visited | frozenset({next_idx})
            new_state = SearchState(next_idx, new_visited, new_time, new_cost)

            new_state_key = (next_idx, new_visited)
            if new_state_key in closed and closed[new_state_key] <= node.g + travel_t + visit_t:
                continue

            h = compute_h(new_state)
            new_node = SearchNode(new_state, parent=node,
                                  g=node.g + travel_t + visit_t, h=h)
            heapq.heappush(open_list, new_node)

    # Không tìm được lộ trình hoàn chỉnh → trả về greedy tốt nhất
    return _greedy_fallback(start_idx, candidates, places_df, time_matrix,
                            cost_matrix, time_limit_hours)


def _greedy_fallback(
    start_idx: int,
    candidates: List[int],
    places_df: pd.DataFrame,
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    time_limit_hours: float,
) -> Dict[str, Any]:
    """Greedy nearest-neighbor fallback khi A* không tìm được đường."""
    current = start_idx
    remaining = list(candidates)
    route = []
    total_time = 0.0
    total_cost = 0.0

    while remaining:
        # Chọn điểm gần nhất chưa thăm
        nearest = min(remaining, key=lambda x: time_matrix[current][x])
        travel_t = time_matrix[current][nearest]
        visit_t = places_df.iloc[nearest]["visit_duration_hours"] if nearest < len(places_df) else 1.0

        if total_time + travel_t + visit_t > time_limit_hours:
            break

        total_time += travel_t + visit_t
        total_cost += cost_matrix[current][nearest]
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest

    entry_fees = sum(places_df.iloc[idx]["entry_fee_vnd"] for idx in route if idx < len(places_df))
    names = [places_df.iloc[idx]["place_name"] for idx in route if idx < len(places_df)]

    return {
        "route": route,
        "total_time": total_time,
        "total_travel_cost": total_cost,
        "total_entry_fee": entry_fees,
        "found": False,   # đánh dấu là fallback
        "path_names": names,
    }


# ============================================================
# PUBLIC API
# ============================================================

def find_optimal_daily_route(
    day_places_df: pd.DataFrame,
    places_df: pd.DataFrame,
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    time_limit_hours: float = 10.0,
    budget_vnd: float = float('inf'),
) -> Dict[str, Any]:
    """
    API chính: tìm thứ tự tham quan tối ưu trong một ngày.

    Args:
        day_places_df: Subset DataFrame các địa điểm trong ngày (từ CSP output)
        places_df: Full DataFrame 50 địa điểm (để tra index ma trận)
        time_matrix: Ma trận thời gian (50x50)
        cost_matrix: Ma trận chi phí (50x50)
        time_limit_hours: Giờ tối đa trong ngày
        budget_vnd: Ngân sách di chuyển trong ngày

    Returns:
        Dict với "route", "total_time", "total_travel_cost", "path_names"
    """
    # Lấy index trong ma trận cho từng địa điểm
    place_names = list(places_df["place_name"])

    candidates = []
    for _, row in day_places_df.iterrows():
        if row["place_name"] in place_names:
            idx = place_names.index(row["place_name"])
            candidates.append(idx)

    if not candidates:
        return {"route": [], "total_time": 0, "total_travel_cost": 0,
                "total_entry_fee": 0, "found": True, "path_names": []}

    start_idx = candidates[0]
    return astar_route(
        start_idx=start_idx,
        candidates=candidates,
        places_df=places_df,
        time_matrix=time_matrix,
        cost_matrix=cost_matrix,
        time_limit_hours=time_limit_hours,
        budget_limit_vnd=budget_vnd,
    )
