# Data Collection & Preprocessing Review
## AI Travel Planner & Recommender System

> **Mục đích tài liệu này:** Giải thích chi tiết toàn bộ quá trình thu thập, làm sạch, và tiền xử lý dữ liệu cho các thành viên còn lại của nhóm — bao gồm lý do lựa chọn từng dataset, cách xử lý sự khác biệt giữa các bộ dữ liệu, và cách đầu ra được chuẩn hoá để phục vụ các module AI.

---

## Mục lục

1. [Tổng quan kiến trúc dữ liệu](#1-tổng-quan-kiến-trúc-dữ-liệu)
2. [Thu thập dữ liệu — 5 Datasets](#2-thu-thập-dữ-liệu--5-datasets)
3. [Vấn đề cốt lõi: Mỗi dataset có chỉ số khác nhau](#3-vấn-đề-cốt-lõi-mỗi-dataset-có-chỉ-số-khác-nhau)
4. [Chiến lược xử lý từng dataset](#4-chiến-lược-xử-lý-từng-dataset)
5. [Feature Engineering — Tạo đầu vào cho AI](#5-feature-engineering--tạo-đầu-vào-cho-ai)
6. [Cấu trúc thư mục đầu ra](#6-cấu-trúc-thư-mục-đầu-ra)
7. [Hướng dẫn sử dụng cho các thành viên](#7-hướng-dẫn-sử-dụng-cho-các-thành-viên)

---

## 1. Tổng quan kiến trúc dữ liệu

```
Kaggle (5 datasets)
        │
        ▼
  [data/raw/]          ← Dữ liệu gốc, KHÔNG chỉnh sửa
        │
        ▼ clean_*()
  [data/cleaned/]      ← CSV đã làm sạch, tên cột chuẩn hoá
        │
        ▼ build_*()
  [data/features/]     ← .npy / .csv sẵn sàng nạp vào model AI
        │
        ▼ plt.savefig()
  [data/processed/]    ← Biểu đồ EDA (.png)
```

**Nguyên tắc thiết kế:**
- `data/raw/` — Không bao giờ sửa. Nếu cần chạy lại từ đầu, pipeline tự download lại.
- `data/cleaned/` — Mỗi file `.csv` tương ứng 1 dataset gốc, đã chuẩn hoá tên cột, đã xử lý missing.
- `data/features/` — Các file `.npy` (NumPy array) và `.csv` này là **đầu vào trực tiếp** cho các module của Thành viên 2 và 3.

---

## 2. Thu thập dữ liệu — 5 Datasets

### Phương thức thu thập

Toàn bộ 5 dataset được tải tự động từ **Kaggle** qua thư viện `opendatasets`, được gọi một lần duy nhất khi chạy notebook trên Google Colab:

```python
download_all_datasets(use_opendatasets=True)
```

Khi chạy lần đầu, Colab sẽ yêu cầu nhập **Kaggle username** và **API key** (lấy tại `kaggle.com/settings → API → Create New Token`). Từ lần 2 trở đi, nếu thư mục `data/raw/` đã có dữ liệu, bước này được bỏ qua.

### Bảng tổng hợp 5 datasets

| # | Tên dataset | Nguồn Kaggle | Kích thước | Phục vụ AI |
|---|---|---|---|---|
| 1 | **Vietnam Weather Data** | `vanviethieuanh/vietnam-weather-data` | ~181,960 rows, 40 tỉnh, 2009–2021 | (C) IF-THEN + (D) Bayesian Network |
| 2 | **515K Hotel Reviews Europe** | `jiashenliu/515k-hotel-reviews-data-in-europe` | ~515,000 rows | (E) ML — sentiment classification |
| 3 | **Travel Review Ratings (UCI)** | `ishbhms/travel-review-ratings` | 5,456 users × 24 categories | (E) ML — user classification |
| 4 | **Traveler Trip Data** | `rkiattisak/traveler-trip-data` | ~21,000 trips | (B) CSP — budget/time constraints |
| 5 | **Worldwide Travel Cities** | `furkanima/worldwide-travel-cities-ratings-and-climate` | 560 cities | (E) ML — city style classification |

### Lý do lựa chọn từng dataset

**Dataset 1 — Vietnam Weather:**
- Dữ liệu thực tế từ 40 tỉnh/thành Việt Nam, trải dài 12 năm (2009–2021) → đủ để ước lượng xác suất thời tiết theo mùa và theo tỉnh.
- Cần cho module (C): quyết định loại bỏ hoạt động outdoor khi trời mưa.
- Cần cho module (D): tính P(rain | province, month), P(hot | province, month).

**Dataset 2 — Hotel Reviews:**
- 515K reviews với điểm đánh giá từ 0–10 → nguồn nhãn lớn, không cần gán nhãn thủ công.
- Đủ dữ liệu text để huấn luyện TF-IDF + model phân loại sentiment.

**Dataset 3 — Travel Ratings (UCI):**
- Mỗi user có rating cho 24 loại hình du lịch (culture, nature, beach...) → hồ sơ sở thích người dùng dạng vector số.
- Phù hợp để phân cụm (K-Means) tạo nhóm traveler_type cho Decision Tree.

**Dataset 4 — Traveler Trips:**
- Có thông tin thực tế về chi phí chỗ ở, chi phí di chuyển, thời gian chuyến đi, điểm đến.
- Dùng để huấn luyện CSP với ràng buộc ngân sách và thời gian từ dữ liệu thực.

**Dataset 5 — World Cities:**
- Xếp hạng các thành phố theo nhiều tiêu chí (culture, adventure, nightlife...) → tham chiếu phân loại loại hình của từng điểm đến.

---

## 3. Vấn đề cốt lõi: Mỗi dataset có chỉ số khác nhau

Đây là thách thức chính trong phần data preprocessing. 5 datasets hoàn toàn **không có cấu trúc chung**:

| Dataset | Loại dữ liệu chính | Đơn vị đo | Phạm vi giá trị |
|---|---|---|---|
| Vietnam Weather | Khí tượng học | mm, °C, %, hPa, m/s | rain: 0–500mm; temp: 10–42°C; humidity: 30–100% |
| Hotel Reviews | Text + Score | Điểm 0–10 | Reviewer_Score: 2.5–10.0 (thực tế thấp nhất ~2.5) |
| Travel Ratings | Điểm đánh giá | Thang 1–5 (UCI) | 1.0 – 5.0 trên 24 categories |
| Traveler Trips | Tài chính + Thời gian | USD, ngày | Cost: $0–$50,000+; Duration: 1–365 ngày |
| World Cities | Chỉ số tổng hợp | Thang 0–100 hoặc 1–5 | Tuỳ thuộc cột |

**Vấn đề phát sinh:**
- Không thể so sánh trực tiếp score của Hotel Reviews (0–10) với Travel Ratings (1–5).
- Các cột weather có đơn vị vật lý (mm, °C) không thể đưa vào model học máy trực tiếp.
- Traveler Trips có chi phí dạng string (`"$1,200"`, `"1200.0"`, `"1200"`) — không nhất quán.
- Tên cột hoàn toàn khác nhau giữa các dataset, không có chuẩn chung.

### Giải pháp tổng thể: "Xử lý từng dataset độc lập, đầu ra đồng nhất"

Thay vì cố merge tất cả dữ liệu thành một bảng lớn (không khả thi do domain khác nhau), cách tiếp cận được chọn là:

1. **Làm sạch riêng biệt** từng dataset với hàm `clean_*()` chuyên biệt.
2. **Chuẩn hoá tên cột** về tên có ý nghĩa thống nhất trong phạm vi từng dataset.
3. **Tạo nhãn trung gian** để các dataset nói chuyện được với nhau (ví dụ: `sentiment_binary`, `budget_level`, `outdoor_suitable`).
4. **Feature Engineering** tạo ra các ma trận/vector chuẩn hoá phù hợp với từng loại AI algorithm.

---

## 4. Chiến lược xử lý từng dataset

### 4.1 Vietnam Weather — `clean_vietnam_weather()`

**Vấn đề gặp phải:**
- Tên cột không nhất quán giữa các phiên bản dataset trên Kaggle (có thể là `Max`, `TMAX`, `max_temp` tùy nguồn tải).
- Khoảng 5–15% giá trị nhiệt độ và độ ẩm bị thiếu.
- Không có cột `season` hay các nhãn phân loại thời tiết.

**Cách xử lý:**

```
Bước 1 — Chuẩn hoá tên cột (Column Mapping)
   "Max" / "TMAX" / "max_temp"  →  "temp_max"
   "Rain" / "precip" / "rain"   →  "rain_mm"
   "Humidity" / "humid"         →  "humidity"
   "Station" / "city"           →  "province"
   ...

Bước 2 — Parse datetime
   "date" → tách ra: year, month, day, season
   season: tháng 2-4 = spring | 5-7 = summer | 8-10 = autumn | 11-1 = winter

Bước 3 — Xử lý Missing Values
   Chiến lược: fillna bằng median THEO TỪNG TỈNH (group-wise median)
   Lý do: nhiệt độ Hà Nội và Cần Thơ hoàn toàn khác nhau
         → median toàn bộ sẽ sai lệch lớn
   
   df[col] = df.groupby("province")[col].transform(lambda x: x.fillna(x.median()))
   
   Nếu sau đó vẫn còn NaN (tỉnh có quá ít dữ liệu): fillna bằng median toàn bộ.

Bước 4 — Tạo nhãn phân loại (cho module C và D)
   is_rainy       : rain_mm > 1.0  → 0/1
   rain_level     : cut([-1,0,5,20,50,999]) → none/light/moderate/heavy/extreme
   is_hot         : temp_max > 35  → 0/1
   is_humid       : humidity > 80  → 0/1
   outdoor_suitable: rain_mm ≤ 5 AND temp_max ≤ 38 AND humidity ≤ 90 → 0/1
```

**Tại sao dùng ngưỡng này?**
- `rain_mm > 1.0` — Theo tiêu chuẩn WMO (World Meteorological Organization), ngày có mưa là ngày có lượng mưa > 1mm.
- `temp_max > 35` — Ngưỡng "nắng nóng" theo Trung tâm Khí tượng Thuỷ văn Việt Nam.
- `humidity > 80` — Độ ẩm trên 80% được coi là "oi bức", không thoải mái cho hoạt động ngoài trời.

---

### 4.2 Hotel Reviews — `clean_hotel_reviews()`

**Vấn đề gặp phải:**
- Có 2 cột review riêng biệt: `Positive_Review` và `Negative_Review`.
- Nhiều review chứa placeholder text như `"No Negative"`, `"Nothing"`, `"N A"` thay vì nội dung thực sự.
- `Reviewer_Score` trên thang 0–10, **khác hoàn toàn** với Travel Ratings (1–5).

**Cách xử lý:**

```
Bước 1 — Gộp 2 cột text thành 1
   full_review = Positive_Review + " " + Negative_Review
   Lý do: TF-IDF hoạt động tốt nhất với đoạn văn dài, có ngữ cảnh đầy đủ.

Bước 2 — Phát hiện placeholder text
   has_negative = NOT contains("no negative|nothing|none|na")
   has_positive = NOT contains("no positive|nothing|none|na")
   → Không XOÁ, chỉ đánh dấu để model có thể học được pattern này.

Bước 3 — Tạo nhãn Sentiment TỪ điểm số (KHÔNG cần gán tay)
   Reviewer_Score (0-10) → 4 nhãn:
     [0, 4)   → "negative"
     [4, 6)   → "neutral"
     [6, 8)   → "positive"
     [8, 10]  → "very_positive"
   
   sentiment_binary (cho binary classification):
     Score >= 7  → 1 (tích cực)
     Score <  7  → 0 (tiêu cực)

Bước 4 — Tạo thêm features kỹ thuật
   review_word_count  : số từ trong full_review
   review_char_count  : số ký tự trong full_review
   → Các features này có thể tương quan với chất lượng review.
```

**Lưu ý cho Thành viên 4 (ML):**
- File cleaned: `data/cleaned/hotel_reviews.csv`
- TF-IDF đã được tính sẵn: `data/features/review_tfidf.npz` (sparse matrix, 50K sample × 5000 features)
- Nhãn nhị phân: `data/features/review_labels.npy`
- Điểm gốc: `data/features/review_scores.npy`

---

### 4.3 Travel Ratings (UCI) — `clean_travel_ratings()`

**Vấn đề gặp phải:**
- Thang điểm 1–5, **khác với** Hotel Reviews (0–10) — KHÔNG thể cộng hay so sánh trực tiếp.
- Một số user có nhiều ô `NaN` (chưa đánh giá loại hình đó).
- Tên 24 cột là tên danh mục du lịch tiếng Anh dài và không đồng nhất.

**Cách xử lý:**

```
Bước 1 — Chuyển tất cả cột về numeric
   pd.to_numeric(..., errors="coerce")  ← lỗi → NaN

Bước 2 — Xử lý NaN bằng 0 (KHÔNG dùng median)
   Lý do: NaN ở đây nghĩa là "chưa đánh giá", KHÔNG phải "dữ liệu bị mất"
         → Thay bằng 0 mang ý nghĩa "không có kinh nghiệm" với loại hình đó
         → Median sẽ tạo ra thông tin giả.

Bước 3 — Tạo profile features cho mỗi user
   top_category  : loại hình được đánh giá cao nhất (idxmax)
   avg_rating    : rating trung bình trên tất cả loại hình
   rating_std    : độ lệch chuẩn rating → đo mức độ chọn lọc
   num_rated     : số loại hình đã đánh giá (> 0)
```

**Bước tiếp theo trong notebook (Feature Engineering):**
- `StandardScaler` → chuẩn hoá về mean=0, std=1.
- `KMeans(n_clusters=5)` → phân 5 nhóm traveler (culture lover, adventurer, beach lover, foodie, balanced).
- Kết quả lưu vào: `user_features.npy` (vector đã scale) + `user_cluster_labels.npy` (nhãn cluster).

---

### 4.4 Traveler Trips — `clean_traveler_trips()`

**Vấn đề gặp phải:**
- Chi phí lưu trú và di chuyển ở dạng string không chuẩn: `"$1,200"`, `"1200.50"`, `"1,200.00"`.
- Ngày bắt đầu/kết thúc ở nhiều định dạng khác nhau.
- Có trips với `Duration = 0` hoặc `Cost = 0` (dữ liệu thiếu thực sự, không phải miễn phí).

**Cách xử lý:**

```
Bước 1 — Parse datetime
   "Start date", "End date" → pd.to_datetime(..., errors="coerce")

Bước 2 — Làm sạch cột chi phí (Numeric Cleaning)
   Regex: loại bỏ tất cả ký tự không phải số hoặc dấu chấm
   df[col] = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
   Sau đó: pd.to_numeric(..., errors="coerce")

Bước 3 — Tạo cột tổng hợp
   total_cost    = Accommodation cost + Transportation cost
   cost_per_day  = total_cost / Duration (days)   [tránh chia cho 0]
   
   budget_level (nhãn định tính):
     $0–500       → "budget"
     $500–1,500   → "mid_range"
     $1,500–5,000 → "premium"
     > $5,000     → "luxury"

Bước 4 — Loại bỏ rows không hợp lệ
   dropna(subset=[cost_cols]) → loại bỏ trips không có thông tin chi phí
```

**Lưu ý cho Thành viên 2 (CSP):**
- File chính cho CSP: `data/cleaned/hotel_bookings.csv` (118K bookings, có `total_nights`, `total_cost`, `budget_level`, `season`)
- File bổ sung: `data/cleaned/traveler_trips.csv` (21K trips, có `Duration (days)`, `cost_per_day` nếu cần tham chiếu thêm)
- Cột `budget_level` và `total_cost` trong `hotel_bookings.csv` dùng để thiết lập ràng buộc ngân sách trong CSP.
- Cột `total_nights` và `season` dùng để ràng buộc thời gian và mùa du lịch.

---

### 4.5 World Cities — `clean_world_cities()`

**Vấn đề gặp phải:**
- Nhiều cột rating dùng thang khác nhau tuỳ version dataset (0–100 hoặc 1–5).
- Có thể có cột lat/lng với tên khác nhau (`Latitude`, `latitude`, `lat`).

**Cách xử lý:**

```
Bước 1 — Tự động phát hiện cột rating
   Các cột có tên chứa: "culture", "adventure", "nature", "beach",
   "nightlife", "cuisine", "wellness", "urban", "rating", "score"
   → Chuyển về numeric, fillna = 0

Bước 2 — Chuẩn hoá lat/lng
   Thử cả 3 tên phổ biến: "Latitude"/"latitude"/"lat", "Longitude"/"longitude"/"lng"
   → Chuyển về numeric
```

---

## 5. Feature Engineering — Tạo đầu vào cho AI

Đây là bước quan trọng nhất: **biến dữ liệu thô đã làm sạch thành các cấu trúc dữ liệu cụ thể mà các thuật toán AI cần**.

### 5.1 Ma trận khoảng cách, chi phí, thời gian di chuyển

**Dùng cho:** (A) Thuật toán A\*, (B) CSP

Vì không có dataset nào chứa thông tin khoảng cách giữa các điểm du lịch Việt Nam, ta **tự tính** từ toạ độ địa lý:

```
Công thức Haversine:
  d = 2R × arctan2(√a, √(1−a))
  a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
  R = 6,371 km (bán kính Trái Đất)
```

Kết quả là 3 ma trận vuông kích thước **30×30** (30 điểm du lịch nổi tiếng Việt Nam):

| File | Nội dung | Đơn vị | Ghi chú |
|---|---|---|---|
| `distance_matrix.npy` | Khoảng cách Haversine | km | Đường chim bay |
| `cost_matrix.npy` | Chi phí di chuyển ước tính | VND | = distance × 3,000 VND/km |
| `travel_time_matrix.npy` | Thời gian di chuyển ước tính | giờ | = distance / 40 km/h |

**Hằng số được chọn có chủ đích:**
- **3,000 VND/km** — Chi phí xe khách/limousine trung bình tuyến liên tỉnh Việt Nam.
- **40 km/h** — Tốc độ trung bình thực tế trên đường quốc lộ Việt Nam (có tính kẹt xe, đèo dốc).

> **Lưu ý quan trọng cho Thành viên 2:** Đây là ước tính. Nếu cần độ chính xác cao hơn, có thể thay bằng Google Maps Distance Matrix API. Nhưng cho mục đích học thuật, các hệ số này đủ hợp lý.

### 5.2 Places DataFrame

**Dùng cho:** (A) A\*, (B) CSP, (C) IF-THEN

File `data/features/vn_tourist_places.csv` chứa thông tin của 30 điểm du lịch, được **hard-code từ nguồn thực tế** (Wikipedia, các trang du lịch) vì Kaggle không có dataset Việt Nam chi tiết:

| Cột | Ý nghĩa | Ví dụ |
|---|---|---|
| `place_name` | Tên địa danh | "Ha Long Bay" |
| `latitude`, `longitude` | Toạ độ GPS | 20.9101, 107.1839 |
| `category` | Loại hình | culture / nature / beach / adventure / entertainment |
| `province` | Tỉnh/Thành phố | "Quang Ninh" |
| `entry_fee_vnd` | Phí vào cổng (VND) | 0 (miễn phí) hoặc 200,000 |
| `visit_duration_hours` | Thời gian tham quan ước tính | 2.0 hoặc 3.0 |
| `opening_hour` / `closing_hour` | Giờ mở/đóng cửa | 7 – 17 |

### 5.3 Bảng xác suất thời tiết

**Dùng cho:** (D) Bayesian Network

Được tính từ 12 năm dữ liệu thời tiết bằng cách group-by `(province, month)`:

```python
P(rain | province, month)    = is_rainy.mean()      per group
P(outdoor_ok | prov, month)  = outdoor_suitable.mean() per group
P(hot | province, month)     = is_hot.mean()         per group
P(humid | province, month)   = is_humid.mean()       per group
```

File `data/features/weather_probabilities.csv` có cấu trúc:

```
province    | month | p_rain | p_outdoor_ok | p_hot | p_humid
Ha Noi      |   1   |  0.12  |    0.75      | 0.00  |  0.45
Ha Noi      |   2   |  0.18  |    0.70      | 0.00  |  0.52
...
```

**Cách Thành viên 3 sử dụng:**
- Truy vấn: `P(rain | "Da Nang", month=6)` → lấy giá trị `p_rain` từ bảng
- Kết hợp với Naive Bayes để tính xác suất chuyến đi thành công.

### 5.4 TF-IDF Text Features

**Dùng cho:** (E) ML — Sentiment Classification

Vì 515K reviews quá lớn để xử lý toàn bộ trong Colab (giới hạn RAM ~12GB), ta **sample 50,000 reviews**:

```
50,000 reviews  →  TfidfVectorizer(max_features=5000, ngram_range=(1,2))
                →  Sparse matrix 50,000 × 5,000
```

Tại sao `ngram_range=(1,2)`:
- Unigrams (`"clean"`, `"dirty"`) nắm bắt từ đơn.
- Bigrams (`"very clean"`, `"not good"`) nắm bắt sắc thái phủ định và nhấn mạnh.

File đầu ra:
- `review_tfidf.npz` — Sparse matrix (dùng scipy.sparse để tiết kiệm RAM)
- `review_labels.npy` — Nhãn 0/1 (negative/positive)
- `review_scores.npy` — Điểm gốc 0–10 (cho regression nếu cần)

### 5.5 User Feature Vectors (K-Means Clustering)

**Dùng cho:** (E) ML — User Classification

```
5,456 users × 24 rating categories
        │
        ▼ StandardScaler (zero mean, unit variance)
5,456 × 24 normalized matrix
        │
        ▼ KMeans(k=5)
5 clusters: culture / adventure / nature / beach / balanced (tên đặt dựa trên centroid)
```

Tại sao `StandardScaler` trước KMeans:
- Nếu không scale, user có nhiều rating cao sẽ bị ảnh hưởng không đáng bởi các category phổ biến hơn.
- KMeans tính khoảng cách Euclidean → đơn vị phải đồng nhất.

---

## 6. Cấu trúc thư mục đầu ra

Sau khi chạy toàn bộ notebook, cấu trúc thư mục sẽ như sau:

```
data/
├── raw/
│   ├── vietnam-weather-data/          ← Từ Kaggle
│   ├── 515k-hotel-reviews-data.../    ← Từ Kaggle
│   ├── travel-review-ratings/         ← Từ Kaggle
│   ├── traveler-trip-data/            ← Từ Kaggle
│   └── worldwide-travel-cities-.../  ← Từ Kaggle
│
├── cleaned/
│   ├── vietnam_weather.csv            ← 181K rows, cột chuẩn hoá
│   ├── hotel_reviews.csv              ← 515K rows, có sentiment labels
│   ├── travel_ratings.csv             ← 5,456 rows, có top_category
│   ├── travel_ratings_clustered.csv   ← Có thêm cột cluster (0-4)
│   ├── traveler_trips.csv             ← ~21K rows, có total_cost, budget_level
│   └── world_cities.csv               ← 560 rows
│
├── features/
│   ├── distance_matrix.npy            ← shape (30, 30), đơn vị km
│   ├── cost_matrix.npy                ← shape (30, 30), đơn vị VND
│   ├── travel_time_matrix.npy         ← shape (30, 30), đơn vị giờ
│   ├── vn_tourist_places.csv          ← 30 điểm, đầy đủ thông tin
│   ├── weather_probabilities.csv      ← province × month → xác suất
│   ├── review_tfidf.npz               ← sparse (50000, 5000)
│   ├── review_labels.npy              ← shape (50000,), dtype int
│   ├── review_scores.npy              ← shape (50000,), dtype float
│   ├── tfidf_features.csv             ← tên 5000 features TF-IDF
│   ├── user_features.npy              ← shape (5456, 24), đã StandardScaler
│   └── user_cluster_labels.npy        ← shape (5456,), dtype int (0-4)
│
└── processed/
    ├── eda_weather.png
    ├── eda_weather_corr.png
    ├── eda_outdoor_season.png
    ├── eda_reviews.png
    ├── eda_ratings.png
    ├── eda_ratings_corr.png
    ├── eda_trips.png
    ├── distance_matrix_heatmap.png
    ├── places_map.png
    ├── weather_rain_prob_heatmap.png
    └── user_clusters_pca.png
```

---

## 7. Hướng dẫn sử dụng cho các thành viên

### Thành viên 2 — A* Search (A) + CSP Solver (B)

**File cần tạo:** `modules/search.py`, `modules/csp_solver.py`, `modules/planner.py`

**Cần load:**

```python
import numpy as np
import pandas as pd

# Ma trận khoảng cách, chi phí, thời gian — cho A* và CSP
dist_matrix = np.load('data/features/distance_matrix.npy')    # (30, 30) km
cost_matrix = np.load('data/features/cost_matrix.npy')        # (30, 30) VND
time_matrix = np.load('data/features/travel_time_matrix.npy') # (30, 30) hours

# Thông tin điểm du lịch — cho state representation trong A*
df_places = pd.read_csv('data/features/vn_tourist_places.csv')

# Thứ tự tên địa điểm (index 0-29 tương ứng với ma trận)
place_names = df_places['place_name'].tolist()

# Dữ liệu đặt phòng — cho ràng buộc CSP (ngân sách, thời gian lưu trú)
df_bookings = pd.read_csv('data/cleaned/hotel_bookings.csv')
```

**Cấu trúc `df_places` (30 hàng × 8 cột):**

| place_name | latitude | longitude | category | province | entry_fee_vnd | visit_duration_hours | opening_hour | closing_hour |
|---|---|---|---|---|---|---|---|---|
| Ha Long Bay | 20.9101 | 107.1839 | nature | Quang Ninh | 0 | 3.0 | 7 | 17 |
| ... | | | | | | | | |

**Cấu trúc `df_bookings` (118K hàng — dữ liệu thực tế cho CSP):**

| Cột | Mô tả |
|---|---|
| `adr` | Average Daily Rate (USD/đêm) |
| `total_nights` | Tổng số đêm lưu trú |
| `total_cost` | Chi phí ước tính = adr × total_nights |
| `budget_level` | `budget` / `mid_range` / `premium` / `luxury` |
| `season` | `spring` / `summer` / `autumn` / `winter` |
| `arrival_date` | Ngày đến (datetime) |
| `customer_type` | Loại khách (Transient, Contract, Group...) |

**Ví dụ truy vấn:**
```python
# Khoảng cách và chi phí di chuyển giữa hai điểm
i = place_names.index("Ha Long Bay")
j = place_names.index("Imperial City Hue")
print(f"Distance: {dist_matrix[i, j]:.1f} km")
print(f"Travel time: {time_matrix[i, j]:.1f} hours")
print(f"Travel cost: {cost_matrix[i, j]:,.0f} VND")

# Lọc điểm theo giờ mở cửa (cho CSP ràng buộc thời gian)
open_places = df_places[(df_places['opening_hour'] <= 8) & (df_places['closing_hour'] >= 18)]

# Phân bố ngân sách thực tế (tham chiếu cho CSP constraints)
budget_dist = df_bookings['budget_level'].value_counts()
avg_cost_per_night = df_bookings.groupby('budget_level')['adr'].mean()
```

**State representation cho A\*:**
```python
# State = (vị_trí_hiện_tại, tập_điểm_đã_thăm, thời_gian_còn_lại, ngân_sách_còn_lại)
# Ví dụ: (2, frozenset({0, 1}), 6.5, 1_500_000)
# Nghĩa: đang ở điểm index 2, đã thăm điểm 0 và 1, còn 6.5 giờ, còn 1.5M VND
```

---

### Thành viên 3 — IF-THEN Rules (C) + Bayesian Network (D)

**File cần tạo:** `modules/knowledge_base.py`, `modules/bayesian_net.py`

**Cần load:**

```python
import numpy as np
import pandas as pd

# === Cho Bayesian Network (D) ===
weather_probs = pd.read_csv('data/features/weather_probabilities.csv')
# Cấu trúc: province | month | p_rain | p_outdoor_ok | p_hot | p_humid

# Truy vấn xác suất có điều kiện
p_rain = weather_probs[
    (weather_probs['province'].str.contains('Da Nang', case=False)) &
    (weather_probs['month'] == 6)
]['p_rain'].values[0]
# → P(mưa | Đà Nẵng, tháng 6)

# Dữ liệu thời tiết gốc (nếu cần tính lại hoặc kiểm tra)
df_weather = pd.read_csv('data/cleaned/vietnam_weather.csv')
# Cột quan trọng: province, month, season, is_rainy, outdoor_suitable, is_hot, is_humid, rain_mm, temp_max

# === Cho IF-THEN Rules (C) ===
df_places = pd.read_csv('data/features/vn_tourist_places.csv')
# Cột quan trọng: place_name, category, entry_fee_vnd, opening_hour, closing_hour
```

**Bảng `weather_probabilities.csv` — cấu trúc đầy đủ:**

```
province    | month | p_rain | p_outdoor_ok | p_hot | p_humid
Ha Noi      |   1   |  0.12  |    0.75      | 0.00  |  0.45
Ha Noi      |   6   |  0.48  |    0.42      | 0.65  |  0.82
Da Nang     |   9   |  0.62  |    0.31      | 0.45  |  0.85
...
```

**Ví dụ cài đặt Knowledge Base — IF-THEN Rules (C):**
```python
def apply_rules(places_df, weather_info, user_profile):
    result = places_df.copy()

    # Rule 1: IF mưa nặng → loại bỏ outdoor
    if weather_info['is_rainy'] and weather_info['rain_mm'] > 10:
        result = result[~result['category'].isin(['nature', 'beach', 'adventure'])]

    # Rule 2: IF ngân sách thấp → ưu tiên miễn phí
    if user_profile['budget_level'] == 'budget':
        result = result.sort_values('entry_fee_vnd')

    # Rule 3: IF gia đình có trẻ em → loại bỏ nightlife
    if user_profile['group_type'] == 'family':
        result = result[result['category'] != 'entertainment']

    return result
```

**Ví dụ cài đặt Bayesian Network (D):**
```python
def get_weather_probability(province, month, weather_probs_df):
    row = weather_probs_df[
        (weather_probs_df['province'].str.contains(province, case=False)) &
        (weather_probs_df['month'] == month)
    ]
    return {
        'p_rain':       row['p_rain'].values[0],
        'p_outdoor_ok': row['p_outdoor_ok'].values[0],
        'p_hot':        row['p_hot'].values[0],
        'p_humid':      row['p_humid'].values[0],
    }
```

**Ngưỡng định nghĩa nhãn thời tiết (để cài đặt rules đúng):**
- `is_rainy` → `rain_mm > 1.0` (tiêu chuẩn WMO)
- `outdoor_suitable` → `rain_mm ≤ 5 AND temp_max ≤ 38 AND humidity ≤ 90`
- `is_hot` → `temp_max > 35` (ngưỡng nắng nóng theo KTTV Việt Nam)
- `is_humid` → `humidity > 80`

---

### Thành viên 4 — Decision Tree + Naive Bayes (E) + Tích hợp hệ thống

**File cần tạo:** `modules/ml_models.py` + hoàn thiện notebook chính (`notebooks/main_notebook.ipynb`)

**Cần load:**

```python
import numpy as np
import pandas as pd
from scipy import sparse

# === Cho Sentiment Classification (E) — Naive Bayes ===
X_tfidf = sparse.load_npz('data/features/review_tfidf.npz')  # (50000, 5000) sparse
y_labels = np.load('data/features/review_labels.npy')         # (50000,) binary (0/1)
y_scores = np.load('data/features/review_scores.npy')         # (50000,) điểm 0-10

# === Cho User Classification (E) — Decision Tree ===
X_users = np.load('data/features/user_features.npy')          # (5456, 24) đã StandardScale
y_clusters = np.load('data/features/user_cluster_labels.npy') # (5456,) nhãn 0-4

# Tên cột 24 categories (dùng làm feature names cho Decision Tree)
df_ratings = pd.read_csv('data/cleaned/travel_ratings_clustered.csv')
feature_cols = [c for c in df_ratings.columns
                if c not in ['User Id', 'cluster', 'top_category',
                             'avg_rating', 'rating_std', 'num_rated']]
```

**Lưu ý quan trọng về thang điểm:**

| Nguồn | Thang điểm | Đã xử lý thành |
|---|---|---|
| Hotel Reviews (`Reviewer_Score`) | 0.0 – 10.0 | `sentiment_binary` (0/1, ngưỡng 7.0) |
| Travel Ratings (UCI) | 1.0 – 5.0 | Đã StandardScale về mean=0, std=1 trong `user_features.npy` |
| World Cities | 0–100 hoặc 1–5 | Chỉ chuyển về numeric, chưa scale thêm |

> **KHÔNG** so sánh điểm giữa các dataset này trực tiếp. Mỗi dataset có domain độc lập.

**Phân loại 5 nhóm khách du lịch (nhãn cluster 0–4):**
```python
# Tên nhóm đặt dựa trên centroid K-Means — kiểm tra lại bằng kmeans.cluster_centers_
CLUSTER_NAMES = {
    0: "culture_lover",    # Museums, Theatres, Art galleries, Monuments
    1: "adventure_seeker", # Parks, Resorts, Beaches, View points
    2: "nature_eco",       # Zoos, Gardens, Parks, Beaches
    3: "foodie_social",    # Restaurants, Cafes, Pubs/bars, Malls
    4: "balanced"          # Rating trải đều các loại hình
}
```

**Quy trình huấn luyện gợi ý:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Decision Tree: phân loại nhóm khách ---
X_tr, X_te, y_tr, y_te = train_test_split(
    X_users, y_clusters, test_size=0.2, random_state=42
)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_tr, y_tr)
print(classification_report(y_te, dt.predict(X_te)))

# --- Naive Bayes: phân loại sentiment review ---
# X_tfidf là sparse matrix → dùng MultinomialNB
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_tfidf, y_labels, test_size=0.2, random_state=42
)
nb = MultinomialNB()
nb.fit(X_tr2, y_tr2)
print(classification_report(y_te2, nb.predict(X_te2)))
```

**Hướng dẫn tích hợp hệ thống (pipeline end-to-end):**
```
Input từ user:
  province, budget (VND), num_days, group_type (solo/couple/family/group)

Pipeline:
  1. ML — Decision Tree (TV4): phân loại user → traveler_type (cluster 0-4)
  2. Bayes (TV3):              dự đoán thời tiết → p_rain, p_outdoor_ok
  3. IF-THEN Rules (TV3):     lọc địa điểm phù hợp (thời tiết + ngân sách + nhóm)
  4. CSP — Backtracking (TV2): kiểm tra ràng buộc (budget, total_nights, giờ mở cửa)
  5. A* Search (TV2):          tìm lộ trình tối ưu trên đồ thị 30 điểm du lịch

Output: lịch trình chi tiết từng ngày + giải thích lý do chọn từng địa điểm
```

---

## Tóm tắt các quyết định thiết kế quan trọng

| Quyết định | Lý do |
|---|---|
| Dùng **group-wise median** cho missing weather | Khí hậu mỗi tỉnh khác nhau — median toàn quốc không phù hợp |
| Fillna = **0** cho Travel Ratings | NaN nghĩa là "chưa đánh giá", không phải missing |
| **Regex** để parse cost string | Kaggle data thực tế không đồng nhất về format số tiền |
| **Haversine** thay vì Google Maps | Không cần API key, đủ chính xác cho mục đích học thuật |
| Sample **50K** cho TF-IDF | Colab miễn phí giới hạn ~12GB RAM; 50K × 5000 sparse ≈ vài trăm MB |
| K-Means **k=5** cho user cluster | 5 nhóm traveler phổ biến trong nghiên cứu du lịch: culture, adventure, nature/eco, beach/resort, mixed |
| `StandardScaler` trước KMeans | KMeans nhạy cảm với scale — bắt buộc phải chuẩn hoá trước |
| Tách riêng `cleaned/` và `features/` | `cleaned/` = dữ liệu gốc đã lọc (có thể dùng lại); `features/` = đặc trưng đã tính (dùng trực tiếp cho model) |
