# Summary - Hệ Thống Lập Kế Hoạch & Gợi Ý Du Lịch Tối Ưu Bằng AI

## 1. Ý tưởng 

### Ý tưởng: **AI Travel Planner & Recommender System**

Xây dựng hệ thống AI lập kế hoạch du lịch thông minh, cá nhân hoá lịch trình dựa trên ngân sách, thời gian, sở thích người dùng, và điều kiện thời tiết. Hệ thống tích hợp cả 5 thành phần AI theo yêu cầu đề bài:

| Thành phần | Kết hợp từ | Mô tả |
|---|---|---|
| **(A) Tìm kiếm** | (UCS) + (A*) | Dùng **A*** với heuristic kết hợp chi phí + thời gian để tìm lộ trình tối ưu giữa các điểm du lịch. State = (vị trí hiện tại, tập điểm đã thăm, thời gian còn lại, ngân sách còn lại). |
| **(B) CSP** | (ràng buộc ngân sách/giờ) | **Backtracking + Forward Checking** để sắp xếp lịch trình thoả mãn: tổng chi phí <= ngân sách, giờ mở/đóng cửa, thời gian di chuyển giữa các điểm, số ngày giới hạn. |
| **(C) Suy luận tri thức** | (rule-based) | Hệ luật **IF-THEN**: IF trời mưa → loại bỏ hoạt động outdoor; IF ngân sách < ngưỡng → ưu tiên địa điểm miễn phí; IF người dùng là gia đình → loại bỏ nightlife. |
| **(D) Mạng Bayes** | (thời tiết) + #6 (sở thích user) | **Bayesian Network** mô hình hoá: P(mưa \| mùa, tỉnh, tháng), P(user thích địa điểm \| loại hình, rating, thời tiết). |
| **(E) Học máy** | (Decision Tree) + #6 (phân loại user) | **Decision Tree** phân loại người dùng thành nhóm (culture lover, adventure seeker, foodie...) + **Naive Bayes** dự đoán rating/đánh giá. |

---

## 2. Yêu cầu PDF

| Yêu cầu PDF | Đáp ứng | Chi tiết |
|---|---|---|
| (A) Biểu diễn & Tìm kiếm - State/Action/Goal/Cost + thuật toán | **Có** | State = (location, visited, time_left, budget_left). Action = di chuyển đến điểm kế tiếp. Goal = thăm tất cả điểm mong muốn. Cost = chi phí + thời gian. Thuật toán: **A***. |
| (B) Heuristic hoặc CSP | **Có** | Heuristic function cho A* + CSP backtracking cho ràng buộc lịch trình. |
| (C) Biểu diễn & Suy luận tri thức | **Có** | Hệ luật IF-THEN cho lọc hoạt động theo thời tiết, ngân sách, nhóm người dùng. |
| (D) Mạng Bayes / Xác suất | **Có** | Bayesian Network dự đoán thời tiết và xác suất user thích địa điểm. |
| (E) Học máy | **Có** | Decision Tree + Naive Bayes cho phân loại user và dự đoán rating. |
| Hệ thống tích hợp (L.O.4) | **Có** | Input (sở thích user, ngân sách, ngày) → AI xử lý → Output (lịch trình tối ưu có giải thích). |
| Front-end Google Colab | **Có** | Notebook chạy Runtime → Run All, không mount cloud cá nhân. |

**Kết luận**: Dự án đáp ứng đầy đủ cả **5/5 thành phần** bắt buộc.

---

## 3. Phân công công việc nhóm

### Thành viên 1 — Thu thập, Lọc & Tiền xử lý Dữ liệu

| Công việc | Mô tả | Kết quả đầu ra |
|---|---|---|
| Thu thập dataset | Tải và tổng hợp các dataset từ Kaggle/nguồn công khai | Thư mục `data/raw/` chứa dữ liệu gốc |
| Lọc & làm sạch dữ liệu | Xử lý missing values, duplicates, outliers; lọc dữ liệu Việt Nam từ dataset toàn cầu | Thư mục `data/cleaned/` |
| Tiền xử lý & Feature engineering | Chuẩn hoá, encode categorical, tạo features mới (khoảng cách giữa các điểm, nhãn thời tiết...) | Thư mục `data/processed/` + file `.npy`/`.h5` |
| Xây dựng data pipeline | Viết module Python tự động download + xử lý dữ liệu trong Colab | `modules/data_pipeline.py` |
| EDA (Exploratory Data Analysis) | Phân tích thống kê, trực quan hoá phân bố dữ liệu | Notebook EDA + biểu đồ |

### Thành viên 2 — Thuật toán Tìm kiếm + CSP + Suy luận tri thức

| Công việc | Mô tả | Kết quả đầu ra |
|---|---|---|
| (A) Thuật toán A* | Cài đặt A* trên đồ thị địa điểm du lịch; thiết kế heuristic function | `modules/search.py` |
| (B) CSP Solver | Cài đặt backtracking + forward checking cho ràng buộc lịch trình | `modules/csp_solver.py` |
| (C) Hệ luật IF-THEN | Xây dựng knowledge base luật suy luận (thời tiết, ngân sách, nhóm user) | `modules/knowledge_base.py` |
| Tích hợp A+B+C | Kết nối 3 module thành pipeline: input → lọc luật → CSP → A* → output lịch trình | `modules/planner.py` |

### Thành viên 3 — Mạng Bayes + Học máy + Báo cáo

| Công việc | Mô tả | Kết quả đầu ra |
|---|---|---|
| (D) Bayesian Network | Xây dựng mạng Bayes cho dự đoán thời tiết + sở thích user | `modules/bayesian_net.py` |
| (E) Decision Tree + Naive Bayes | Huấn luyện model phân loại user + dự đoán rating | `modules/ml_models.py` |
| Tích hợp hệ thống | Kết nối tất cả module thành hệ thống hoàn chỉnh trên Colab | Notebook chính |
| Viết báo cáo | Báo cáo PDF 10-20 trang theo cấu trúc yêu cầu | `reports/report.pdf` (Overleaf) |

---

## 4. Datasets đã Research (Phần việc của Thành viên 1)

### 4.1. Tổng quan datasets cần thu thập

Hệ thống cần 4 nhóm dữ liệu chính:

| Nhóm dữ liệu | Phục vụ thành phần | Mô tả |
|---|---|---|
| Địa điểm du lịch (POI) | (A) Tìm kiếm, (B) CSP | Toạ độ, tên, loại hình, rating, giờ mở cửa, phí vào cổng |
| Thời tiết / Khí hậu | (C) Suy luận, (D) Bayes | Nhiệt độ, độ ẩm, lượng mưa, mùa, theo tỉnh/tháng |
| Đánh giá người dùng | (E) Học máy | Review, rating, quốc tịch, loại du lịch, sở thích |
| Lộ trình / Chi phí | (A) Tìm kiếm, (B) CSP | Khoảng cách, thời gian di chuyển, chi phí vận chuyển |

### 4.2. Danh sách datasets cụ thể

#### Dataset 1: Foursquare Open Source Places (Toàn cầu, lọc Việt Nam)
- **Nguồn**: https://opensource.foursquare.com/os-places/ | Hugging Face: foursquare/fsq-os-places
- **Kích thước**: **100,000,000+ POIs** toàn cầu (~10.6 GB Parquet). Lọc VN ước tính: hàng trăm nghìn records.
- **Cột chính**: `fsq_place_id`, `name`, `latitude`, `longitude`, `address`, `locality`, `region`, `country`, `fsq_category_ids`, `fsq_category_labels`
- **Phục vụ**: **(A) Tìm kiếm** — tọa độ POI để xây dựng đồ thị khoảng cách cho A*
- **License**: Apache 2.0
- **Ghi chú**: Filter `country='VN'` để lấy dữ liệu Việt Nam. Categories bao gồm nhà hàng, khách sạn, điểm du lịch, bảo tàng, v.v.

#### Dataset 2: Vietnam Weather Data (40 tỉnh thành, 2009-2021)
- **Nguồn**: https://www.kaggle.com/datasets/vanviethieuanh/vietnam-weather-data
- **Kích thước**: **181,960 records**, 1.97 MB
- **Cột chính**: `province`, `max` (nhiệt độ cao nhất), `min` (nhiệt độ thấp nhất), `wind` (tốc độ gió km/h), `wind_d` (hướng gió), `rain` (lượng mưa mm), `humidi` (độ ẩm %), `cloud` (mây %), `pressure` (áp suất), `date`
- **Phục vụ**: **(C) Suy luận IF-THEN** + **(D) Mạng Bayes**
- **License**: CC0 Public Domain
- **Ghi chú**: Dữ liệu Việt Nam gốc, bao gồm Hà Nội, TP.HCM, Huế, Đà Lạt, Nha Trang, Hải Phòng, Cần Thơ và 33 tỉnh khác. Dùng để tính P(mưa | tỉnh, tháng), xây luật IF mưa > 10mm THEN loại outdoor.

#### Dataset 3: 515K Hotel Reviews Data in Europe
- **Nguồn**: https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
- **Kích thước**: **515,000 reviews** của 1,493 khách sạn, 47.3 MB
- **Cột chính**: `Hotel_Address`, `Average_Score`, `Hotel_Name`, `Reviewer_Nationality`, `Negative_Review`, `Positive_Review`, `Reviewer_Score`, `Tags`, `lat`, `lng`
- **Phục vụ**: **(E) Học máy** — phân loại sentiment, dự đoán rating
- **License**: Public
- **Ghi chú**: Dataset lớn, phù hợp train Decision Tree / Naive Bayes. Cần tiền xử lý text (TF-IDF, word count).

#### Dataset 4: Travel Review Ratings (UCI - Google Reviews)
- **Nguồn**: https://www.kaggle.com/datasets/ishbhms/travel-review-ratings
- **Kích thước**: **5,456 users**, 24 cột rating (thang 1-5)
- **Cột chính**: Rating cho 24 loại hình: `Churches`, `Resorts`, `Beaches`, `Parks`, `Theatres`, `Museums`, `Malls`, `Zoos`, `Restaurants`, `Pubs/bars`, `Hotels`, `Art galleries`, `Cafes`, `View points`, `Monuments`, `Gardens`, v.v.
- **Phục vụ**: **(E) Học máy** — phân loại user thành nhóm (culture lover, adventure seeker, foodie...)
- **License**: CC BY 4.0
- **Ghi chú**: Mỗi dòng là một user + rating 24 loại hình. Dùng Decision Tree/Naive Bayes phân loại traveler type.

#### Dataset 5: Hotel Booking Demand (thay thế Traveler Trip Dataset)
- **Nguồn**: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- **Kích thước**: **119,390 bookings**, 32 cột
- **Cột chính**: `hotel` (Resort/City), `is_canceled`, `lead_time`, `arrival_date_year/month/day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `meal`, `country`, `market_segment`, `distribution_channel`, `adr` (Average Daily Rate), `customer_type`, `deposit_type`, `reservation_status`
- **Phục vụ**: **(B) CSP** — ràng buộc ngân sách (ADR, total_cost), thời gian lưu trú (total_nights), loại khách hàng, mùa du lịch
- **License**: CC BY 4.0
- **Ghi chú**: Thay thế Traveler Trip Dataset (chỉ có 139 rows thực tế, quá ít). Dataset mới có 119K rows, phong phú hơn cho phân tích budget/constraints.

#### Dataset 6: Global Daily Climate Data (1833-nay)
- **Nguồn**: https://www.kaggle.com/datasets/guillemservera/global-daily-climate-data
- **Kích thước**: **~1,250 thành phố**, 223.1 MB (Parquet)
- **Cột chính**: `city_name`, `date`, `season`, `avg_temp_c`, `min_temp_c`, `max_temp_c`, `precipitation_mm`, `avg_wind_speed_kmh`, `avg_sea_level_pres_hpa`, `sunshine_total_min`
- **Phục vụ**: **(D) Mạng Bayes** — dữ liệu lớn để tính xác suất có điều kiện
- **License**: CC BY-NC 4.0
- **Ghi chú**: Time series dài, ước lượng xác suất đáng tin cậy: P(rain | season, city).

#### Dataset 7: Dynamic Tourism Route Dataset (DTRD)
- **Nguồn**: https://www.kaggle.com/datasets/ziya07/dynamic-tourism-route-dataset-dtrd
- **Kích thước**: Dữ liệu route có cấu trúc
- **Cột chính**: Toạ độ điểm du lịch, trình tự route, chi phí, thời gian, phí vào cổng, mức độ phổ biến, điều kiện thời tiết, mức tắc nghẽn, mật độ đông đúc
- **Phục vụ**: **(A) Tìm kiếm** + **(B) CSP**

#### Dataset 8: SmartTourRoutePlanner Tourism Route Dataset
- **Nguồn**: https://www.kaggle.com/datasets/ziya07/smarttourrouteplanner-tourism-route-dataset
- **Cột chính**: Khoảng cách route, thời gian ước tính, phương tiện (xe hơi, bus, tàu, đi bộ), chi phí vận chuyển, phí vào cổng, giới hạn thời gian, giới hạn ngân sách
- **Phục vụ**: **(A) Tìm kiếm** + **(B) CSP**

#### Dataset 9: Worldwide Travel Cities Ratings and Climate
- **Nguồn**: https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate
- **Kích thước**: **560 thành phố**
- **Cột chính**: City, Country, Latitude, Longitude, Monthly temperatures, Budget Level, Thematic Ratings (Culture, Adventure, Nature, Beaches, Nightlife, Cuisine, Wellness, Urban) thang 0-5
- **Phục vụ**: **(E) Học máy** — phân loại thành phố theo phong cách du lịch
- **License**: MIT

#### Dataset 10: TripAdvisor Hotel Reviews (20K)
- **Nguồn**: https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews
- **Kích thước**: **20,000 reviews**
- **Cột chính**: Review text, Rating (1-5)
- **Phục vụ**: **(E) Học máy** — sentiment classification với Naive Bayes
- **License**: CC BY-NC 4.0

### 4.3. Bảng tổng hợp dataset

| # | Dataset | Kích thước | Thành phần AI | Việt Nam? |
|---|---------|-----------|--------------|-----------|
| 1 | Foursquare OS Places | 100M+ POIs | (A) Tìm kiếm | Lọc VN |
| 2 | Vietnam Weather Data | 181,960 records | (C) + (D) Suy luận + Bayes | **Có** |
| 3 | 515K Hotel Reviews | 515,000 reviews | (E) Học máy | Không |
| 4 | Travel Review Ratings | 5,456 users x 24 ratings | (E) Học máy | Không |
| 5 | Hotel Booking Demand | 119,390 bookings | (B) CSP | Không |
| 6 | Global Daily Climate | 1,250 cities, 223 MB | (D) Mạng Bayes | Một phần |
| 7 | Dynamic Tourism Route | Route data | (A) + (B) | Không |
| 8 | SmartTourRoutePlanner | Route + constraints | (A) + (B) | Không |
| 9 | Worldwide Travel Cities | 560 cities | (E) Học máy | Không |
| 10 | TripAdvisor Reviews | 20,000 reviews | (E) Học máy | Không |

**Tổng dung lượng dữ liệu ước tính**: > 300 MB sau khi tổng hợp và xử lý.

### 4.4. Datasets ưu tiên (Core datasets)

1. **Vietnam Weather Data** — dữ liệu Việt Nam gốc, phục vụ 2 thành phần (C + D), 181K records
2. **Foursquare OS Places** (lọc VN) — toạ độ POI để xây đồ thị, hàng trăm nghìn records
3. **515K Hotel Reviews** — dataset lớn nhất cho ML (515K rows)
4. **Travel Review Ratings** — cấu trúc hoàn hảo cho phân loại user (24 features)
5. **Hotel Booking Demand** — 119K bookings, dữ liệu chi phí (ADR)/thời gian lưu trú cho CSP constraints

---

## 5. Kế hoạch tiền xử lý dữ liệu (Chi tiết cho Thành viên 1)

### Bước 1: Thu thập dữ liệu
```
data/
├── raw/                          # Dữ liệu gốc tải về
│   ├── foursquare_places_vn.parquet
│   ├── vietnam_weather.csv
│   ├── hotel_reviews_515k.csv
│   ├── travel_review_ratings.csv
│   ├── hotel_bookings.csv
│   ├── global_climate.parquet
│   ├── tourism_route_dtrd.csv
│   └── worldwide_cities.csv
├── cleaned/                      # Dữ liệu sau khi làm sạch
├── processed/                    # Dữ liệu sau khi tiền xử lý
└── features/                     # File đặc trưng .npy / .h5
```

### Bước 2: Làm sạch dữ liệu
- Xử lý missing values: impute hoặc loại bỏ
- Loại bỏ duplicates
- Xử lý outliers (z-score hoặc IQR)
- Chuẩn hoá format ngày/giờ
- Chuẩn hoá tên tỉnh/thành phố Việt Nam

### Bước 3: Feature Engineering
- **Từ Foursquare**: tính ma trận khoảng cách (Haversine) giữa các POI → đồ thị cho A*
- **Từ Weather**: tạo nhãn `is_rainy`, `weather_severity`, `outdoor_suitable` → cho IF-THEN
- **Từ Weather**: tính P(rain | province, month) → cho Bayesian Network
- **Từ Reviews**: TF-IDF vectorization, word count, sentiment score → cho Decision Tree/Naive Bayes
- **Từ Travel Ratings**: normalize ratings, tạo nhãn `traveler_type` bằng clustering → cho classification
- **Từ Hotel Bookings**: tạo `total_nights`, `total_cost`, `budget_level`, `season` → cho CSP constraints

### Bước 4: Xuất dữ liệu
- Ma trận khoảng cách → `features/distance_matrix.npy`
- Xác suất thời tiết → `features/weather_probs.npy`
- TF-IDF vectors → `features/review_tfidf.npy`
- User feature vectors → `features/user_features.npy`

---

## 6. Cấu trúc thư mục dự án

```
Introduction-AI-Assignment/
├── notebooks/
│   └── main_notebook.ipynb        # Notebook chính chạy trên Colab
├── modules/
│   ├── data_pipeline.py           # (TV1) Download + xử lý dữ liệu
│   ├── search.py                  # (TV2) Thuật toán A*
│   ├── csp_solver.py              # (TV2) CSP backtracking
│   ├── knowledge_base.py          # (TV2) Hệ luật IF-THEN
│   ├── planner.py                 # (TV2) Tích hợp A+B+C
│   ├── bayesian_net.py            # (TV3) Mạng Bayes
│   └── ml_models.py               # (TV3) Decision Tree + Naive Bayes
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── processed/
│   └── features/                  # .npy / .h5 files
├── reports/
│   └── report.pdf                 # Báo cáo PDF (Overleaf)
├── README.md
├── Summary.md                     # File này
└── mlAssignments_MG.pdf           # Đề bài
```

---

## 7. Chi tiết phần đã thực hiện (Thành viên 1)

### 7.1 Module `data_pipeline.py`
File: `modules/data_pipeline.py`

**Chức năng đã cài đặt:**
- `download_all_datasets()` — Tự động tải 5 datasets từ Kaggle qua `opendatasets`
- `load_*()` — 5 hàm load từng dataset (tự tìm file CSV trong thư mục raw)
- `clean_vietnam_weather()` — Chuẩn hoá cột, parse date, xử lý missing (median theo tỉnh), tạo nhãn thời tiết (`is_rainy`, `outdoor_suitable`, `rain_level`, `is_hot`, `is_humid`)
- `clean_hotel_reviews()` — Gộp review text, loại placeholder, tạo nhãn sentiment (4 mức + binary), tính text features (word_count, char_count)
- `clean_travel_ratings()` — Chuyển numeric, fillna, tính `top_category`, `avg_rating`, `rating_std`
- `clean_hotel_bookings()` — Xử lý missing, tạo `total_nights`, `total_guests`, `total_cost` (ADR × nights), `budget_level` (4 mức), `arrival_date`, `season`
- `clean_world_cities()` — Chuẩn hoá cột rating/toạ độ
- `haversine()` — Tính khoảng cách giữa 2 toạ độ GPS (km)
- `build_distance_matrix()` — Ma trận khoảng cách 30x30 điểm du lịch VN
- `build_cost_matrix()` — Ma trận chi phí di chuyển (VND, ~3000 VND/km)
- `build_travel_time_matrix()` — Ma trận thời gian di chuyển (giờ, ~40km/h)
- `build_weather_probability_table()` — Bảng P(rain|province,month), P(outdoor|...), P(hot|...), P(humid|...)
- `build_places_dataframe()` — DataFrame 30 điểm du lịch VN (toạ độ, category, phí vào cổng, giờ mở/đóng)
- `run_full_pipeline()` — Chạy toàn bộ pipeline end-to-end

**Dữ liệu tĩnh đã xây dựng:**
- `VN_PROVINCE_COORDS` — Toạ độ 40 tỉnh thành Việt Nam
- `VN_TOURIST_PLACES` — 30 điểm du lịch nổi tiếng (toạ độ, loại hình, tỉnh, phí vào cổng)

### 7.2 Notebook `notebooks/main_notebook.ipynb`
**Cấu trúc notebook (chạy Runtime → Run All):**
1. **Setup** — Cài thư viện, clone repo, import modules
2. **Download** — Tải 5 datasets từ Kaggle (cần Kaggle API key)
3. **Load & Clean** — Load raw → clean → save cleaned CSV cho từng dataset
4. **EDA** — 8+ biểu đồ phân tích:
   - Lượng mưa/nhiệt độ theo tháng (VN)
   - Top tỉnh mưa nhiều, phân bố độ ẩm
   - Correlation thời tiết
   - Tỉ lệ outdoor_suitable theo mùa
   - Phân bố reviewer score, sentiment, word count
   - Top quốc tịch reviewer
   - Rating trung bình 24 loại hình du lịch + correlation
   - ADR distribution, budget level, số đêm lưu trú, top quốc gia booking
5. **Feature Engineering:**
   - Ma trận khoảng cách 30x30 → `distance_matrix.npy`
   - Ma trận chi phí → `cost_matrix.npy`
   - Ma trận thời gian → `travel_time_matrix.npy`
   - Bảng xác suất thời tiết → `weather_probabilities.csv`
   - TF-IDF 50K reviews (5000 features) → `review_tfidf.npz`
   - User feature vectors (standardized) → `user_features.npy`
   - K-Means clustering 5 nhóm khách → `user_cluster_labels.npy`
   - PCA 2D visualization cho clusters
6. **Tổng kết** — Liệt kê files đã tạo + mapping sang thành phần AI

### 7.3 File outputs

| File | Thành phần AI | Mô tả |
|---|---|---|
| `data/cleaned/vietnam_weather.csv` | (C) + (D) | 181K rows, thời tiết VN đã xử lý |
| `data/cleaned/hotel_reviews.csv` | (E) | 515K reviews đã xử lý + sentiment labels |
| `data/cleaned/travel_ratings.csv` | (E) | 5,456 users x 24 ratings + top_category |
| `data/cleaned/hotel_bookings.csv` | (B) | 118K bookings + total_cost + budget_level + season |
| `data/cleaned/world_cities.csv` | (E) | 560 cities + thematic ratings |
| `data/features/distance_matrix.npy` | (A) | 30x30 khoảng cách (km) |
| `data/features/cost_matrix.npy` | (A) + (B) | 30x30 chi phí (VND) |
| `data/features/travel_time_matrix.npy` | (A) + (B) | 30x30 thời gian (giờ) |
| `data/features/vn_tourist_places.csv` | (A) + (B) | 30 điểm du lịch VN |
| `data/features/weather_probabilities.csv` | (D) | P(rain\|province,month)... |
| `data/features/review_tfidf.npz` | (E) | Sparse TF-IDF 50K x 5000 |
| `data/features/review_labels.npy` | (E) | Nhãn sentiment binary |
| `data/features/user_features.npy` | (E) | 5,456 x 24 standardized |
| `data/features/user_cluster_labels.npy` | (E) | 5 nhóm khách du lịch |

---

## 8. Tiến độ hiện tại

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Chọn đề tài | Hoàn thành | Kết hợp ý tưởng #3 + #6 |
| Đối chiếu yêu cầu PDF | Hoàn thành | Đáp ứng 5/5 thành phần |
| Phân công nhóm | Hoàn thành | 3 người, mỗi người ~33% |
| Research datasets | Hoàn thành | 10 datasets, tổng > 300 MB |
| Viết module data_pipeline.py | Hoàn thành | Download + Clean + Feature Engineering |
| Viết notebook (EDA + preprocessing) | Hoàn thành | 6 sections, 8+ biểu đồ, 14 feature files |
| Cài đặt thuật toán (TV2) | Chưa bắt đầu | A* + CSP + IF-THEN |
| Mạng Bayes + ML (TV3) | Chưa bắt đầu | Bayesian Network + Decision Tree + Naive Bayes |
| Tích hợp hệ thống | Chưa bắt đầu | — |
| Báo cáo | Chưa bắt đầu | — |
