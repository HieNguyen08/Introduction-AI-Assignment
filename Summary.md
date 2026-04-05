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

### Thành viên 2 — Thuật toán Tìm kiếm + CSP

| Công việc | Mô tả | Kết quả đầu ra |
|---|---|---|
| (A) Thuật toán A* | Cài đặt A* trên đồ thị địa điểm du lịch; thiết kế heuristic function (khoảng cách + chi phí + thời gian) | `modules/search.py` |
| (B) CSP Solver | Cài đặt backtracking + forward checking cho ràng buộc lịch trình (ngân sách, giờ mở/đóng, thời gian di chuyển, số ngày) | `modules/csp_solver.py` |
| Tích hợp A+B | Kết nối Search + CSP thành pipeline lập lịch trình: input constraints → CSP lọc → A* tìm route tối ưu | `modules/planner.py` |

### Thành viên 3 — Suy luận tri thức + Mạng Bayes

| Công việc | Mô tả | Kết quả đầu ra |
|---|---|---|
| (C) Hệ luật IF-THEN | Xây dựng knowledge base luật suy luận (thời tiết → loại bỏ outdoor, ngân sách → ưu tiên miễn phí, gia đình → loại nightlife) | `modules/knowledge_base.py` |
| (D) Bayesian Network | Xây dựng mạng Bayes cho dự đoán thời tiết P(rain\|province,month) + sở thích user P(like\|category,rating) | `modules/bayesian_net.py` |
| Tích hợp C+D | Kết nối IF-THEN rules với Bayesian Network: Bayes dự đoán → rules lọc → đề xuất hoạt động phù hợp | Tích hợp vào `modules/planner.py` |

### Thành viên 4 — Học máy + Tích hợp hệ thống

| Công việc | Mô tả | Kết quả đầu ra |
|---|---|---|
| (E) Decision Tree + Naive Bayes | Huấn luyện model phân loại user thành nhóm (culture lover, adventure seeker, foodie...) + dự đoán rating/sentiment | `modules/ml_models.py` |
| Tích hợp hệ thống | Kết nối tất cả module (A→E) thành hệ thống hoàn chỉnh trên Colab: input user → ML phân loại → Bayes + Rules lọc → CSP + A* lập lịch → output | Notebook chính |
| Testing & Demo | Kiểm thử end-to-end với nhiều kịch bản (budget thấp, gia đình, mùa mưa...), chuẩn bị demo | Notebook chính + test cases |

### Báo cáo — Viết chung (cả 4 thành viên)

| Công việc | Mô tả | Kết quả đầu ra |
|---|---|---|
| Viết báo cáo | Mỗi thành viên viết phần mình phụ trách, báo cáo PDF 10-20 trang theo cấu trúc yêu cầu | `reports/report.pdf` (Overleaf) |

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
│   ├── planner.py                 # (TV2) Tích hợp A+B → lập lịch trình
│   ├── knowledge_base.py          # (TV3) Hệ luật IF-THEN
│   ├── bayesian_net.py            # (TV3) Mạng Bayes
│   └── ml_models.py               # (TV4) Decision Tree + Naive Bayes
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

## 8. Chi tiết phần đã thực hiện (Thành viên 3)

### 8.1 Module `knowledge_base.py` — Thành phần (C)
File: `modules/knowledge_base.py`

**Mô tả:** Hệ luật IF-THEN (Rule-Based Expert System) sử dụng Forward Chaining inference engine để lọc và đánh giá địa điểm du lịch dựa trên ngữ cảnh (thời tiết, ngân sách, nhóm du khách, sở thích).

**Classes chính:**

| Class / Function | Mô tả |
|---|---|
| `Rule` | Đại diện 1 luật IF-THEN (condition_fn, action_fn, priority) |
| `KnowledgeBase` | Chứa tập 18 luật, sắp xếp theo priority, inference engine |
| `create_context()` | Helper tạo context dict chuẩn cho inference |
| `filter_places_by_weather()` | Lọc chỉ dựa trên thời tiết |
| `filter_places_full()` | Lọc đầy đủ (thời tiết + ngân sách + nhóm + sở thích) |

**18 luật IF-THEN đã cài đặt:**

| ID | Tên luật | Điều kiện IF | Hành động THEN | Priority |
|---|---|---|---|---|
| R1 | Mưa to → Loại outdoor | rain_mm > 10mm | Loại bỏ nature, beach, adventure | 1 |
| R2 | Mưa vừa → Cảnh báo | 5mm < rain ≤ 10mm | Giảm weather_score outdoor (×0.5) | 2 |
| R3 | Nóng > 35°C | temp_max > 35 | Ưu tiên indoor/beach, giảm adventure | 3 |
| R4 | Lạnh < 15°C | temp_min < 15 | Gợi ý áo ấm, tăng culture/entertainment | 4 |
| R5 | Ẩm > 80% | humidity > 80 | Giảm weather_score outdoor (×0.8) | 5 |
| R6 | Gió mạnh | wind_speed > 40km/h | Loại bỏ adventure | 1 |
| R7 | Ngân sách thấp | budget < 500K VND | Loại điểm phí > 30% budget | 2 |
| R8 | Ngân sách TB | 500K ≤ budget < 2M | Loại điểm phí > 500K | 3 |
| R9 | Gia đình | group_type == "family" | Loại adventure nguy hiểm (phí > 1M) | 3 |
| R10 | Cặp đôi | group_type == "couple" | Tăng preference_score beach/nature (×1.3) | 4 |
| R11 | Nhóm bạn | group_type == "friends" | Tăng adventure/entertainment/beach (×1.3) | 4 |
| R12 | Mùa mưa | season ∈ {summer, autumn} | Tăng indoor (×1.2), giảm outdoor (×0.8) | 5 |
| R13 | Thời tiết đẹp | outdoor_suitable == True | Tăng outdoor (×1.3) | 6 |
| R14 | Thích văn hóa | "culture" ∈ preferences | Tăng culture (×1.4) | 6 |
| R15 | Thích phiêu lưu | "adventure" ∈ preferences | Tăng adventure/nature (×1.4) | 6 |
| R16 | Thích biển | "beach" ∈ preferences | Tăng beach (×1.5) | 6 |
| R17 | Trip ngắn | num_days ≤ 2 | Ưu tiên cùng tỉnh (×1.5) | 4 |
| R18 | Giờ mở cửa | current_hour != None | Loại điểm đã đóng/chưa mở | 2 |

**Ngưỡng thời tiết (khớp với data_pipeline):**
- `is_rainy` → rain_mm > 1.0 (tiêu chuẩn WMO)
- `outdoor_suitable` → rain_mm ≤ 5 AND temp_max ≤ 38 AND humidity ≤ 90
- `is_hot` → temp_max > 35 (ngưỡng nắng nóng KTTV Việt Nam)
- `is_humid` → humidity > 80

**Context dict (đầu vào cho inference):**
```python
context = {
    "rain_mm": float,           # Lượng mưa (mm)
    "temp_max": float,          # Nhiệt độ cao nhất (°C)
    "temp_min": float,          # Nhiệt độ thấp nhất (°C)
    "humidity": float,          # Độ ẩm (%)
    "wind_speed": float,        # Tốc độ gió (km/h)
    "budget_vnd": float,        # Ngân sách (VND)
    "group_type": str,          # "solo" | "couple" | "family" | "friends"
    "season": str,              # "spring" | "summer" | "autumn" | "winter"
    "outdoor_suitable": bool,   # Thời tiết phù hợp outdoor
    "user_preferences": list,   # ["culture", "beach", "adventure", ...]
    "num_days": int,            # Số ngày du lịch
    "current_hour": int|None,   # Giờ hiện tại (0-23)
    "current_province": str,    # Tỉnh hiện tại
}
```

### 8.2 Module `bayesian_net.py` — Thành phần (D)
File: `modules/bayesian_net.py`

**Mô tả:** Mạng Bayes (Bayesian Network) mô hình hoá xác suất thời tiết và sở thích du khách, sử dụng CPT (Conditional Probability Table) được xây dựng từ 181K records dữ liệu thời tiết Việt Nam.

**Cấu trúc mạng Bayes (5 nodes):**
```
Province, Month → Rain          (CPT: 480 entries — 40 tỉnh × 12 tháng)
Province, Month → Outdoor_OK    (CPT: 480 entries)
Province, Month → Hot           (CPT: 480 entries)
Province, Month → Humid         (CPT: 480 entries)
Category, Group_Type, Is_Rain → User_Like  (CPT: 40 entries — 5 cat × 4 group × 2 rain)
```

**Classes chính:**

| Class / Function | Mô tả |
|---|---|
| `BayesNode` | 1 node trong mạng (tên, parents, CPT) |
| `BayesianNetwork` | Toàn bộ mạng, build từ data, query & scoring |
| `integrate_bayes_kb()` | Hàm tích hợp C+D (Bayes → Rules → Scoring) |

**Chức năng đã cài đặt:**
- `build_from_data(weather_probs_df)` — Xây dựng 5 nodes + CPT từ bảng xác suất thời tiết
- `query_rain(province, month)` — P(rain \| province, month)
- `query_outdoor(province, month)` — P(outdoor_suitable \| province, month)
- `query_hot(province, month)` — P(hot \| province, month)
- `query_humid(province, month)` — P(humid \| province, month)
- `query_weather_full(province, month)` — Truy vấn tất cả biến thời tiết
- `query_user_preference(category, group_type, is_rain)` — P(user_like \| ...)
- `score_places(places_df, month, group_type)` — Tính bayesian_score cho tất cả điểm
- `predict_best_month(province, category, group_type)` — Dự đoán tháng tốt nhất du lịch
- `print_network()` — In cấu trúc mạng

**Công thức Bayesian Score:**
- Outdoor (nature, beach, adventure): `Score = P(outdoor_ok | province, month) × P(user_like)`
- Indoor (culture, entertainment): `Score = P(user_like) × 0.9 + 0.1`
- Trong đó: `P(user_like) = P(like|rain)×P(rain) + P(like|¬rain)×P(¬rain)` (Total probability)

### 8.3 Tích hợp C+D vào `planner.py`
File: `modules/planner.py`

**Mô tả:** Module tích hợp trung tâm, kết nối (C) Knowledge Base + (D) Bayesian Network thành pipeline lọc và xếp hạng địa điểm. Cấu trúc sẵn sàng cho TV2 thêm (A) A* Search + (B) CSP.

**Pipeline tích hợp C+D:**
```
Input (province, month, group_type, budget, preferences)
  → (D) Bayes dự đoán P(rain|province,month), P(outdoor|...)
  → Chuyển xác suất thành weather context (expected values)
  → (C) IF-THEN Forward Chaining lọc địa điểm
  → (D) Bayesian scoring xếp hạng điểm còn lại
  → Output (ranked places + metadata + explanations)
```

**Chức năng đã cài đặt:**
- `filter_and_rank_places()` — Hàm chính: lọc + xếp hạng (gọi integrate_bayes_kb)
- `get_weather_recommendation()` — Khuyến nghị thời tiết cho tỉnh/tháng
- `find_best_travel_month()` — Tìm tháng tốt nhất du lịch
- `plan_trip()` — Lập kế hoạch du lịch đầy đủ (greedy day-assignment, TODO: TV2 thay bằng A*+CSP)

### 8.4 Notebook — Section 5 + 6
File: `notebooks/main_notebook.ipynb` (21 cells mới)

**Section 5: Component (C) — Knowledge Base IF-THEN Rules**
- Import + hiển thị bảng 18 luật
- Kịch bản 1: Mưa to (15mm) + Gia đình + Budget 400K → loại outdoor, loại phí cao
- Kịch bản 2: Thời tiết đẹp + Cặp đôi + Budget 5M → ưu tiên beach/nature
- Kịch bản 3: Gió mạnh (45km/h) + Nhóm bạn + Adventure → loại adventure nguy hiểm

**Section 6: Component (D) — Bayesian Network**
- Xây dựng mạng Bayes từ `weather_probabilities.csv`
- Truy vấn P(rain) cho 6 tỉnh × 5 tháng
- Dự đoán tháng tốt nhất du lịch biển Nha Trang + biểu đồ
- Bayesian scoring 30 điểm du lịch (tháng 3, cặp đôi) + biểu đồ ranking
- Tích hợp C+D — 3 kịch bản end-to-end:
  - Đà Nẵng tháng 8 (gia đình, budget 3M, beach/culture)
  - Lâm Đồng tháng 12 (solo, budget 5M, adventure/nature)
  - Hà Nội tháng 3 (cặp đôi, budget 2M, culture)

### 8.5 Kết nối với các thành viên khác

| Từ module | Sử dụng | Cho module |
|---|---|---|
| `data_pipeline.py` (TV1) | `build_places_dataframe()`, `build_weather_probability_table()`, `VN_TOURIST_PLACES` | `knowledge_base.py`, `bayesian_net.py` |
| `knowledge_base.py` (TV3) | `filter_places_full()`, `KnowledgeBase` | `planner.py`, notebook |
| `bayesian_net.py` (TV3) | `integrate_bayes_kb()`, `BayesianNetwork` | `planner.py`, notebook |
| `planner.py` (TV3→TV2) | `filter_and_rank_places()` | TV2 sẽ gọi trước khi A*/CSP |
| `planner.py` (TV3→TV4) | `plan_trip()` | TV4 sẽ tích hợp ML user classification |

---

## 9. Tiến độ hiện tại

| Hạng mục | Phụ trách | Trạng thái | Ghi chú |
|---|---|---|---|
| Chọn đề tài | Cả nhóm | ✅ Hoàn thành | Kết hợp ý tưởng #3 + #6 |
| Đối chiếu yêu cầu PDF | Cả nhóm | ✅ Hoàn thành | Đáp ứng 5/5 thành phần |
| Phân công nhóm | Cả nhóm | ✅ Hoàn thành | 4 người |
| Research datasets | TV1 | ✅ Hoàn thành | 10 datasets, tổng > 300 MB |
| Viết module `data_pipeline.py` | TV1 | ✅ Hoàn thành | Download + Clean + Feature Engineering (17 file outputs) |
| Viết notebook (EDA + preprocessing + Colab compliance) | TV1 | ✅ Hoàn thành | `Runtime → Run All` OK, Kaggle Secrets, scipy/kaggle installed |
| **(A) A* Search + (B) CSP Solver** | **TV2** | ❌ **Chưa bắt đầu** | `search.py` + `csp_solver.py` → xem **Mục 10** để hướng dẫn chi tiết |
| (C) IF-THEN Rules + (D) Bayesian Network | TV3 | ✅ Hoàn thành | `knowledge_base.py` (18 luật) + `bayesian_net.py` (5 nodes) + tích hợp C+D vào `planner.py` |
| (E) Decision Tree + Naive Bayes | TV4 | ✅ Hoàn thành | `ml_models.py` (merged via system-integration) |
| Tích hợp hệ thống + Testing | TV4 | 🔄 Chờ TV2 | Cần A*/CSP để kết nối đầy đủ A→E |
| Báo cáo | Cả nhóm | ❌ Chưa bắt đầu | Mỗi người viết phần mình (Overleaf) |

---

## 10. Hướng dẫn code cho TV2 — Thành phần (A) A* + (B) CSP

> **Dành cho Huy (TV2 — Trần Ngọc Khánh Huy)**. Đây là hướng dẫn chi tiết để cài đặt `search.py` và `csp_solver.py`, tích hợp vào `planner.py`. Data đã sẵn sàng, TV3 đã xây xong khung `planner.py` với stubs chờ TV2.

### 10.1 Dữ liệu đã sẵn sàng (không cần tạo thêm)

| File | Shape / Size | Nội dung | Dùng cho |
|---|---|---|---|
| `data/features/vn_tourist_places.csv` | 267 hàng × 10 cột | Địa điểm VN: tên, toạ độ, category, tỉnh, phí vào cổng, giờ mở/đóng cửa, visit_duration | A* + CSP |
| `data/features/distance_matrix.npy` | (50, 50) — km | Ma trận khoảng cách giữa 50 địa điểm đầu tiên (0.2–1,639 km) | A* cost / heuristic |
| `data/features/cost_matrix.npy` | (50, 50) — VND | Chi phí di chuyển giữa 50 địa điểm (~3,000 VND/km) | A* cost / CSP |
| `data/features/travel_time_matrix.npy` | (50, 50) — giờ | Thời gian di chuyển giữa 50 địa điểm (0–41 giờ) | A* cost / CSP |
| `data/features/hotel_price_stats.csv` | 23 hàng × 7 cột | Giá khách sạn trung bình theo tỉnh và hạng (budget/mid/premium) | CSP ràng buộc ngân sách |
| `data/cleaned/hotel_bookings.csv` | 118,565 hàng × 39 cột | Booking data: adr (giá/đêm), budget_level, total_nights, season | CSP phân tích ngân sách |

> ⚠️ **Quan trọng**: Ma trận 50×50 tương ứng với **50 địa điểm ĐẦU TIÊN** trong `vn_tourist_places.csv` (index 0–49). Khi dùng A*, chỉ index các địa điểm trong vùng đó.

**Cột của `vn_tourist_places.csv`:**
```
place_name, latitude, longitude, category, province,
entry_fee_vnd, visit_duration_hours, opening_hour, closing_hour, description
```

**Cột của `hotel_price_stats.csv`:**
```
province, price_tier (budget/mid_range/premium),
avg_price, median_price, min_price, max_price, hotel_count
```

### 10.2 Kiến trúc tổng quan — Luồng A* + CSP trong hệ thống

```
Input: province, month, group_type, budget_vnd, num_days, user_preferences
        ↓
[planner.filter_and_rank_places()]  ← đã có (TV3)
        ↓ ranked_places (filtered by weather + rules)
[CSP Solver]  ← TV2 viết
        ↓ feasible_schedule: {day_1: [p1, p2, p3], day_2: [...], ...}
[A* Search]   ← TV2 viết (chạy trên từng ngày)
        ↓ optimal_route per day (thứ tự tham quan tối ưu)
Output: daily_plan với route đã tối ưu
```

### 10.3 Module `modules/search.py` — Thuật toán A*

#### Định nghĩa bài toán tìm kiếm

| Thành phần | Định nghĩa |
|---|---|
| **State** | `(current_idx, visited_frozenset, time_used_h, budget_used_vnd)` |
| **Initial State** | `(start_idx, frozenset({start_idx}), 0.0, 0.0)` |
| **Action** | Di chuyển đến địa điểm chưa thăm trong `candidates` |
| **Goal Test** | Đã thăm tất cả địa điểm trong `candidates` |
| **Cost g(n)** | `travel_time + visit_duration` (hoặc kết hợp thời gian + chi phí) |
| **Heuristic h(n)** | MST trên các địa điểm chưa thăm (admissible) hoặc nearest-neighbor lower bound |

#### Template code `modules/search.py`

```python
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
```

### 10.4 Module `modules/csp_solver.py` — CSP với Backtracking + Forward Checking

#### Định nghĩa bài toán CSP

| Thành phần | Định nghĩa |
|---|---|
| **Variables** | `X_d` = danh sách địa điểm được chọn cho ngày `d` (d = 1..num_days) |
| **Domains** | Tập địa điểm đã được filter+rank bởi `filter_and_rank_places()` |
| **Constraints** | (1) Tổng chi phí ≤ budget; (2) Tổng giờ/ngày ≤ time_limit; (3) Mỗi địa điểm thăm ≤ 1 lần; (4) Giờ mở/đóng cửa; (5) Ngân sách khách sạn theo tỉnh |
| **Objective** | Tối đa hoá tổng `bayesian_score` của các địa điểm được chọn |

#### Template code `modules/csp_solver.py`

```python
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

    if best_solution["score"] < 0:
        # Fallback: greedy nếu CSP không tìm được
        return _greedy_schedule(domain_df, num_days, max_places_per_day)

    return {
        "schedule": best_solution["schedule"],
        "score": best_solution["score"],
        "total_cost": best_solution["total_cost"],
        "feasible": True,
    }


def _greedy_schedule(domain_df: pd.DataFrame, num_days: int,
                     max_per_day: int) -> Dict[str, Any]:
    """Fallback: chia đều địa điểm theo ngày."""
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
```

### 10.5 Tích hợp vào `planner.py` — Thay thế stub greedy

Sau khi viết xong `search.py` và `csp_solver.py`, TV2 cần **thay thế đoạn greedy** trong `plan_trip()` tại `modules/planner.py` ([dòng 288](modules/planner.py#L288)):

**Tìm đoạn này trong `plan_trip()` (dòng ~288):**
```python
# Step 4: Phân chia địa điểm theo ngày (simple greedy — TV2 sẽ thay bằng A* + CSP)
selected = ranked_places.head(max_places_per_day * num_days)
daily_plan = {}
for day in range(1, num_days + 1):
    start_idx = (day - 1) * max_places_per_day
    end_idx = min(day * max_places_per_day, len(selected))
    day_places = selected.iloc[start_idx:end_idx]
    daily_plan[f"Ngày {day}"] = day_places
```

**Thay bằng:**
```python
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
```

### 10.6 Cập nhật phần import ở đầu `planner.py`

Thêm vào đầu file (sau các import hiện tại):
```python
# TV2 sẽ uncomment khi search.py và csp_solver.py đã sẵn sàng:
# from modules.search import find_optimal_daily_route, load_matrices, load_places
# from modules.csp_solver import solve_schedule
```

### 10.7 Notebook — Section 7 + 8 cần thêm (TV2 tự viết)

Sau khi hoàn thành `search.py` và `csp_solver.py`, TV2 cần thêm 2 sections vào `notebooks/main_notebook.ipynb`:

**Section 7: Component (A) — A* Search**
```python
# Cell 7.1: Import và load data
from modules.search import astar_route, load_matrices, load_places
dist_mat, cost_mat, time_mat = load_matrices()
places_df = load_places()

# Cell 7.2: Demo A* — Tìm lộ trình 1 ngày
# Cho 5 địa điểm Hà Nội (index 0-4), tìm thứ tự tối ưu
result = astar_route(
    start_idx=0,
    candidates=[0, 1, 2, 3, 4],
    places_df=places_df,
    time_matrix=time_mat,
    cost_matrix=cost_mat,
    time_limit_hours=10.0,
)
print("Lộ trình tối ưu:", result["path_names"])
print(f"Tổng thời gian: {result['total_time']:.1f} giờ")
print(f"Chi phí di chuyển: {result['total_travel_cost']:,.0f} VND")

# Cell 7.3: So sánh A* vs Greedy (biểu đồ bar chart: tổng thời gian)
```

**Section 8: Component (B) — CSP Solver**
```python
# Cell 8.1: Import và setup
from modules.csp_solver import solve_schedule
from modules.planner import filter_and_rank_places

# Cell 8.2: Lấy ranked places (từ C+D)
ranked, meta = filter_and_rank_places(
    province="Da Nang", month=3, group_type="family",
    budget_vnd=3_000_000, user_preferences=["beach", "culture"], num_days=3
)

# Cell 8.3: Chạy CSP solver
schedule = solve_schedule(
    ranked_places_df=ranked,
    province="Da Nang",
    num_days=3,
    budget_vnd=3_000_000,
    budget_level="mid_range",
)

for day, places in schedule["schedule"].items():
    print(f"\nNgày {day}:")
    for p in places:
        print(f"  - {p['place_name']} ({p['category']}) — {p.get('entry_fee_vnd',0):,.0f} VND")
print(f"\nTổng chi phí ước tính: {schedule['total_cost']:,.0f} VND")
```

### 10.8 Checklist hoàn thành cho TV2

- [ ] Tạo branch `Huy` từ `main`: `git checkout -b Huy`
- [ ] Tạo `modules/search.py` (copy template từ Mục 10.3 và điền)
- [ ] Tạo `modules/csp_solver.py` (copy template từ Mục 10.4 và điền)
- [ ] Thay đoạn greedy trong `planner.py` bằng code từ Mục 10.5
- [ ] Thêm Section 7 (A*) và Section 8 (CSP) vào notebook
- [ ] Chạy `Runtime → Run All` kiểm tra không lỗi
- [ ] Push branch: `git push origin Huy`
- [ ] Tạo Pull Request vào `main`

> **Lưu ý**: Notebook phải chạy **Runtime → Run All** không lỗi (yêu cầu PDF Section 6). Dùng `try/except` bọc các cell demo A*/CSP nếu cần.
