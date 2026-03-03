# AI Travel Planner & Recommender System

## Thông tin môn học

| | |
|---|---|
| **Môn học** | Giới thiệu về Trí tuệ Nhân tạo (Introduction to AI) |
| **Mã môn học** | CO3001 |
| **Học kỳ** | II — Năm học 2025–2026 |
| **Trường** | Đại học Bách Khoa, ĐHQG-HCM |
| **GVHD** | TS. Trương Vĩnh Lân |

## Thành viên nhóm

| STT | Họ tên | MSSV | Email |
|-----|--------|------|-------|
| 1 | Nguyễn Minh Hiếu | 2153343 | hieu.nguyen080203@hcmut.edu.vn |
| 2 | Trần Ngọc Khánh Huy | 2153343 | hieu.nguyen080203@hcmut.edu.vn |
| 3 | Phan Thảo Vy | 2153343 | hieu.nguyen080203@hcmut.edu.vn |
| 4 | Hoàng Thiện Nhân | 2153343 | hieu.nguyen080203@hcmut.edu.vn |

## Mục tiêu bài tập lớn

Xây dựng hệ thống AI lập kế hoạch du lịch thông minh, cá nhân hoá lịch trình dựa trên ngân sách, thời gian, sở thích người dùng và điều kiện thời tiết. Hệ thống tích hợp **5 thành phần AI** theo yêu cầu đề bài:

| Thành phần | Mô tả |
|---|---|
| **(A) Biểu diễn & Tìm kiếm** | A* với heuristic kết hợp chi phí + thời gian để tìm lộ trình tối ưu |
| **(B) Heuristic / CSP** | Backtracking + Forward Checking cho ràng buộc ngân sách, giờ mở cửa, thời gian di chuyển |
| **(C) Suy luận tri thức** | Luật IF–THEN: lọc hoạt động theo thời tiết, ngân sách, nhóm người dùng |
| **(D) Mạng Bayes** | Mô hình P(mưa \| tỉnh, tháng), P(user thích địa điểm \| loại hình, rating) |
| **(E) Học máy** | Decision Tree + Naive Bayes phân loại nhóm du khách và dự đoán rating |

## Hướng dẫn chạy notebook

### Trên Google Colab (khuyến nghị)

1. Mở notebook qua link Colab bên dưới (hoặc upload file `notebooks/main_notebook.ipynb`).
2. Chọn **Runtime → Run all**.
3. Khi được hỏi Kaggle API key, nhập `username` và `key` từ file `kaggle.json` (tải tại [kaggle.com/settings](https://www.kaggle.com/settings) → API → Generate New Token).
4. Notebook sẽ tự động clone repo, tải dữ liệu, xử lý và xuất kết quả.

### Trên máy local

```bash
# Clone repository
git clone https://github.com/HieNguyen08/Introduction-AI-Assignment.git
cd Introduction-AI-Assignment

# Cài thư viện
pip install pandas numpy matplotlib seaborn scikit-learn opendatasets

# Đảm bảo đã cấu hình Kaggle CLI (~/.kaggle/kaggle.json)
# Chạy notebook
jupyter notebook notebooks/main_notebook.ipynb
```

### Yêu cầu thư viện

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `opendatasets`

## Cấu trúc thư mục

```
Introduction-AI-Assignment/
├── notebooks/
│   └── main_notebook.ipynb          # Notebook chính (Google Colab)
├── modules/
│   ├── data_pipeline.py             # Download + xử lý dữ liệu
│   ├── search.py                    # Thuật toán A*
│   ├── csp_solver.py                # CSP backtracking
│   ├── planner.py                   # Tích hợp Search + CSP
│   ├── knowledge_base.py            # Hệ luật IF-THEN
│   ├── bayesian_net.py              # Mạng Bayes
│   └── ml_models.py                 # Decision Tree + Naive Bayes
├── data/
│   ├── raw/                         # Dữ liệu gốc (tải tự động từ Kaggle)
│   ├── cleaned/                     # Dữ liệu đã làm sạch
│   ├── processed/                   # Biểu đồ EDA
│   └── features/                    # File đặc trưng .npy / .csv
├── reports/
│   └── report.pdf                   # Báo cáo PDF
├── README.md
├── Summary.md                       # Tóm tắt dự án & phân công
└── mlAssignments_MG.pdf             # Đề bài
```

## Datasets

| # | Dataset | Kích thước | Thành phần AI |
|---|---------|-----------|--------------|
| 1 | [Vietnam Weather Data](https://www.kaggle.com/datasets/vanviethieuanh/vietnam-weather-data) | 181,960 rows | (C) + (D) |
| 2 | [515K Hotel Reviews Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) | 515,738 rows | (E) |
| 3 | [Travel Review Ratings](https://www.kaggle.com/datasets/ishbhms/travel-review-ratings) | 5,456 users | (E) |
| 4 | [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) | 119,390 rows | (B) |
| 5 | [Worldwide Travel Cities](https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate) | 560 cities | (A) + (C) |

## Links

| | |
|---|---|
| **Google Colab** | [Mở notebook trên Colab](https://colab.research.google.com/github/HieNguyen08/Introduction-AI-Assignment/blob/main/notebooks/main_notebook.ipynb) |
| **Báo cáo Overleaf** | [Overleaf](https://www.overleaf.com/6935761933tywyhjstsskf#35f45c) |
| **GitHub** | [HieNguyen08/Introduction-AI-Assignment](https://github.com/HieNguyen08/Introduction-AI-Assignment) |
