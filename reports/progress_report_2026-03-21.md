# [CO3061] Báo cáo tiến độ định kỳ môn Nhập môn trí tuệ Nhân tạo

**Kính gửi thầy:** Trương Vĩnh Lân – Giảng viên môn Nhập môn trí tuệ Nhân tạo (CO3061)
**Tên em là:** Nguyễn Minh Hiếu – Nhóm trưởng Group 3 lớp A01
**Mã số sinh viên:** 2153343

Thay mặt nhóm, em xin phép gửi tới thầy báo cáo chi tiết về tiến độ thực hiện bài tập lớn định kỳ của nhóm trong giai đoạn **từ ngày 07/03/2026 đến ngày 21/03/2026** như sau:

---

## 1. Thành viên nhóm và đề tài

- **Tập hợp thành viên:** Nhóm đã hoàn tất việc tập hợp đầy đủ thành viên từ ngày 26/02/2026.

- **Đề tài chọn lựa:** Xây dựng hệ thống AI lập kế hoạch du lịch thông minh, cá nhân hóa lịch trình dựa trên ngân sách, thời gian, sở thích và điều kiện thời tiết.

- **Phân công công việc:** Công việc cụ thể đã được phân chia cho từng cá nhân vào ngày 03/03/2026 dựa trên các module chính của dự án và được ghi lại vào tệp Summary.md trên dự án của nhóm.

---

## 2. Tiến độ công việc (07/03 – 21/03)

Trong hai tuần vừa qua, nhóm đã có những tiến triển đáng kể so với báo cáo trước:

- **Nhánh Nhân (Hoàng Thiện Nhân) đã hoàn thành module:** Thành viên Nhân đã hoàn tất module được giao trên nhánh riêng và đã được merge vào nhánh chính (main) thông qua Pull Request #1. Cụ thể, các công việc đã hoàn thành bao gồm:
  - **(C) Knowledge Base IF-THEN:** Xây dựng hệ thống luật IF-THEN gồm 18 rules phục vụ lọc và gợi ý địa điểm dựa trên thời tiết, ngân sách, nhóm du lịch, sở thích và thời gian (file `modules/knowledge_base.py`, 841 dòng code).
  - **(D) Bayesian Network:** Xây dựng mạng Bayes gồm 5 node để suy luận xác suất thời tiết và sở thích người dùng, tính toán P(mưa | tỉnh, tháng), P(thích | loại hình, nhóm, thời tiết) (file `modules/bayesian_net.py`, 749 dòng code).
  - **Tích hợp C + D:** Module `planner.py` đã tích hợp thành công Knowledge Base và Bayesian Network để lọc, xếp hạng địa điểm du lịch dựa trên ngữ cảnh người dùng.

- **Cải thiện Data Pipeline (Nguyễn Minh Hiếu):** Cải tiến đáng kể module xử lý dữ liệu (`modules/data_pipeline.py`):
  - Thay thế toàn bộ `print()` bằng module `logging` chuẩn Python.
  - Trích xuất các hằng số cấu hình (chi phí di chuyển, tốc độ, ngưỡng thời tiết) thành phần config riêng biệt.
  - Thêm hàm validation đầu vào (`_validate_dataframe`) cho tất cả hàm tiền xử lý.
  - Viết 22 smoke tests cho toàn bộ pipeline (`tests/test_data_pipeline.py`).
  - Thêm file `requirements.txt` liệt kê đầy đủ dependencies.

- **Giữ nguyên phạm vi Việt Nam cho tập dữ liệu:** Nhóm xác nhận giữ nguyên scope dữ liệu tập trung vào Việt Nam. Hệ thống hiện đã mô hình hóa 35 địa điểm du lịch Việt Nam với đầy đủ tọa độ, ma trận khoảng cách, chi phí và thời gian di chuyển.

- **Hai thành viên còn lại vẫn đang tiến hành code:** Trần Ngọc Khánh Huy (TV2) và Phan Thảo Vy (TV3) đang tiếp tục phát triển các module được phân công.

---

## 3. Định hướng kỹ thuật và các bước tiếp theo

Hiện tại, các thành viên đang tập trung nghiên cứu lý thuyết và triển khai các module chính còn lại, cụ thể:

- **Tối ưu hóa lịch trình:** Nghiên cứu áp dụng thuật toán tìm kiếm **A\*** và bài toán **Thỏa mãn ràng buộc (CSP)** với Backtracking + Forward Checking để đề xuất lộ trình du lịch tối ưu. Các ma trận khoảng cách, chi phí và thời gian di chuyển (35×35) đã sẵn sàng để sử dụng.

- **Dự báo và phân loại:** Tìm hiểu về **Mạng Bayes (Bayesian Networks)** để dự đoán rủi ro thời tiết và sử dụng **Cây quyết định (Decision Trees/ML)** để phân loại sở thích khách hàng. Module Bayesian Network đã hoàn thành và tích hợp, module ML đang được phát triển.

- **Thời hạn hoàn thành dự kiến:** Ngày 12/04/2026 sẽ hoàn tất toàn bộ mã nguồn, tuần học từ ngày 13/04/2026 đến 19/04/2026 sẽ hoàn thiện bản báo cáo bằng Latex.

---

## 4. Mã nguồn và quản lý dự án

Thầy có thể theo dõi chi tiết quá trình triển khai và cấu trúc mã nguồn của nhóm được trình bày tóm tắt trong tệp README.md và trình bày cụ thể trong tệp Summary.md tại đường dẫn GitHub: https://github.com/HieNguyen08/Introduction-AI-Assignment.git

---

Nhóm sẽ nỗ lực để hoàn thành các cột mốc tiếp theo đúng thời hạn. Em xin chân thành cảm ơn thầy!

Trân trọng,
Nguyễn Minh Hiếu
