# Lộ trình Kiểm định Ngoài mẫu & Tiêu chí Tốt nghiệp (Phase 3 — EVM-1)

Tài liệu này xác lập các tiêu chí định lượng khắt khe nhất để kiểm định hiệu năng dự báo thực tế ngoài mẫu (Out-of-Sample) của hệ thống XPIS v1.2 trong giai đoạn **Edge Monitoring dài hạn**, tuân thủ nguyên tắc đóng băng kiến trúc hạ tầng và tập trung tối ưu hóa lợi thế toán học (Edge).

---

## 🎯 1. Tiêu chí Tốt nghiệp EVM-1 (Graduation Criteria)

Hệ thống XPIS v1.2 chỉ được phép tốt nghiệp và chuyển giao sang vận hành tự động toàn phần sau giai đoạn **300 đến 500 kỳ quay ngoài mẫu thực tế** nếu đạt đồng thời các chỉ số thống kê dưới đây:

### A. Phép Thử Đối Đối Chiếu Bắt Buộc (Baselines / Adversaries)
Hiệu năng dự báo của XPIS v1.2 phải liên tục so sánh đối chiếu chéo với 3 đối thủ baseline:
1. **Uniform Probability Baseline**: Phân phối ngẫu nhiên đều đại diện cho mức nổ ngẫu nhiên của nhà đài.
2. **Best Single Component Baseline**: Mô hình đơn lẻ có hiệu năng dự báo tốt nhất trong hệ thống (ví dụ: LightGBM).
3. **Previous Production Version**: Phiên bản XPIS v1.0/v1.1 cũ để khẳng định không bị thụt lùi hiệu năng.

### B. Chỉ số Đánh giá Chính và Phụ (Primary & Secondary Metrics)
* **Chỉ số Đánh giá Chính (Primary Metrics — Ưu tiên số 1)**:
  * **Brier Score**: $\le 0.2200$.
  * **Expected Calibration Error (ECE)**: $\le 0.0800$.
  * **Log Loss / Reliability**: Duy trì ổn định, không xuất hiện calibration drift kéo dài.
* **Chỉ số Đánh giá Phụ (Secondary Metrics — Ưu tiên số 2)**:
  * **Kelly ROI / PnL**: ROI lũy kế dương sau khi trừ đi chi phí ($27$k/số cược).
  * **Maximum Drawdown**: $\le 35\%$ tổng quỹ vốn phân bổ Kelly.
  *(Chỉ số ROI có phương sai lớn và chịu ảnh hưởng của may rủi ngắn hạn nên chỉ đóng vai trò thứ cấp so với chất lượng phân phối xác suất thô Brier/ECE)*

### C. Kiểm Định Ý Nghĩa Thống Kê (Statistical Significance Test)
Để loại bỏ hoàn toàn yếu tố ngẫu nhiên, XPIS v1.2 chỉ được phép tốt nghiệp nếu phép thử giả thuyết thống kê **Paired Bootstrap** hoặc **Permutation Test** đối chiếu giữa XPIS và Best Single Component cho ra kết quả:
* Khoảng tin cậy 95% của độ lệch Brier Score:
  $$\Delta Brier = Brier(XPIS) - Brier(BestComponent) \le -0.0100$$
* Khoảng tin cậy **$CI95$ của $\Delta Brier$ hoàn toàn loại trừ điểm 0** ($95\%$ CI does not contain $0.0$).

---

## 📉 2. Giám sát Rolling Window & Cảnh báo Sớm (Prediction Drift Dashboard)

Để phát hiện sớm sự suy thoái hiệu năng ngoài mẫu, hệ thống tự động đo lường và vẽ biểu đồ rolling cửa sổ trượt:
* **Rolling Brier & ECE (30 ngày)**: Phát hiện suy thoái phân phối xác suất do trôi dạt dữ liệu (data drift).
* **Rolling Kelly ROI (30 ngày & 90 ngày)**: Theo dõi biến động PnL và sụt giảm vốn.
* **Feature & Belief Drift Dashboard**: Log chi tiết hàng ngày để truy vết đóng góp của từng Node.

---

## 🍂 3. Cơ chế Lão hóa Tri thức (Knowledge Aging / Belief Decay)

Nhằm đảm bảo tri thức luôn phản ánh bằng chứng thực tế mới nhất:
* Nếu một Belief đã được phê duyệt (`Validated`) nhưng **không nhận được bất kỳ thực nghiệm mới hoặc dự báo ngoài mẫu mới nào hỗ trợ ghi nhận trong vòng 180 ngày liên tiếp**, độ tin cậy `confidence` của Belief đó sẽ tự động suy giảm tuyến tính với tốc độ **`-0.05` mỗi 30 ngày quá hạn**.
* Khi có dự báo hoặc thực nghiệm mới được ghi nhận (cập nhật bằng chứng mới nhất), thời gian trễ sẽ tự động được reset về 0, bảo vệ các tri thức đang hoạt động khỏi bị suy thoái.
* Khi độ tin cậy rơi xuống dưới ngưỡng thiết lập, Belief sẽ tự động bị hạ cấp:
  $$\text{Validated} \ (\ge 0.80) \ \longrightarrow \ \text{Experimental} \ (\ge 0.50) \ \longrightarrow \ \text{Deprecated} \ (< 0.50)$$

---

## 🔒 4. Nguyên tắc Quản trị Sandbox & Đóng băng Production

1. **Đóng băng Tuyệt đối Kiến trúc Nhánh Chính (Main Branch Freeze)**: Không thêm module mới vào Main.
2. **Cách ly Sandbox**: Mọi phát triển feature mới chỉ thực hiện trên nhánh `sandbox` độc lập.
3. **Cửa ngõ Tích hợp (Gate)**: Chỉ merge sandbox vào Main khi kiểm định chéo chứng minh cải thiện đáng kể có ý nghĩa thống kê ($p < 0.05$).
