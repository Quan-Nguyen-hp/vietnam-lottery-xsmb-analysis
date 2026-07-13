# Hướng dẫn sử dụng hệ thống dự đoán thích ứng XSMB

## 📁 Cấu trúc

```
predictions/
├── prediction_log.json     # Log tất cả dự đoán + kết quả
└── adaptive_weights.json   # Trọng số Ensemble (cập nhật mỗi 7 ngày)

daily_predict.py            # Dự đoán hàng ngày
daily_update.py             # Cập nhật kết quả + tính lại trọng số
prediction_dashboard.py     # Dashboard thống kê
```

---

## 🔄 Quy trình hàng ngày

### Bước 1 — Trước khi xổ số (sáng sớm)
```bash
python daily_predict.py
```
- Tự động fetch dữ liệu mới nhất từ web
- Chạy Weighted Ensemble với trọng số thích ứng
- Lưu dự đoán vào `prediction_log.json`
- In ra **4 số được chọn** cho ngày hôm nay

### Bước 2 — Sau 18:35 (khi có kết quả xổ số)
```bash
python daily_update.py
```
- Tự động fetch kết quả thực tế từ web
- Tính số nháy trúng, doanh thu, lãi/lỗ
- Cập nhật log
- **Mỗi 7 ngày**: tự điều chỉnh trọng số Ensemble theo hiệu suất

### Bất kỳ lúc nào
```bash
python prediction_dashboard.py          # xem toàn bộ lịch sử
python prediction_dashboard.py --days 30  # 30 ngày gần nhất
```

---

## ⚙️ Tùy chọn CLI

### daily_predict.py
| Tùy chọn | Mô tả |
|---|---|
| `--date YYYY-MM-DD` | Dự đoán cho ngày cụ thể (mặc định: hôm nay) |
| `--top-k N` | Số lượng số chọn (mặc định: 4) |
| `--dry-run` | Chỉ in kết quả, không lưu log |
| `--no-fetch` | Bỏ qua bước fetch dữ liệu mới |

### daily_update.py
| Tùy chọn | Mô tả |
|---|---|
| `--date YYYY-MM-DD` | Cập nhật ngày cụ thể (mặc định: hôm qua nếu trước 18:35) |
| `--force-reweight` | Bắt buộc tính lại trọng số ngay |

### prediction_dashboard.py
| Tùy chọn | Mô tả |
|---|---|
| `--days N` | Chỉ xem N ngày gần nhất |

---

## 💡 Thông số tối ưu (từ backtest 1000 ngày)

| Thông số | Giá trị |
|---|---|
| Phương pháp | Weighted Ensemble (5 mô hình) |
| top_k | **4 số/ngày** (ROI tốt nhất: +3.49%) |
| Chi phí | 4 × 27,000đ = **108,000đ/ngày** |
| Vốn dự phòng cần có | ~1,728,000đ (chịu chuỗi thua 16 ngày) |
| Tỉ lệ thắng ngày | ~32% |
| Tháng có lãi | ~50% số tháng |

---

## ⚖️ Thuật toán cập nhật trọng số

Mỗi 7 ngày có kết quả:
1. Tính ROI của từng phương pháp trong 7 ngày gần nhất
2. So sánh với ROI trung bình tất cả phương pháp
3. Phương pháp ROI cao → tăng trọng số; ROI thấp → giảm
4. Trọng số giới hạn trong **[0.1, 10.0]**

```
new_weight = old_weight × (1 + 0.15 × (roi_method - roi_avg) / 100)
```

---

## 📊 Kết quả backtest (top_k=4, 1000 ngày)

| Khoảng | ROI |
|---|---|
| 30 ngày gần nhất | **+46.67%** |
| 90 ngày | +14.07% |
| 365 ngày | +3.72% |
| 1000 ngày | +3.49% |
