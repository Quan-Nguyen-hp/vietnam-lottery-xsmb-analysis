# Tổng hợp ablation bộ lọc rủi ro — 365 ngày

- **Cửa sổ**: 2025-07-12 đến 2026-07-15
- **Tham số chung**: `top_k=4`, `min_probability=0.31`, `min_confidence=0.45`
- **Mục đích**: chẩn đoán exposure thấp; không dùng để thay đổi holdout policy.

| Kịch bản | Thay đổi | Bets | Ngày cược | Hits | ROI | CI95 ROI | P(ROI>0) |
|---|---|---:|---:|---:|---:|---|---:|
| A — Hiện tại | Kelly quyết định BET + diversification bật | 16 | 15 | 3 | -31.25% | [-100.00%, +52.78%] | 19.6% |
| B — Kelly chỉ phân bổ | `p/confidence` quyết định BET; Kelly chỉ sizing | 250 | 190 | 68 | -0.27% | [-21.12%, +22.22%] | 48.4% |
| C — Tắt diversification | Kelly quyết định BET; diversification tắt | 16 | 15 | 3 | -31.25% | [-100.00%, +52.78%] | 19.6% |

## Kết luận

1. **Bộ lọc diversification không gây ra exposure thấp**: A và C giống nhau về số cược, hits và ROI.
2. **Kelly đang làm giảm exposure mạnh**: bỏ Kelly khỏi điều kiện chọn cược tăng từ 16 lên 250 số.
3. **Nhưng tăng exposure chưa tạo ra lợi thế**: kịch bản B vẫn có ROI âm nhẹ và CI95 chứa 0.
4. Vì vậy, có cơ sở sửa kiến trúc theo hướng **Kelly chỉ phân bổ vốn**, nhưng chưa có cơ sở bật cược tiền thật hoặc hạ thêm ngưỡng.
5. Edge Gate vẫn **FAIL**; cả ba nhánh chỉ là paper-trade.

Các báo cáo chi tiết:

- [A — hiện tại](risk_filter_ablation_A_current_365d.md)
- [B — Kelly chỉ sizing](risk_filter_ablation_B_kelly_sizing_only_365d.md)
- [C — không diversification](risk_filter_ablation_C_no_diversification_365d.md)
