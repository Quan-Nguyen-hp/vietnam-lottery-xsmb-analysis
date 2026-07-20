# Forced Top-K ranking — tổng hợp 730 ngày

Đây là chế độ **mỗi ngày bắt buộc chọn đúng K số**, lấy Top-K theo MetaFusion,
không áp dụng probability gate, confidence gate, diversification hay Kelly.
Mục đích là đo riêng chất lượng xếp hạng.

| K | Bets | Hits | ROI | Bootstrap CI95 | P(ROI>0) |
|---:|---:|---:|---:|---|---:|
| 1 | 730 | 206 | **+3.47%** | [-10.09%, +17.53%] | 67.8% |
| 2 | 1,460 | 368 | -7.58% | [-16.62%, +1.96%] | 5.7% |
| 3 | 2,190 | 554 | -7.25% | [-14.78%, +0.62%] | 3.5% |
| 4 | 2,920 | 737 | -7.45% | [-13.86%, -0.80%] | 1.4% |

## Kết luận

- Nếu bắt buộc đủ số mỗi ngày, Top-1 là cấu hình tốt nhất trong cửa sổ này.
- Top-2/3/4 đều có ROI âm; Top-4 có cận dưới CI95 âm trong ranking-only.
- Top-1 vẫn chưa đạt Edge Gate vì CI95 còn chứa 0.
- Đây là nghiên cứu xếp hạng, không phải policy production; production vẫn giữ
  bộ lọc rủi ro và paper-trade.
