# Kiểm toán độ chính xác XSMB — 03/10/2026

Mục tiêu: cải thiện dự báo thực sự ngoài mẫu hoặc kết luận không tìm thấy cải thiện đáng tin cậy. Không dùng LLM chọn số, không sửa policy hay dùng holdout để chọn mô hình.

## Đường dự báo thực tế

CSV 27 giải → `DataLoader` ma trận xuất hiện nhị phân `int8` → evidence trước ngày T → 6 nhóm feature → 11 mô hình thống kê + LightGBM → hiệu chuẩn 45 ngày fit / 45 ngày selection → fusion theo chất lượng, EMA từ prediction log và boost regime → p ≥ 0,31, confidence ≥ 0,45 → tối đa 2 số → diversification → paper-trade qua Edge Gate.

LightGBM học nhãn **xuất hiện ít nhất một lần**, không học số nháy. Huấn luyện 365 ngày kết thúc trước 90 ngày validation. Các nhãn validation đều trước T. Cả 100 số là ứng viên; không có bước LLM sinh số. `daily_update.py` chấm số nháy, trả 99 nghìn/nháy và chi phí 27 nghìn/số. Backtest nhanh retrain 30 ngày, thiếu EMA/regime giống production, nên không dùng thay exact baseline.

## Protocol chốt trước khi xem kết quả replay

- Dữ liệu hiện có đến 29/08/2026; holdout khóa từ 21/07/2026. Không chấm hoặc tối ưu trên holdout.
- Exact replay: 22/01–20/07/2026, 180 kỳ lịch dự kiến, 30 draw warm-up trước đó. Chia ba đoạn liên tiếp để kiểm tra ổn định.
- Trạng thái khởi đầu: log rỗng, EMA bắt đầu lại và alpha tại draw đầu đánh giá là 0,6. Đây là controlled replay cùng initial state, không tái tạo đầy đủ state log production lịch sử.
- CSV đưa vào model được cắt trước holdout. Mỗi dự báo T chỉ dùng lịch sử có ngày < T. Gọi trực tiếp `run_shared_prediction_pipeline`, giữ nguyên Top-2, ngưỡng, seed và business semantics.
- Cache, edge artifacts và log của thí nghiệm ở `.cache/accuracy-*`; không ghi log hoặc policy production. Bản nguồn ban đầu lưu trong `.cache/accuracy-original/source`.
- Chấm precision@2, recall, draw hit rate, số nháy/draw, phân phối số nháy, exposure và ROI. Policy có SKIP được báo riêng với ranking bắt buộc Top-2; các challenger cũng được so trên cùng exposure từng ngày.
- Đối chứng cố định: random seed 42, binary frequency toàn lịch sử/30/90/365 draw, existing count EWMA half-life 90/window 365, uniform fusion của các model hiện hữu.
- Thử shrink xác suất production 50% về `1 - 0.99**27` và uniform hoàn toàn như giả thuyết giảm nhiễu; uniform là mô hình giả định 27 vị trí độc lập đồng xác suất, không phải chứng minh cơ chế xổ số.
- Ablation research cố định trước so sánh kết quả hoàn chỉnh: loại riêng `max_delay`, `bayesian_predictor`, `loto_repeat` khỏi fusion, chuẩn hóa lại trọng số; đối chiếu cả forced Top-2 và cùng exposure. Không chọn lại trọng số, không thêm chỉ báo.
- CI95 ghép cặp cho precision@2: moving-block bootstrap 7 draw, 5.000 lần, seed 42. Không xem tăng điểm nhỏ trong một đoạn là cải thiện đáng tin cậy.
- Đây là replay lịch sử đã từng được nghiên cứu, **không phải final validation độc lập**. Không promote bất kỳ challenger nào từ các điểm này.

## Các lỗi ưu tiên

1. `int8` matrix multiplication tràn ở conditional feature, fallback Markov, inverted pairs và NMI.
2. Prediction log không lọc ngày trước target khi chọn EMA weights.
3. Cache chỉ khóa theo ngày, không gắn với lịch sử nguồn; `use_cache=False` vẫn đọc Parquet.
4. Array trong evidence đọc từ Parquet bị kiểm tra truth-value, gây extractor thất bại và thiếu cột mà pipeline vẫn tiếp tục.

Sửa tối thiểu bằng phép đếm an toàn, kiểm tra lịch sử và fingerprint cache, xử lý list/array nhất quán, namespace feature mới, và dừng rõ nếu extractor thất bại. Kiểm thử RED→GREEN trước sửa. Giữ mọi output cũ và holdout policy; đánh dấu phiên bản mới để không gộp các dự báo từ pipeline khác nhau.
