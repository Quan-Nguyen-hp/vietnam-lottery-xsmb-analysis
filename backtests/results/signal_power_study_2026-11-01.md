# Nghiên cứu công suất cao: có tín hiệu dự báo XSMB ngoài mẫu không? — 01/11/2026

**Kết luận: NO RELIABLE IMPROVEMENT FOUND** (kết luận loại B). Không đổi production, policy, threshold hay Top-K. Có một cấu trúc thống kê nhỏ nhưng thật (tần suất chữ số đầu) mà **không đủ để khai thác** và **chưa thể xác nhận độc lập**.

Bổ sung cho `accuracy_audit_2026-10-03.md` (sửa lỗi correctness + replay 176 draw). Báo cáo đó chỉ có 176 draw nên công suất thấp. Ở đây dùng **7.042 draw** (2007 → 20/07/2026) bằng harness nhanh, tách khỏi pipeline production.

## Giao thức (chốt trước khi xem kết quả)

- Dữ liệu cắt cứng trước holdout khóa 21/07/2026 (đọc từ `evaluation_policy.json`; test khẳng định). Holdout **không được chạm**.
- Tín hiệu tại draw t chỉ dùng `c[:t]`. Kiểm bằng poison-test: đổi toàn bộ dữ liệu từ t trở đi thành rác, hàng t không đổi (PASS, thêm vào `tests/test_signal_power_study.py`).
- Ba giai đoạn chronological: DEV 2007–2015 (3.254 draw), VAL 2016–2020 (1.785), FINAL 2021–trước holdout (2.003).
- Metric theo kinh tế sản phẩm (99k/nháy, 27k/số): IC Spearman theo ngày và **excess hits** của top-K so với TB cả 100 số cùng ngày (kỳ vọng ngẫu nhiên chính xác, không nhiễu realization). Break-even: excess ≥ 27/99 − 0,27 = **+0,00273 nháy/số**.
- Bất định: block bootstrap 7 ngày, seed 42; null dịch vòng nhãn (3.000 lần); Bonferroni theo số tín hiệu. `RANDOM_noise` làm điều khiển hiệu chuẩn (|t| ≤ 1,57).

## Đã thử (22 tín hiệu từ hệ thống hiện có + 1 ridge)

freq 7/14/30/60/90/180/365/all, EWMA half-life 15/45/90/180/365, delay, delay/mean-gap, lặp hôm qua/2 ngày, số lộn, tần suất cùng thứ trong tuần, tần suất chữ số đầu/đuôi 90 draw, và **kết hợp ridge walk-forward** của 10 tín hiệu (cận trên "học được" từ các feature này).

## Kết quả

| Câu hỏi | Kết quả |
|---|---|
| Có tín hiệu vượt Bonferroni (|t| ≥ 3,05, 22 tín hiệu) trong một giai đoạn? | **Không.** Cao nhất DEV: head_freq_90 t = 2,92. |
| Delay, repeat, số lộn, day-of-week, Markov-style | IC ≈ 0, đổi dấu giữa các giai đoạn → không có thông tin. |
| Ridge kết hợp có hơn từng tín hiệu đơn? | **Không.** top-2 excess VAL −0,0123, FINAL −0,0036; IC VAL 0,0009. Thông tin trong các feature hiện có không học được thêm. |
| Họ tần suất/EWMA (top-2) | Hầu hết excess dương ở cả 3 giai đoạn (≈ +0,003…+0,019), nhưng null dịch vòng: tốt nhất freq_90 gộp p = 0,005 → Bonferroni ×13 ≈ **0,065**; không CI nào gộp loại 0 một cách đáng tin cậy sau hiệu chỉnh. |
| Tần suất **chữ số đầu** 30/60/90 draw | **Thật nhưng rất nhỏ**: IC gộp +0,0040…+0,0046, t gộp **3,35–3,85** (Bonferroni 32 tín hiệu: 3,16), IC dương ở cả DEV/VAL/FINAL (0,0051/0,0043/0,0041), 15/20 năm dương (binomial p = 0,02). Cửa sổ 180/365 mất hiệu lực (t 1,0–1,6). |
| Khai thác được? | **Không chứng minh được.** Chọn trực tiếp theo head chỉ cho excess ≈ +0,001 (< break-even 0,0027). Hai quy tắc chọn thử (H1, H2): H1 âm ở VAL/FINAL; H2 (z head+tail+freq90) dương ở cả 6 ô (ROI ước lượng +0,2% … +3,4%) nhưng mọi CI95 chứa 0 (ngoại trừ một ô). |

## Điều có vẻ hứa hẹn nhưng không vượt kiểm định

- Tần suất 90 draw: nhìn tốt trên replay 176 draw (28,4%) và trên top-2 VAL (+0,0187, CI [+0,0025; +0,0350]); nhưng đó là cực đại trong nhiều phép thử, không lặp lại nhất quán (DEV CI chứa 0, p hiệu chỉnh 0,065).
- Head frequency: đáng chú ý về mặt thống kê, nhưng IC ≈ 0,0045 ⇒ giới hạn trên excess top-2 chỉ cỡ +0,005 nháy/số, tức ngang break-even. Muốn phát hiện excess cỡ đó ở mức SE ~0,004 cần xấp xỉ 7.000 draw cho **một** tham số; sai số ROI theo năm lớn hơn nhiều so với lợi thế.
- **Tín hiệu head được phát hiện sau khi nhìn cả ba giai đoạn**, nên VAL/FINAL không còn là xác nhận độc lập. Đây là thăm dò.

## Quyết định

| Hạng mục | Quyết định |
|---|---|
| Thay đổi mã production | Không. Không có bằng chứng out-of-sample đủ mạnh; chỉ giữ các sửa correctness từ 03/10. |
| Loại tín hiệu yếu khỏi fusion | Chưa: ablation fusion 03/10 và ridge ở đây không cho thấy loại bỏ giúp được; CI chứa 0. Giữ nguyên để không đổi business semantics. |
| Promote challenger | Không. Edge Gate (bootstrap ROI lower 95% > 0) vẫn không đạt. |
| Hướng nghiên cứu nhỏ nhất tiếp theo | Đóng băng **một** quy tắc (H2) và chỉ đánh giá prospective trên dữ liệu sau holdout đủ dài (cỡ mẫu đúng được tính trong `frozen_challenger_v1/PREREGISTRATION.md`; ghi chú ban đầu "≥ 700 draw để SE excess ≈ 0,004" là SAI vì sd excess top-2 theo ngày ≈ 0,375 nên SE = 0,375/√N). Không dùng holdout 180 ngày hiện tại để tinh chỉnh; không thêm indicator. |

## Giới hạn

- Harness nhanh không phải pipeline production (không LightGBM/calibration/EMA/regime); câu "pipeline production kém hơn tần suất đơn giản" chỉ được ủng hộ bởi replay 176 draw (chưa có ý nghĩa thống kê).
- Dữ liệu là snapshot hiện tại, không có revision theo thời điểm công bố; 4 ngày 16–19/02/2026 thiếu.
- Toàn bộ 2007–2026 đã được nghiên cứu nhiều lần trong repository nên không còn vùng "chưa nhìn" nào ngoài holdout khóa.
- Null dịch vòng với tín hiệu chậm có phân phối lệch; dùng p thô + Bonferroni thay vì max-z để hiệu chỉnh.

## File và tái lập

- `backtests/signal_power_study.py` (`--poison-check`, `--shift-null`), `backtests/head_signal_followup.py`, `tests/test_signal_power_study.py`.
- Output: `backtests/results/signal_power_study.{md,json}`, `signal_power_study_shift_null.json`, `head_signal_followup.{md,json}`.
- Chạy lại: `python backtests/signal_power_study.py --poison-check`, `python backtests/signal_power_study.py --shift-null`, rồi `PYTHONPATH=backtests python backtests/head_signal_followup.py`.
- Không sửa `predictions/`, `evaluation_policy.json`, prediction log hay holdout.
