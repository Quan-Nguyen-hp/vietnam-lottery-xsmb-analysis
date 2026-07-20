# Roadmap kiểm định lợi thế XPIS

## Mục tiêu

Chỉ thay đổi chính sách cược hoặc mức vốn khi hệ thống chứng minh được lợi thế ngoài mẫu có ý nghĩa thống kê. Không dùng LLM để chọn số, vì LLM không tạo thêm tín hiệu thống kê cho kết quả xổ số.

## Trạng thái hiện tại

- Backtest walk-forward theo semantics hiện tại (Kelly chỉ sizing), 730 ngày (`2024-07-08` đến `2026-07-15`) với
  `top_k=4`, `p >= 0.31`, `confidence >= 0.45` có ROI cược phẳng `-5.14%` trên 460 lượt cược.
- Khoảng ROI bootstrap 95% là `[-20.90%, +11.32%]`; permutation p-value `0.8060`;
  Statistical Edge Gate vẫn **FAIL**.
- Các sweep ngưỡng cũ dùng kết quả backtest có rò rỉ dữ liệu nên bị vô hiệu; không dùng chúng để thay policy.
- Kết quả hiện tại chỉ phù hợp cho paper-trade/theo dõi, không phải bằng chứng tăng vốn.
- **§5 Model Slimming**: `loto_repeat` là ứng viên loại, nhưng chưa đủ bằng chứng thống kê; production vẫn giữ đủ 11 models.
- **§6 Edge Gate**: Đã triển khai `EdgeGate` class, tích hợp vào production pipeline. Gate hiện tại FAIL (CI chứa 0).
- **§7 Kelly Transparency**: Đã triển khai kelly_policy, tách điểm vs tiền, disclaimer khi chưa khai báo vốn.
- **Kelly production**: Đã chuyển Kelly sang vai trò sizing-only; p/confidence quyết định BET, Kelly chỉ tính allocation. Kết quả cũ trước thay đổi không so sánh trực tiếp với production mới.

## 1. Pipeline chung cho production và backtest

**Trạng thái: đang triển khai**

- [x] Tạo hàm `run_shared_prediction_pipeline()` trong `daily_predict.py`.
- [x] Production gọi hàm chung qua wrapper `run_predict()`.
- [x] Thêm `run_exact_production_backtest()` gọi trực tiếp pipeline production, dùng lịch sử dự báo nội bộ và không ghi vào log production.
- [x] Chạy smoke test exact-mode 30 ngày (`2026-06-16` đến `2026-07-15`): 1 lượt cược, 0 nháy trúng. Xác nhận pipeline dùng chung hoạt động; mẫu quá nhỏ để đánh giá lợi thế.
- [ ] Chạy báo cáo exact-mode trên cửa sổ holdout đã khóa; chế độ này chậm vì recalibrate hằng ngày.

Tiêu chí hoàn thành: production và exact backtest có cùng training window, calibration, fusion, EMA weights, decision policy và output contract.

## 2. Khóa tập holdout

- [x] Khóa prospective holdout trong `predictions/evaluation_policy.json`: từ `2026-07-21`, tối thiểu 180 kỳ,
  sau khi áp dụng Kelly sizing-only và chuyển production sang `top_k=2`.
- [x] Thêm `backtests/run_locked_holdout.py`; runner từ chối chạy khi chưa đủ dữ liệu hoặc holdout không còn là cửa sổ mới nhất.
- [ ] Không thay đổi model/ngưỡng/top_k trước khi holdout đạt 180 kỳ.
- [ ] Dùng phần dữ liệu trước ranh giới cho lựa chọn model/ngưỡng.
- [ ] Chạy một lần exact-mode trên 180 kỳ holdout sau khi đủ dữ liệu.
- [ ] Không đổi ngưỡng dựa trên kết quả holdout.
- [x] Loại các quan sát 2026-07-16–2026-07-19 khỏi holdout vì dùng decision semantics cũ.

## 3. Baseline bắt buộc

- [x] Chọn ngẫu nhiên cùng số lượng cược và cùng lịch ngày với XPIS (seed 42).
- [x] Baseline tần suất đơn giản cùng mức exposure.
- [x] Baseline không cược.
- [x] Báo cáo ROI và bootstrap 95% đã sửa rò rỉ tại `backtests/results/xpis_backtest_no_leakage_365d.md`.

Kết quả exploratory 365 ngày đã sửa rò rỉ: XPIS `-31.25%`, ngẫu nhiên `+37.50%`, tần suất lịch sử `-31.25%`, cùng exposure 16 lượt cược; mọi khoảng bootstrap 95% đều chứa 0. Chưa có chiến lược nào chứng minh được lợi thế.

Chỉ giữ model/cấu hình khi vượt baseline ngoài mẫu với bootstrap 95%.

## 4. Kiểm định tín hiệu

- [x] Thêm permutation test 5.000 lần: giữ lịch/số XPIS đã chọn, xáo kết quả giữa các ngày.
- [x] Đo p-value: tỷ lệ PnL hoán vị lớn hơn hoặc bằng XPIS.
- [x] Ghi kết quả đã sửa rò rỉ vào `backtests/results/xpis_backtest_no_leakage_365d.md`.

Kết quả exploratory 365 ngày đã sửa rò rỉ: PnL XPIS `-135.000đ`, PnL hoán vị trung bình `+54.229đ`, p-value một phía `0,8726`. Không có bằng chứng về tín hiệu dự báo; giữ paper-trade và không diễn giải hiệu suất là lợi thế.

## 5. Làm gọn mô hình

- [x] Lập bảng đóng góp ngoài mẫu của từng model: Brier, log loss, Precision@K và ROI.
- [x] Báo cáo đã sửa rò rỉ tại `backtests/results/model_slimming_no_leakage_365d.md`.
- [x] Kết quả sàng lọc: `loto_repeat` là ứng viên loại (ROI < freq baseline AND Brier > median).
- [ ] Chỉ loại hoặc giảm trọng số khi CI95 của ΔROI ghép cặp Pruned-fixed − Full > 0.
- [x] Tạo `get_pruned_models()` trong `src/probability/__init__.py`.
- [x] Giữ `_PRUNED_MODEL_NAMES = set()` vì chưa có model nào vượt điều kiện loại.
- [x] Kiểm tra lại ensemble rút gọn trên validation; chỉ đánh giá cuối trên holdout.
  - Kết quả cũ 16 bets/ROI -31.25% đã vô hiệu vì dùng Kelly như cổng chọn cược.
  - Với Kelly sizing-only, Full 11: 250 bets, 68 hits, ROI -0.27%.
  - Bỏ riêng `loto_repeat` (10 models): 250 bets, 65 hits, ROI -4.67%.
  - ΔROI Pruned-fixed − Full: -4.40%, CI95 [-15.03%, +5.87%], P(ΔROI>0)=16.1% → không loại model.

## 6. Paper-trade Edge Gate

- [x] Tạo `src/decision/edge_gate.py` — EdgeGate class: check(), update(), force_fail(), force_pending().
- [x] Lưu trạng thái Edge Gate vào `predictions/evaluation_policy.json` (field `edge_gate`).
- [x] Tích hợp vào `daily_predict.py`: khi gate FAIL, force tất cả BET → PAPER_TRADE.
- [x] Print cảnh báo rõ ràng: "⛔ EDGE GATE FAIL — PAPER_TRADE MODE".
- [x] Tạo `backtests/evaluate_edge_gate.py` — chạy định kỳ để cập nhật gate state từ prediction_log.
- [ ] Chạy evaluate_edge_gate.py sau khi holdout đủ 180 kỳ.

## 7. Báo cáo Kelly minh bạch

- [x] Thêm `kelly_policy` vào `predictions/evaluation_policy.json` (declared, starting_capital_points, point_value_vnd).
- [x] Sửa `DayDecision` trong `src/decision/engine.py`: thêm fields kelly_capital_declared/points/point_value.
- [x] `DayDecision.to_dict()` xuất `kelly_transparency` + `allocation_points` / `allocation_vnd` riêng biệt.
- [x] Sửa `print_prediction()` trong `daily_predict.py`: hiển thị vốn khi đã khai báo, disclaimer khi chưa.
- [x] Sửa `prediction_dashboard.py`: thêm disclaimer Kelly transparency.
- [ ] Người dùng khai báo vốn trong evaluation_policy.json khi sẵn sàng.

## Nguyên tắc thay đổi policy

1. Không hạ ngưỡng chỉ để tăng số ngày cược.
2. Không chọn cấu hình theo ROI của chính holdout.
3. Mọi thay đổi phải có báo cáo tái lập được, seed cố định và so sánh baseline.
4. Khi bằng chứng không đủ, giữ chế độ paper-trade.

## 8. Tách xác suất xuất hiện và kỳ vọng số nháy

- [x] Bổ sung `EvaluationMetrics.count_forecast_metrics()` cho MAE, RMSE, Poisson deviance và count calibration.
- [x] Công thức kinh tế đúng: `EV = 99 × E[count] − 27`; hòa vốn tại `E[count] > 0.272727`.
- [x] Thêm nghiên cứu walk-forward tại `backtests/count_expectation_research.py`.
- [x] Chạy báo cáo 365 ngày và 3 năm, không leakage.
- [x] Xác nhận uniform forecast `E[count]=0.27`, tương ứng ROI nền lý thuyết khoảng `-1%`.
- [x] `EWMA half-life 90 + LCB gate` là challenger thăm dò: 3 năm ROI `+10.59%`, CI95 `[+0.21%, +21.03%]`.
- [ ] Không promote challenger: permutation p=`0.2028`, CI từng epoch 365 ngày đều chứa 0 và kết quả chưa điều chỉnh multiple testing.
- [x] Tích hợp `EWMA half-life 90 + LCB` vào shadow paper-trade; không ảnh hưởng bets production.
- [x] `daily_update.py` chấm riêng hits, chi phí, doanh thu và PnL giả lập của challenger.
- [ ] Chỉ cân nhắc production sau prospective validation độc lập; không tune challenger trên shadow results.
- [x] Thử thêm `count_ewma_poisson` vào MetaFusion thành ensemble 12 model, research-only.
  - Full 11 model: 250 bets, 68 hits, ROI -0.27%, CI95 [-21.12%, +22.22%].
  - Full 12 model: 169 bets, 45 hits, ROI -2.37%, CI95 [-29.49%, +27.42%].
  - Model mới có Brier 0.182358 và ECE 0.005438 nhưng không cải thiện hiệu suất; không đưa vào production.

## 9. Theo dõi prospective count challenger

- [x] Tạo `backtests/evaluate_count_challenger.py`, chỉ đọc prediction log và không thay production.
- [x] Count challenger vẫn theo dõi riêng từ `prospective_start_date=2026-07-20`; không dùng cửa sổ này để đánh giá policy Top-2.
- [x] Đếm đúng nhiều nháy, chi phí, PnL và bootstrap ROI CI95.
- [x] Gate luôn `PENDING` trước khi đủ 180 ngày; không tự động promote.
- [x] Tích hợp evaluator vào `auto_runner.py` sau bước cập nhật kết quả.
- [x] Tạo báo cáo `backtests/results/count_challenger_prospective.md`.

## 10. Constrained stacking thay MetaFusion heuristic

- [x] Tạo `ConstrainedStacking` challenger với optimizer simplex và regularization về uniform.
- [x] Ràng buộc trọng số không âm và tổng trọng số bằng 1.
- [x] Huấn luyện walk-forward trên 45 ngày fit; retrain mỗi 30 ngày.
- [x] Chạy benchmark 365 ngày với Kelly sizing-only.
  - Exposure: 98 bets, 84 ngày cược; ROI `+8.50%`, CI95 `[-29.79%, +50.16%]`.
  - Ranking-only Top-4: ROI `-4.57%`, CI95 `[-13.61%, +4.22%]`.
  - Permutation p-value: `0.5587`.
- [x] Chạy thêm benchmark 730 ngày (2024-07-08 đến 2026-07-15).
  - Exposure: 119 bets, 103 ngày cược; ROI `+14.01%`, CI95 `[-21.69%, +52.78%]`.
  - Ranking-only Top-4: ROI `-5.57%`, CI95 `[-12.23%, +1.08%]`.
  - Permutation p-value: `0.2993`.
- [x] So sánh 730 ngày với uniform fusion trên cùng Top-4 exposure.
  - Stacking: Brier `0.182037`, ECE `0.004663`, Precision@4 `0.2288`, 752 nháy, ranking ROI `-5.57%`.
  - Uniform: Brier `0.182867`, ECE `0.018122`, Precision@4 `0.2315`, 752 nháy, ranking ROI `-5.57%`.
- [x] Loại challenger: calibration tốt hơn nhưng không vượt uniform về ranking/ROI; CI95 chứa 0 và permutation không có ý nghĩa.
- [x] Production giữ MetaFusion champion.

## 11. Uniform fusion như policy hoàn chỉnh

- [x] Thêm cờ research-only `uniform_fusion` vào walk-forward backtest; không thay production.
- [x] Giữ nguyên 11 models, calibration LightGBM, confidence, ngưỡng, Top-K, diversification và Kelly sizing-only.
- [x] Chạy benchmark 730 ngày trên đúng cửa sổ `2024-07-08` đến `2026-07-15`.
  - Uniform policy: 1.766 bets/648 ngày cược, ROI `-8.02%`, CI95 `[-16.40%, +0.60%]`,
    permutation p-value `0.9810`.
  - MetaFusion policy: 460 bets/351 ngày cược, ROI `-5.14%`, CI95 `[-20.90%, +11.32%]`,
    permutation p-value `0.8060`.
- [x] Ranking-only Top-4:
  - Uniform: 752 nháy, ROI `-5.57%`, CI95 `[-11.97%, +1.08%]`.
  - MetaFusion: 737 nháy, ROI `-7.45%`, CI95 `[-13.86%, -0.80%]`.
- [x] Kết luận: uniform ranking tốt hơn MetaFusion, nhưng xác suất uniform không tương thích với absolute gate `p >= 0.31`,
  làm exposure tăng 3,84 lần và ROI policy xấu hơn. Không promote uniform fusion.
- [x] Thử challenger tách gate/ranking: giữ đúng ngày và số lượng BET của MetaFusion, dùng uniform score để chọn Top-N.
  - Cùng exposure: 460 bets.
  - Gated uniform ranking: ROI `-11.52%`, CI95 `[-26.83%, +4.99%]`.
  - MetaFusion champion: ROI `-5.14%`, CI95 `[-20.90%, +11.32%]`.
  - Paired ΔROI (challenger − champion): `-6.38%`, CI95 `[-23.06%, +10.27%]`,
    P(ΔROI>0)=`21.1%`.
- [x] Loại gated-uniform ranking; MetaFusion tiếp tục là champion. Không thay policy production hoặc holdout.
- [ ] Không tiếp tục tối ưu fusion trước khi có thêm dữ liệu prospective; các thử nghiệm tiếp theo chỉ được mở khi có giả thuyết định trước.

## 12. Giám sát holdout prospective

- [x] Tạo `backtests/holdout_status.py` để kiểm tra số ngày holdout đã có, khoảng ngày dữ liệu và số prediction log đã hoàn tất.
- [x] Loại rõ các bản ghi trước `2026-07-21` khỏi trạng thái holdout; dự đoán 2026-07-20 dùng policy cũ nên không được tính.
- [x] Tích hợp kiểm tra vào `auto_runner.py` ở cả phiên trước và sau xổ số.
- [x] Giữ script chỉ đọc: không thay model, threshold, top_k, policy hoặc Edge Gate.
- [x] Hiển thị trạng thái holdout trên `prediction_dashboard.py`, kể cả khi chưa có kết quả nào.
- [x] Xuất trạng thái dạng JSON tại `backtests/results/locked_holdout_status.json`.
- [x] Hiển thị cùng trạng thái trên web dashboard `index.html`.
- [x] Web dashboard dùng cache-busting và fallback an toàn khi artifact holdout chưa được deploy.
- [x] Khi `SKIP`, web dashboard ẩn hoàn toàn các ứng viên xác suất thô để tránh bị hiểu nhầm thành khuyến nghị.
- [x] Khi có số được chọn, khối xác suất chỉ hiển thị đúng các số trong `bets`, kèm cảnh báo paper-trade.
- [ ] Cập nhật GitHub Pages artifact để chứa `backtests/results/locked_holdout_status.json` và `predictions/evaluation_policy.json`.
- [ ] Chờ dữ liệu từ `2026-07-21` đủ tối thiểu 180 ngày rồi mới chạy locked holdout exact-mode.

## 13. Top-K ablation: chỉ chọn một số

- [x] Chạy MetaFusion walk-forward 730 ngày với `top_k=1`, giữ nguyên models, calibration, probability/confidence gate,
  diversification và Kelly sizing-only.
- [x] Policy có gate (tối đa 1 số/ngày):
  - 352 bets/352 ngày cược; 99 nháy.
  - ROI `+3.12%`, PnL `+297.000đ`, CI95 `[-15.47%, +22.91%]`.
  - P(ROI>0)=`61.7%`; permutation p-value=`0.4941`.
- [x] Ranking-only, bắt buộc Top-1 mỗi ngày:
  - 730 bets; 206 nháy.
  - ROI `+3.47%`, CI95 `[-10.09%, +17.53%]`, P(ROI>0)=`67.8%`.
- [x] So với Top-4 cùng cửa sổ: Top-4 policy ROI `-5.14%`, CI95 `[-20.90%, +11.32%]`.
- [ ] Không promote `top_k=1`: CI95 vẫn chứa 0 và permutation test không có ý nghĩa thống kê.
  Production đã chọn `top_k=2` như một giới hạn exposure thận trọng; Top-1 chỉ là challenger cho một holdout tương lai.
- [x] Chạy thêm Top-2 và Top-3 cùng cửa sổ:
  - Top-2: 442 bets/351 ngày cược, ROI `-2.11%`, CI95 `[-18.24%, +14.79%]`,
    permutation p=`0.6961`.
  - Top-3: 459 bets/351 ngày cược, ROI `-4.94%`, CI95 `[-20.72%, +11.59%]`,
    permutation p=`0.7972`.
- [x] Xếp hạng exploratory theo ROI: Top-1 `+3.12%` > Top-2 `-2.11%` > Top-3 `-4.94%` > Top-4 `-5.14%`.
  Đây là kết quả chọn cấu hình trên cùng cửa sổ, không phải bằng chứng độc lập; không dùng để vượt Edge Gate.
- [x] Xác nhận forced Top-K (đủ số mỗi ngày) trong `forced_topk_ranking_2y_summary.md`:
  Top-1 `+3.47%`, Top-2 `-7.58%`, Top-3 `-7.25%`, Top-4 `-7.45%`.
- [x] Theo quyết định vận hành, đổi production từ `top_k=4` sang `top_k=2` để giảm exposure dư thừa.
  Đây không phải Edge Gate promotion; hệ thống tiếp tục paper-trade và holdout được khóa lại từ 2026-07-21.
