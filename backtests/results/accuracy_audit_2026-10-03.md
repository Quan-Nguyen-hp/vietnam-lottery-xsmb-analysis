# Kiểm toán và thực nghiệm độ chính xác XSMB — 03/10/2026

**Kết luận: NO RELIABLE IMPROVEMENT FOUND.** Giữ các sửa lỗi tính đúng đắn và chặn thông tin tương lai; không promote challenger, không đổi threshold/Top-K, không chuyển khỏi paper-trade.

Baseline forced Top-2: **25.28%**. Sau sửa correctness: **24.43%**. Chênh lệch **-0.85 điểm phần trăm** (-3.37% tương đối), CI95 ghép cặp **[-2.56; +0.85] điểm %**. Đây là replay lịch sử, không phải bằng chứng cải thiện tương lai đã được xác nhận.

## Pipeline đã kiểm tra

`data/xsmb-2-digits.csv` → `DataLoader` → lịch sử có ngày < T → EvidenceBuilder → 6 nhóm feature → 11 mô hình thống kê + LightGBM → calibration → dynamic fusion/EMA/regime → confidence/risk → p ≥ 0,31 và confidence ≥ 0,45 → tối đa 2 số → quyết định paper-trade → actual results được chấm sau dự báo.

LightGBM học nhãn nhị phân **xuất hiện ít nhất một lần**, không học số nháy. Huấn luyện 365 draw kết thúc trước 90 draw validation: 45 fit sigmoid/isotonic, 45 chọn calibrator và tính trọng số fusion. Candidate universe là đủ 100 số. Thông tin ngày T như weekday biết trước và được phép sử dụng; kết quả T không được dùng. Các predictor còn lại dùng tần suất, delay, conditional, Markov, repeat, pairs, EWMA và ngày trong tuần; một số score được rescale và không tự động là xác suất đã hiệu chuẩn.

`daily_update.py` trả 99 nghìn/nháy với chi phí 27 nghìn/số. Vì vậy precision đếm **số chọn xuất hiện**, còn mean hits và ROI đếm **tất cả nháy**. Không đánh đồng hai nhãn. Backtest nhanh retrain 30 draw và thiếu state EMA/regime tương đương production, nên baseline ở đây gọi trực tiếp `run_shared_prediction_pipeline()` từng draw.

## Protocol và giới hạn độc lập

- Dataset có 7.537 dòng, 01/10/2005–29/08/2026, 27 giải mỗi dòng; kiểm tra duplicate, null, miền giá trị 0–99.
- Đánh giá **22/01–20/07/2026, 176 draw/352 picks** cho forced Top-2. Bốn ngày 16–19/02/2026 thiếu trong dataset; không tự điền kết quả. 175 kỳ cách dữ liệu cuối 1 ngày; 1 kỳ cách dữ liệu cuối 5 ngày.
- Cả hai replay dùng cùng CSV đã cắt tại 20/07, nguồn/config đóng băng, seed, 30 draw warm-up và log khởi đầu rỗng. EMA đầu đánh giá alpha=0,6, sau đủ lịch sử alpha=0,9. Đây là controlled replay, không khôi phục toàn bộ state log production cũ.
- Mỗi draw chỉ fit/feature từ prefix trước target; actual chỉ được đọc bởi scorer sau dự báo. Cache, MI edges và log thí nghiệm nằm riêng.
- Cửa sổ lịch sử này đã từng được nghiên cứu trong repository. Vì vậy nó là kiểm tra chronological ngoài mẫu theo từng draw, **không phải final evaluation độc lập để chọn/promote model**. Các báo cáo cũ tối ưu trên cùng cửa sổ không chứng minh edge độc lập.
- Holdout khóa từ **21/07/2026**, minimum 180 draw; dữ liệu hiện có khoảng 40 ngày thuộc holdout. Không chấm/tune các draw đó. Không sửa evaluation policy, prediction log hay Edge Gate. Không gộp log khác phiên bản vào cùng một kết luận holdout; bản mới cần cửa sổ prospective riêng, còn đánh giá phiên bản cũ cần nguồn đã đóng băng.
- CI95 dùng circular moving-block bootstrap 7 draw, 5.000 resample, seed 42, ghép cặp theo ngày. CI là chẩn đoán độ bất định; không hiệu chỉnh multiple comparisons và không dùng kết quả tốt nhất để promote.

## Sửa lỗi được giữ

| Lỗi | Bằng chứng/ảnh hưởng | Sửa tối thiểu |
|---|---|---|
| `int8` matmul overflow | Conditional 400 lần cho −0,279 thay vì 400/401; inverted pair 365 lần sai; NMI hai chuỗi giống nhau 0,399 thay vì 1 | Cast trước nhân ma trận trong Bayesian features, Markov fallback, inverted pairs, NMI; entropy xử lý xác suất biên |
| EMA lấy log target/future | Log chưa lọc ngày có thể làm backtest lấy trọng số không tồn tại tại T | Chỉ log hợp lệ trước T, sort theo ngày, dedup mỗi ngày |
| Cache chỉ keyed theo ngày | Lịch sử đổi hoặc snapshot có nguồn khác vẫn được dùng; phát hiện 6 snapshot legacy không khớp độ dài lịch sử hiện có | Hash lịch sử/S, chỉ reuse provenance khớp, kiểm tra target/history, namespace mới; giữ snapshot evidence cũ bất biến |
| Parquet list thành ndarray | Truth-value gây lỗi, FeatureStore bỏ extractor và mất 19 cột (79 → 60 gồm date/number) | Xử lý array/list nhất quán, lỗi extractor phải dừng rõ; `use_cache=False` bỏ cả disk cache |
| Metadata không mô tả đúng feature | Bản cache/probability đã sửa cần phân biệt với nguồn cũ | Feature v2, evidence v1.1, engine 1.3.3, khai báo đầy đủ extractor versions |

Các kiểm thử hồi quy đã quan sát fail trước sửa và pass sau sửa. Những thay đổi này được giữ vì lý do correctness, không vì một điểm số lịch sử đẹp hơn.

## So sánh forced Top-2

Forced Top-2 là chẩn đoán ranking với exposure cố định, kể cả những ngày policy thực tế SKIP.

| Method | Số xuất hiện/picks | Precision@2 | Nháy/draw | Draw có ≥1 nháy | ROI flat |
|---|---:|---:|---:|---:|---:|
| Production ban đầu (forced Top-2) | 89/352 | 25.28% | 0.540 | 46.02% | -1.04% |
| Production sau sửa correctness (forced Top-2) | 86/352 | 24.43% | 0.517 | 43.75% | -5.21% |
| Random, seed 42 | 76/352 | 21.59% | 0.494 | 36.93% | -9.38% |
| Frequency toàn lịch sử | 86/352 | 24.43% | 0.568 | 42.61% | 4.17% |
| Frequency 30 draw | 91/352 | 25.85% | 0.602 | 44.32% | 10.42% |
| Frequency 90 draw | 100/352 | 28.41% | 0.676 | 48.30% | 23.96% |
| Frequency 365 draw | 96/352 | 27.27% | 0.648 | 47.73% | 18.75% |
| Count EWMA half-life 90 | 96/352 | 27.27% | 0.648 | 46.02% | 18.75% |
| Fusion trọng số đều | 88/352 | 25.00% | 0.551 | 44.89% | 1.04% |
| Bỏ MaxDelay | 87/352 | 24.72% | 0.545 | 43.75% | 0.00% |
| Bỏ Bayesian | 83/352 | 23.58% | 0.517 | 41.48% | -5.21% |
| Bỏ Repeat | 87/352 | 24.72% | 0.534 | 44.89% | -2.08% |

Random seed 42 chỉ là một realization. Đối chứng analytic uniform tính trên số unique thực tế mỗi draw cho expected precision **23.83%**; lift sau sửa là **+0.60 điểm %**, CI95 **[-3.39; +4.90] điểm %**. Không dùng realization random yếu để tuyên bố edge.

Mean nháy/draw sau sửa trừ baseline **-0.0227**, CI95 **[-0.0625, 0.017045454545454544]**. Bỏ riêng một draw, delta precision nằm trong **[-1.14; -0.57] điểm %**; kiểm tra này không thay thế độ ổn định nhiều thời kỳ.

## Policy thực tế và exposure

| Method | Số xuất hiện/picks | Precision trên picks | Nháy/draw | Draw có ≥1 nháy | ROI flat |
|---|---:|---:|---:|---:|---:|
| Policy ban đầu | 24/98 | 24.49% | 0.159 | 13.07% | 4.76% |
| Policy sau sửa correctness | 23/94 | 24.47% | 0.153 | 12.50% | 5.32% |

Policy ban đầu active 77/176 draw, sau sửa 77/176. ROI CI95 ban đầu [-29.72; +45.83]%, sau sửa [-29.36; +46.67]%. Exposure thay đổi là hệ quả score đúng hơn dưới cùng threshold; không được xem precision/ROI của hai tập picks khác nhau là so sánh cố định exposure.

Challenger dưới đây chọn đúng số lượng picks mà policy sau sửa chọn ở từng ngày; không thay đổi cổng hoạt động của policy:

| Method | Số xuất hiện/picks | Precision trên picks | Nháy/draw | Draw có ≥1 nháy | ROI flat |
|---|---:|---:|---:|---:|---:|
| Policy sau sửa | 23/94 | 24.47% | 0.153 | 12.50% | 5.32% |
| Frequency 90 cùng exposure | 25/94 | 26.60% | 0.193 | 14.20% | 32.62% |
| Frequency 365 cùng exposure | 22/94 | 23.40% | 0.153 | 12.50% | 5.32% |
| Count EWMA cùng exposure | 19/94 | 20.21% | 0.142 | 10.80% | -2.48% |
| Fusion đều cùng exposure | 22/94 | 23.40% | 0.136 | 12.50% | -6.38% |

Phân phối nháy, recall và điểm từng ngày nằm đầy đủ trong JSON/CSV kèm theo. Không tối ưu riêng ROI hay tỷ lệ active days.

## Độ ổn định

| Giai đoạn | Draw | Baseline precision | Sau sửa precision | Delta điểm % | CI95 delta |
|---|---:|---:|---:|---:|---:|
| 2026-01-22 → 2026-03-25 | 59 | 27.97% | 26.27% | -1.69 | [-5.93; +1.69] điểm % |
| 2026-03-26 → 2026-05-23 | 59 | 22.88% | 22.88% | +0.00 | [-3.39; +3.39] điểm % |
| 2026-05-24 → 2026-07-20 | 58 | 25.00% | 24.14% | -0.86 | [-2.59; +0.00] điểm % |

Tần suất 90 draw ban đầu nhìn có triển vọng: 28.41% so với baseline 25.28%; Δ +3,125 điểm %, nhưng CI95 [-2.84; +9.09] điểm % chứa 0 và đảo chiều giữa các đoạn. Để kiểm tra giả thuyết đó, áp dụng nguyên các tín hiệu/window cố định lên **6.118 draw** 2009–2025, chia 2.166 / 2.146 / 1.806 draw, không fit lại window theo kết quả.

Mỗi ô là precision; delta điểm % so với analytic uniform expectation; CI95 delta:

| Signal | 2009–2014 | 2015–2020 | 2021–2025 |
|---|---|---|---|
| count_ewma90 | 24.88%; Δ +1.04; CI [-0.24; +2.31] điểm % | 24.49%; Δ +0.62; CI [-0.69; +1.92] điểm % | 23.92%; Δ +0.12; CI [-1.18; +1.43] điểm % |
| frequency_30 | 24.79%; Δ +0.95; CI [-0.34; +2.26] điểm % | 24.58%; Δ +0.72; CI [-0.58; +1.95] điểm % | 23.03%; Δ -0.76; CI [-2.18; +0.63] điểm % |
| frequency_365 | 24.45%; Δ +0.61; CI [-0.69; +1.91] điểm % | 22.32%; Δ -1.54; CI [-2.79; -0.28] điểm % | 23.89%; Δ +0.09; CI [-1.26; +1.53] điểm % |
| frequency_90 | 23.64%; Δ -0.20; CI [-1.45; +1.04] điểm % | 25.21%; Δ +1.34; CI [+0.04; +2.62] điểm % | 23.67%; Δ -0.13; CI [-1.52; +1.30] điểm % |
| frequency_all | 24.75%; Δ +0.91; CI [-0.25; +2.06] điểm % | 23.32%; Δ -0.54; CI [-1.81; +0.73] điểm % | 22.70%; Δ -1.10; CI [-2.42; +0.26] điểm % |

Không tín hiệu nào có CI dương ổn định trong cả ba giai đoạn. Một kết quả dương đơn lẻ sau nhiều phép so sánh không đủ làm bằng chứng predictive edge. Chẩn đoán 6.118 draw đã chạy lại từ script lưu kèm, JSON và CSV trùng byte/nội dung.

## Nhật ký quyết định thí nghiệm

Tất cả challenger sau đây dùng 176 draw, cùng 352 picks. Baseline so sánh là pipeline sau sửa correctness; các thông số được định sẵn và không tìm kiếm thêm sau khi xem kết quả.

| Hypothesis/thay đổi ranking | Baseline | Candidate | Delta điểm % | CI95 delta | Quyết định |
|---|---:|---:|---:|---:|---|
| Tần suất toàn bộ | 24.43% | 24.43% | +0.00 | [-5.68; +5.68] điểm % | Không promote |
| Tần suất 30 | 24.43% | 25.85% | +1.42 | [-4.55; +7.10] điểm % | Không promote |
| Tần suất 90 | 24.43% | 28.41% | +3.98 | [-1.70; +9.66] điểm % | Không promote |
| Tần suất 365 | 24.43% | 27.27% | +2.84 | [-2.27; +7.67] điểm % | Không promote |
| Count EWMA 90 | 24.43% | 27.27% | +2.84 | [-3.41; +8.53] điểm % | Không promote |
| Fusion đều | 24.43% | 25.00% | +0.57 | [-3.12; +4.26] điểm % | Không promote |
| Loại MaxDelay | 24.43% | 24.72% | +0.28 | [-3.69; +4.26] điểm % | Không promote |
| Loại Bayesian | 24.43% | 23.58% | -0.85 | [-5.40; +3.69] điểm % | Không promote |
| Loại Repeat | 24.43% | 24.72% | +0.28 | [-1.70; +2.27] điểm % | Không promote |

- Sửa overflow/cache/future weights: **giữ** vì correctness; kết quả và độ bất định nêu trên.
- Shrink 50% về `1−0,99^27`: Brier 0.182235 → 0.181715; ranking và precision giữ nguyên. **Không giữ** như cải tiến chọn số. Đây là prior giả định 27 vị trí độc lập đồng xác suất, không phải chứng minh cơ chế xổ số.
- Uniform probability: Brier 0.181511; tie được chọn random cùng seed. **Không dùng Brier đẹp hơn để claim accuracy tăng**.
- Candidate precision cao nhất trong bảng sau sửa: `frequency_90_top2` 28.41%, Δ +3.98 điểm %, CI95 [-1.70; +9.66] điểm %. Đây là maximum trong nhiều phép thử, không phải winner đã validated.

Không đủ bằng chứng để thêm kiến trúc, indicator hoặc tìm kiếm hyperparameter. Bước nghiên cứu nhỏ nhất hợp lý là đóng băng một candidate đơn giản và đánh giá prospective trên cửa sổ mới, tách riêng phiên bản/policy; không tiếp tục khai thác holdout cũ.

## File và kiểm chứng

Chín file production sửa: `daily_predict.py`, `src/evidence/builder.py`, `src/evidence/store.py`, `src/features/feature_store.py`, `src/features/bayesian_features.py`, `src/features/delay_features.py`, `src/probability/markov.py`, `src/probability/inverted_pairs.py`, `src/decision/risk_filters.py`. Thêm runner nghiên cứu `backtests/chronological_accuracy.py` và hai file regression tests. Không broad refactor/dependency mới.

- Full suite: **112 passed**, chạy fresh với `uv run --no-sync python -m pytest -q --basetemp .cache/pytest-accuracy-completion`.
- Hai test file bị ảnh hưởng: **20 passed**. Ruff trên runner/tests/scripts bằng chứng: pass. `git diff --check`: pass.
- Giải nén cả hai bộ forecasts/history và source ZIP, kiểm SHA256 rồi chấm lại: mọi metric khớp chính xác với evidence đã lưu. Full suite cuối chạy 43,00 giây.
- Pipeline thật tại 22/01/2026 khớp replay đã lưu với max absolute delta **0** cho ensemble/raw/calibrated LGB. Poison tất cả dữ liệu từ T trở đi và thêm 60 log tương lai: ba mảng xác suất delta **0**, bets giống nhau. Chi tiết `pipeline_verification.json`. Đây là kiểm tra tại một ngày, cộng các regression tests về prefix/cache; không tuyên bố chứng minh tuyệt đối mọi đường caller tùy ý.
- Reviewer độc lập không thấy lỗi Critical/Important. Hai nhận xét nhỏ về metadata/provenance đã sửa và kiểm thử.
- GitNexus xác nhận phạm vi tracked là 9 file, đánh dấu impact risk CRITICAL. Đây là các thành phần pipeline/cache dùng chung production và backtest; cảnh báo phạm vi đã được xem xét cùng regression tests và replay đầy đủ. GitNexus không tính các artifact/test mới chưa tracked vào chín file này.
- Các nguồn replay đóng băng trong ZIP. Bản final chỉ khác nguồn corrected replay ở metadata/version và namespace cache, không đổi phép tính; diff đầy đủ lưu trong `metadata_changes_after_source_freeze.patch`, và smoke pipeline xác nhận số khớp. SHA256 dữ liệu, cấu hình, nguồn và forecasts nằm trong summaries/manifest.
- Bare pytest ban đầu vướng thư mục temp Windows; dùng `--basetemp` trong workspace để chạy toàn bộ suite. Không che lỗi test bằng skip.
- Dataset hiện tại là một snapshot lịch sử; không có dữ liệu publication-time revision để chứng minh mọi bản ghi đã giống hệt dữ liệu công bố tại từng thời điểm.

## Tái lập và artifacts

Artifacts ở `backtests/results/accuracy_audit_2026-10-03/`: summaries, daily scores, paired comparison, stability, source ZIP, forecast JSONL gzip, history CSV gzip, verification và SHA256. Baseline nguồn ban đầu tại commit `47489b43` nhưng source ZIP/hash mới là bằng chứng chính xác của run, không chỉ tên commit.

Chạy lại numerical replay từ frozen source đã giải nén ở `.cache/recheck-original/source` hoặc `.cache/recheck-corrected/source`; giải nén `history.csv.gz` thành `source/data/xsmb-2-digits.csv` cho cả hai, dùng thư mục output mới:

```powershell
uv run --no-sync python backtests/chronological_accuracy.py --source-root .cache/recheck-original/source --start 2026-01-22 --end 2026-07-20 --output .cache/recheck-original/run --label original
uv run --no-sync python backtests/chronological_accuracy.py --source-root .cache/recheck-corrected/source --start 2026-01-22 --end 2026-07-20 --output .cache/recheck-corrected/run --label corrected
uv run --no-sync python backtests/results/accuracy_audit_2026-10-03/compare_replays.py --original .cache/recheck-original/run --corrected .cache/recheck-corrected/run --output .cache/recheck-comparison.json
uv run --no-sync python backtests/results/accuracy_audit_2026-10-03/reproduce_stability.py --output .cache/recheck-stability
```

Replay có thể resume với `--resume` chỉ khi nguồn/config/data giống và forecast là prefix liên tục; không đọc kết quả tương lai để chọn state. `--score-only` chấm lại forecasts đã đủ; các files forecast/history nén kèm cho phép kiểm tra kết quả mà không train lại.
