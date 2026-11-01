# Pre-registration R2 — XSMB Frozen Challenger V1 (corrective)

**Revision:** `frozen-challenger-v1-prereg-r2` · **Đóng băng:** 01/11/2026 · **Thay thế:** gate G1–G3 và phần mô tả cỡ mẫu của `frozen-challenger-v1.0.0`
**Không đổi:** prediction rule, production policy, Top-K, holdout. Chưa push/tag/promote.

Nguồn sự thật máy đọc được: `spec.json`, `manifest.json`, `power_audit.json` (cùng thư mục).

## 1. Điều gì đã sai ở bản v1 và đã sửa thế nào

| Vấn đề (Control Plane) | Xử lý trong R2 |
|---|---|
| A. Joint power G1–G3 chỉ 63,5%, không phải 80% | Power 80% nay chỉ gắn với **S** (một kiểm định); joint power của gate cũ ghi rõ là **63,5% và đã bị thay thế** |
| B. G3 chỉ cần excess > 0 nhưng break-even là +0,00273: có thể PASS mà ROI kỳ vọng âm | Tách **S** (khoa học) và **E** (kinh tế); E so cận dưới với **break-even**, không phải 0 |

Rule dự báo không đổi: `score[n] = z(H[n//10]) + z(L[n%10]) + z(f[n])`, `f` = nháy trong 90 draw hợp lệ liền trước T, top-2, đồng điểm → số nhỏ trước. Bằng chứng không đổi: code v1 hash `3b869f44…0a65`, spec v1 hash `5388d448…1c5`, và **fingerprint của toàn bộ picks** trên 7.407 ngày trước holdout `43b1b916…d5ad` (R2 ghim cả ba).

## 2. Kiểm tra độc lập công thức (kết quả `power_audit.json`, 7/7 kiểm tra đạt)

σ(IC theo ngày) = 0,0999; σ(excess top-2 theo ngày) = 0,375; N = 2.561; κ = excess/IC = 0,882.

| Đại lượng | Closed-form | Kiểm tra độc lập |
|---|---:|---|
| SE của IC tại 2.561 | 0,001974 | — |
| SE của excess top-2 tại 2.561 | **0,00741** (không phải 0,004) | — |
| N cho power S 80% | 2.439 (chuẩn) | **2.441** (t-Student df = n−1, tìm tuần tự) |
| Power S tại IC = δ (0,00567) | **81,9%** | Monte Carlo gate thật: **82,4%** |
| Sai số loại I của S (IC thật = 0) | 2,5% | MC: 2,6% |
| Joint power G1–G3 cũ | 63,5% | MC: **63,4%** (xác nhận con số đó là đúng) |
| E1 tại biên (excess thật = break-even) | 2,5% | MC: 2,0% |

Monte Carlo: 500 mô phỏng × bootstrap 500, resample ngày thật trước holdout, chèn hiệu ứng thật, chạy đúng hàm gate.

## 3. Hai kết luận tách biệt

| | Điều kiện (chấm một lần, N_target = 2.561 draw hợp lệ đầu tiên ≥ 21/07/2026) |
|---|---|
| **S — Scientific confirmation** | Cận dưới một phía 97,5% (block bootstrap, block 7, 10.000 lần, seed 42) của IC trung bình **> 0** |
| **E — Economic qualification** | S đạt **và** cận dưới một phía 97,5% của excess top-2 trung bình **> break-even = +0,002727** |

Ba kết cục duy nhất:

1. `NOT_CONFIRMED` → bỏ challenger, không tinh chỉnh trên holdout.
2. `SCIENTIFIC_CONFIRMED_NOT_ECONOMICALLY_QUALIFIED` → chỉ lưu bằng chứng; **không đổi production/paper-trade**. S **không** tự dẫn tới E. "Chưa đủ điều kiện kinh tế" ≠ "đã chứng minh lỗ".
3. `SCIENTIFIC_CONFIRMED_ECONOMIC_QUALIFIED` → Control Plane mới được cân nhắc một so sánh paper-trade riêng; spec này không tự đổi production.

Kết quả còn được báo kèm (không phải gate): ROI flat tương đương của excess trung bình và của cận dưới, và excess − break-even.

## 4. Power thực tế của từng kết luận tại N = 2.561

| IC thật | 0,0020 | 0,0030 | 0,0040 | 0,0045 | 0,0053 | **0,0057 (MWU)** | 0,0075 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Power S | 17% | 33% | 53% | 63% | 77% | **82%** | 97% |

| Excess top-2 thật | **0,0050 (MWU)** | 0,0066 (discovery) | 0,0100 | 0,0200 | 0,0235 | 0,0300 |
|---|---:|---:|---:|---:|---:|---:|
| Power E | **4,9%** | 7,5% | 16% | 64% | 80% | 96% |

> [!IMPORTANT]
> **E gần như không thể đạt.** Để E có power 80% tại mức MWU (+0,0050) cần ≈ **213.700 draw (≈ 585 năm)**; tại excess discovery (+0,0066) cần ≈ 73.600 draw (≈ 202 năm). Tại N = 2.561, E chỉ qualified (80%) nếu excess thật ≥ **+0,0235** (ROI flat ≈ +9,4%), lớn gấp ~4× mọi giá trị từng quan sát. Kết cục thực tế khả dĩ nhất nếu hiệu ứng có thật nhưng nhỏ là kết cục số 2.

> [!WARNING]
> Ước lượng discovery nằm sát MWU và được chọn **sau khi nhìn cả ba giai đoạn** (winner's curse). Nếu hiệu ứng thật chỉ bằng 0,7× MWU, power S giảm còn ≈ 54%; bằng 0,5× còn ≈ 33% (MC). IC của chính rule này ở VAL 2016–2020 chỉ 0,0013 (DEV 0,0064; FINAL 0,0071). 2.561 kỳ là **mốc nghiên cứu để lưu giữ bằng chứng**, không phải thiết kế bảo đảm 80% cho kết luận kinh tế.

## 5. Dùng dữ liệu các kỳ holdout trước T khi dự báo T

Được phép: kết quả các kỳ có ngày **< T** làm lịch sử. Cấm: kết quả T và mọi ngày sau T; chấm/nhìn trước khi đủ N_target. Cơ chế cưỡng chế, đều có test:

- `forecast_from_lines` lọc dòng theo `date < T` **trước khi parse**; poison toàn bộ T và tương lai (kể cả xóa hẳn) → picks không đổi.
- `scores_all` hàng t chỉ dùng `c[max(0,t−90):t]`; `score_window` (đường chấm thật) trùng picks của `forecast_from_lines` từng ngày, kể cả khi có dòng không hợp lệ rải rác.
- Tính hợp lệ của dòng tương lai (hỏng/thiếu) không thể đổi score của T (cửa sổ = 90 dòng hợp lệ liền trước T); hợp lệ chỉ xét định dạng (27 giá trị trong 0..99), không xét kết quả.
- Trước N_target, `score-holdout` chỉ **đếm cột ngày**; chưa parse giá trị. Hiện 40/2.561 dòng; không dòng holdout nào được chấm.
- `holdout_result.json` ghi bằng chế độ `x` và **dùng chung với v1**: chỉ một lần chấm cho mọi revision. Không chạy `frozen_challenger_v1.py score-holdout` (đã SUPERSEDED).

## 6. Hash R2

| Mục | SHA-256 |
|---|---|
| `spec.json` R2 (JSON chuẩn hóa) | `94080dbcd7195d35a3184faa1f029a08dde69edca395d38dc91033da514e6067` |
| `backtests/frozen_challenger_v1_prereg_r2.py` (CRLF→LF) | `01c8c1bcf889c9007986f385aeb65fea5b2693866c2b464dbe6fc6c36dfcec66` |
| `power_audit.json` | `c9e7f10829e00698fb027c4d2f19a894e45db3b00bf85ae451b6da1a1f16492c` |
| Fingerprint picks của rule trên dữ liệu trước holdout | `43b1b9162a4534b83701816e483ba8a88d158596b04d73a8e4d40415120cd5ad` |
| Dữ liệu CSV trước 21/07/2026 (7.497 dòng) | `c31e4ed799f113912f42fe894f14b124574dc05545623658edcbdf3ab0544bcd` |
| v1: code / spec | `3b869f4e…0a65` / `5388d448…1c5` (không đổi) |

Kiểm tra: `python backtests/frozen_challenger_v1_prereg_r2.py verify` → `VERIFY R2 OK` (kiểm cả v1).

## 7. Định hướng lưu giữ

Không theo đuổi như nhiệm vụ phát triển thường xuyên. Production và paper-trade tiếp tục theo policy hiện tại; không đổi rule vì kết quả ngắn hạn. Chỉ chấm khi đủ dữ liệu, bằng đúng `score-holdout` của R2.

## 8. Điểm vào chính thức duy nhất và giới hạn one-shot (công khai)

- **Điểm vào chính thức duy nhất để chấm holdout:** `python backtests/frozen_challenger_v1_prereg_r2.py score-holdout`. Mọi cách chấm khác không hợp lệ.
- **Giới hạn đã biết:** `python backtests/frozen_challenger_v1.py score-holdout` (v1, đã SUPERSEDED) vẫn chạy được về mặt kỹ thuật. Không thể khóa đường này mà không đổi byte đã đóng băng: v1 chỉ dựa vào `verify()` và file kết quả, còn R2 ghim hash của v1 và gọi lại `v1.verify()`; đổi tên/sửa manifest, tạo file kết quả giả hoặc sửa v1 đều hoặc phá hash, hoặc làm R2 tự từ chối.
- Hệ quả: v1 chỉ tự từ chối cho đến khi có đủ 2.561 dòng sau holdout. Sau thời điểm đó, nếu ai chạy nhầm v1, file `holdout_result.json` dùng chung bị tiêu thụ bằng gate cũ và R2 sẽ từ chối. **Vì vậy one-shot chỉ được bảo đảm bằng kỷ luật vận hành (và lệnh trên), không được tuyên bố là đảm bảo toàn cục bằng kỹ thuật.**
- Giảm thiểu không đổi byte đóng băng: `SUPERSEDED.md` cùng thư mục và chú thích ở đầu `PREREGISTRATION.md` (v1). `spec.json` R2 đã đóng băng nên không chứa đoạn này; tài liệu này được cập nhật sau freeze và không nằm trong manifest.
- Nếu muốn khóa bằng kỹ thuật, cần một revision mới (R3) sửa `score-holdout` để từ chối khi thấy tombstone, hoặc một bản v1 mới; đây là quyết định của Control Plane.
