> [!WARNING]
> **SUPERSEDED bởi pre-registration R2** (`prereg_r2/`). Các mục 4 (kết luận về cỡ mẫu/power chung 63,5% mô tả như thiết kế chuẩn) và 5 (gate G1-G3) KHÔNG còn là protocol xác nhận. Rule dự báo (mục 1) và hash (mục 2) vẫn hiệu lực và không đổi. Nội dung gốc giữ nguyên bên dưới làm bằng chứng.

# Pre-registration — XSMB Frozen Prospective Challenger V1

**Phiên bản:** `frozen-challenger-v1.0.0` · **Đóng băng:** 01/11/2026 · **Trạng thái:** `FROZEN_PROSPECTIVE_NOT_PRODUCTION`

Nguồn sự thật máy đọc được: `spec.json` + `manifest.json` (cùng thư mục). Tài liệu này chỉ diễn giải. Production, `evaluation_policy.json` và prediction log **không bị đụng**.

## 1. Rule đóng băng (không còn tham số tự do)

Với ngày dự báo T, chỉ dùng các draw hợp lệ có ngày < T:

```text
f[n]     = số nháy của số n trong 90 draw hợp lệ liền trước T
H[h]     = Σ f[n] trên 10 số cùng chữ số đầu h      L[l] = Σ f[n] trên 10 số cùng chữ số đuôi l
score[n] = z(H[n//10]) + z(L[n%10]) + z(f[n])        z = chuẩn hóa chéo 100 số/ngày, sd tổng thể (sd=0 → 0)
picks    = top-2 theo score (làm tròn 12 chữ số); đồng điểm → số nhỏ hơn trước
```

Dòng không đủ đúng 27 giá trị trong 0..99 bị bỏ hoàn toàn (không là lịch sử, không được chấm). Cần ≥ 90 draw lịch sử, nếu không báo lỗi. Rule này tương đương chính xác quy tắc H2 trong `head_signal_followup.py` (test khẳng định, sai số ≤ 1e-9).

## 2. Hash

| Mục | SHA-256 |
|---|---|
| `spec.json` (JSON chuẩn hóa) | `5388d4481b5234b55657ba3152fe00f73416f22ad04eda21e05fe45593aa51c5` |
| `backtests/frozen_challenger_v1.py` (CRLF chuẩn hóa thành LF) | `3b869f4edacca8b39531675b805e0912fe010c930e696f9799879513a65e0a65` |
| 7.497 dòng CSV trước 21/07/2026 | `c31e4ed799f113912f42fe894f14b124574dc05545623658edcbdf3ab0544bcd` |

`python backtests/frozen_challenger_v1.py verify` phải in `VERIFY OK`; `score-holdout` tự từ chối nếu hash lệch. **Nên commit/tag git ngay** để có mốc thời gian bên ngoài.

## 3. Metric chính và uplift tối thiểu có giá trị

- **Primary:** IC Spearman trung bình theo ngày giữa `score` và số nháy từng số (100 số/ngày). Chọn vì là phép đo toàn bộ bảng xếp hạng, công suất cao nhất khi chỉ có 1 draw/ngày.
- **Secondary (sản phẩm):** excess top-2 = nháy TB của 2 số được chọn − nháy TB cả 100 số cùng ngày. Break-even = 27/99 − 0,27 = **+0,00273** nháy/số.
- **Minimum worthwhile uplift:** excess top-2 ≥ **+0,0050** nháy/số (ROI flat tương đương ≈ **+0,83%**, khoảng 1,8× break-even) ⇔ IC ≈ **0,00567** (hệ số excess/IC κ = 0,882 đo trong discovery). Nhỏ hơn mức này không đáng đổi production/paper-trade.

## 4. Cỡ mẫu (tính TRƯỚC khi mở holdout, chỉ từ dữ liệu trước 21/07/2026)

σ(IC theo ngày) = 0,0999; α một phía = 0,025; power 0,80; δ = 0,00567:

`N = ((1,96 + 0,84) · 0,0999 / 0,00567)² = 2.439` → ×1,05 dự phòng draw lỗi → **N_target = 2.561 draw hợp lệ ≈ 7,0 năm**, sớm nhất khoảng **25/07/2033**.

| N draw | MDE của IC (power 80%) | MDE của excess top-2 |
|---:|---:|---:|
| 180 (sàn tối thiểu của policy) | 0,0209 | 0,0783 |
| 365 | 0,0147 | 0,0550 |
| 730 | 0,0104 | 0,0389 |
| 1.460 | 0,0073 | 0,0275 |
| **2.561** | **0,0055** | 0,0208 |

Công suất chung của cả ba cổng ở N_target ≈ **63,5%** (không phải 80%, vì G2/G3 chỉ cho phép khi cả IC và excess đều thuận).

> [!WARNING]
> Ước lượng IC discovery (0,0053) nằm sát δ và được chọn **sau khi nhìn cả ba giai đoạn** nên có winner's curse; công suất thực tế thấp hơn 63,5%. Cũng lưu ý IC của chính rule này ở VAL 2016–2020 chỉ 0,0013 (DEV 0,0064, FINAL 0,0071), tức hiệu ứng không đều giữa các giai đoạn. Các số discovery là in-sample, **không phải bằng chứng**.

Ghi chú sửa lỗi: báo cáo trước ghi "≥ 700 draw để SE excess ≈ 0,004" là sai (sd excess top-2 theo ngày ≈ 0,375 ⇒ SE = 0,375/√N); đã sửa trong `signal_power_study_2026-11-01.md`.

## 5. Cổng xác nhận (confirmatory gate) — chấm một lần

Cửa sổ chấm: **N_target draw hợp lệ đầu tiên có ngày ≥ 21/07/2026**; không chấm giữa chừng, không dừng sớm. Tất cả phải đúng:

| Cổng | Điều kiện |
|---|---|
| G1 | Cận dưới một phía 97,5% (block bootstrap, block 7, 10.000 lần, seed 42) của IC trung bình > 0 |
| G2 | IC trung bình ≥ 0,00309 (IC break-even = 0,00273/κ) |
| G3 | Excess top-2 trung bình > 0 |

- **PASS:** mới được cân nhắc một so sánh paper-trade riêng với policy production. Spec này không tự đổi production.
- **FAIL:** loại bỏ challenger, không tinh chỉnh trên holdout; rule mới = phiên bản mới + dữ liệu mới.

`score-holdout` kiểm hash → từ chối nếu kết quả đã tồn tại → chỉ **đếm cột ngày** để kiểm tra đủ N_target (chưa đủ thì dừng, không parse giá trị) → chấm → ghi `holdout_result.json` bằng chế độ `'x'` (không ghi đè được).

## 6. Không đọc holdout trong giai đoạn đóng băng

- `read_prefix()` stream CSV theo dòng; với dòng có ngày ≥ 21/07/2026 chỉ phân tích cột ngày (test chèn giá trị rác vào dòng holdout để chứng minh không bị parse). Hiện có 40 dòng holdout; chưa dòng nào được đọc giá trị hay chấm.
- Mọi số discovery/MDE/N chỉ dùng 7.497 dòng trước holdout.
- Minh bạch: các script nghiên cứu trước đó (`signal_power_study.py`) nạp cả file CSV vào bộ nhớ rồi mới cắt theo ngày; giá trị holdout không được dùng/hiển thị, nhưng module mới dùng cơ chế stream chặt hơn.
- Dự báo prospective cho ngày T ≥ 21/07/2026 buộc phải dùng giá trị các ngày trước đó làm lịch sử (đó là bản chất của rule, không phải tuning); `forecast --date` hỗ trợ việc này nhưng không được dùng để chấm trước cổng.

## 7. Điều cần quyết định (ngoài phạm vi tôi tự quyết)

N_target ≈ 7 năm với 1 draw/ngày. Nếu muốn rút ngắn mà không hạ chuẩn, cách khả thi duy nhất là **tái lập giả thuyết cấu trúc trên dữ liệu độc lập** (ví dụ XSMN/XSMT nếu có nguồn), vì thêm draw độc lập mới làm tăng công suất; hạ δ hoặc nới α chỉ làm kết luận kém tin cậy hơn.
