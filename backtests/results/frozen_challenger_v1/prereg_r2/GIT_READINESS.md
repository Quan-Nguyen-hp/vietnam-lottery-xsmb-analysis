# Git readiness / closeout report — Frozen Challenger V1 / R2 (01/11/2026)

Không push, không merge, không tag, không promote. Quyền tag/push thuộc Control Plane.

## Lịch sử commit (nhánh `develop`, chưa có upstream)

| # | Nội dung | SHA | Tree |
|---|---|---|---|
| base | `47489b435a2b153e8cedc8706d78dcdcd0894500` | (Merge PR #1) | `183c7c99093eeea0f2ffccf47016e156621bdf29` |
| 1 | `fix: correct temporal and overflow defects in prediction pipeline` (33 file) | `4e4e218b71f66499729eee21958789cde9e29183` | `0bda1291d3c99ea27190d560ef0f8ca5d8014d5b` |
| 2 | research + frozen challenger V1/R2 + dependency | xem `git log -2` (một commit không thể chứa SHA của chính nó) | |

Commit 1 chỉ chứa các sửa correctness production, test hồi quy và bằng chứng audit 03/10 (replay kết thúc 20/07/2026, trước holdout). Commit 2 chỉ chứa nghiên cứu 01/11, frozen challenger v1/R2 và khai báo dependency.

## Dependency

`scipy` được code đã commit dùng trực tiếp từ trước (`src/meta/constrained_stacking.py`, `src/research/feature_evaluator.py`, `backtests/model_slimming.py`) và nay cả harness frozen. Giải quyết nhỏ nhất: nhóm `[dependency-groups] research = ["scipy>=1.14"]` trong `pyproject.toml` (+3 dòng) và `uv.lock` (+4 dòng, phiên bản scipy vẫn 1.18.0, không nâng package nào). Không thêm vào `dependencies` runtime của production.

## Hash bất biến (xác minh lại ngay trước commit 2: `VERIFY OK`, `VERIFY R2 OK`)

| Artifact | SHA-256 |
|---|---|
| v1 code `backtests/frozen_challenger_v1.py` | `3b869f4edacca8b39531675b805e0912fe010c930e696f9799879513a65e0a65` |
| v1 spec | `5388d4481b5234b55657ba3152fe00f73416f22ad04eda21e05fe45593aa51c5` |
| Dữ liệu CSV trước 21/07/2026 (7.497 dòng) | `c31e4ed799f113912f42fe894f14b124574dc05545623658edcbdf3ab0544bcd` |
| R2 spec | `94080dbcd7195d35a3184faa1f029a08dde69edca395d38dc91033da514e6067` |
| R2 code `backtests/frozen_challenger_v1_prereg_r2.py` | `01c8c1bcf889c9007986f385aeb65fea5b2693866c2b464dbe6fc6c36dfcec66` |
| R2 power audit | `c9e7f10829e00698fb027c4d2f19a894e45db3b00bf85ae451b6da1a1f16492c` |
| Fingerprint picks của rule (7.407 ngày trước holdout) | `43b1b9162a4534b83701816e483ba8a88d158596b04d73a8e4d40415120cd5ad` |

Sửa bất kỳ file nào đã hash phải tạo revision mới. Hash code được tính sau chuẩn hóa CRLF→LF.

## Giới hạn one-shot

Điểm vào chính thức duy nhất để chấm holdout là `frozen_challenger_v1_prereg_r2.py score-holdout`. Lệnh `score-holdout` của v1 (đã SUPERSEDED) vẫn chạy được về kỹ thuật và không thể khóa mà không đổi byte đã đóng băng; one-shot chỉ được bảo đảm bằng kỷ luật vận hành, không phải bảo đảm toàn cục bằng kỹ thuật. Chi tiết: `PREREGISTRATION_R2.md` mục 8.

## Phạm vi loại trừ (WIP không liên quan, giữ nguyên ở dạng untracked)

`dev_evidence_rematerialization_daily_scores.csv`, `dev_evidence_rematerialization_summary.json`, `evidence_manifest.json`, `val_daily_scores.csv`, `val_forecast_bootstrap_evidence.json`, `val_summary.json`, và hai file `docs/superpowers/specs/XPIS v3 — M3 …`.

## Trạng thái

- `holdout_result.json` không tồn tại; `score-holdout` (R2) từ chối: 40/2.561 dòng; không dòng holdout nào được parse/chấm.
- `predictions/` và `data/` không thay đổi.
- Test: xem báo cáo closeout (full suite chạy sau mỗi commit).