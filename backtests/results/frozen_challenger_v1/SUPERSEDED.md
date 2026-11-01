# frozen-challenger-v1.0.0 gate: SUPERSEDED

Gate G1-G3 và mô tả cỡ mẫu của v1 được thay bằng `prereg_r2/` (01/11/2026). Lý do: G3 chỉ yêu cầu excess > 0 trong khi break-even là +0.00273; joint power 63.5% không phải 80%.

Artifacts v1 (spec.json, manifest.json, `backtests/frozen_challenger_v1.py`) giữ NGUYÊN và vẫn verify; R2 ghim hash của chúng. `holdout_result.json` dùng chung: chỉ một lần chấm cho mọi revision. KHÔNG chạy `frozen_challenger_v1.py score-holdout`; dùng `frozen_challenger_v1_prereg_r2.py score-holdout`.

**Điểm vào chính thức duy nhất để chấm holdout:** `backtests/frozen_challenger_v1_prereg_r2.py score-holdout`. Lệnh v1 vẫn chạy được về kỹ thuật (không thể khóa mà không đổi byte đóng băng); one-shot chỉ được bảo đảm bằng kỷ luật vận hành. Chi tiết: `prereg_r2/PREREGISTRATION_R2.md` mục 8.
