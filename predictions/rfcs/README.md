# Nhật ký Quản trị Kiến trúc XPIS (RFC & ADR Registry)

Mục lục này lưu giữ vết toàn bộ các thay đổi kiến trúc chính thức được đề xuất (RFC) và các quyết định đã ký duyệt (ADR) tuân thủ theo nguyên lý **P4 — Registry Driven** và **P6 — Scientific Validation** của XPIS v1.2.

---

## 1. Nhật ký Quyết định Kiến trúc (ADR - Architecture Decision Records)

* **[ADR-001: Mutual Information Empirical Correlation](ADR-001_mutual_information.md)** — APPROVED
  - **Mô tả**: Thay thế Pearson Correlation bằng Normalized Mutual Information (NMI) để bắt tương quan nhị phân phi tuyến.
  - **Checksum**: `predictions/mi_edges.json` (Edge List).
* **[ADR-002: Dynamic Rolling Weights Fusion](ADR-002_dynamic_rolling_weights.md)** — APPROVED
  - **Mô tả**: Thiết lập trọng số Meta Fusion động (Rolling EWMA Quality Score) từ Brier, LogLoss và Precision.
* **[ADR-003: Portfolio Risk Filters & Diversification](ADR-003_portfolio_risk_filters.md)** — APPROVED
  - **Mô tả**: Áp dụng bộ lọc đa dạng hóa danh mục cược loto và giới hạn Exposure Limit 20% đầu/đuôi.

---

## 2. Nhật ký Đề xuất Thay đổi (RFC - Request For Comments)

| Mã RFC | Tiêu đề | Trạng thái | Ghi chú |
|---|---|:---:|---|
| **RFC-001** | Tích hợp ma trận tương quan phi tuyến NMI | **Approved** | Chuyển thành ADR-001 |
| **RFC-002** | Tích hợp hệ trọng số động EWMA cho Layer 5 | **Approved** | Chuyển thành ADR-002 |
| **RFC-003** | Sử dụng mô hình Deep Learning (Transformer/LSTM) | **Rejected** | Bị bác bỏ do cỡ mẫu dữ liệu nhỏ ($N \approx 7000$), rủi ro Overfitting cực cao và tốn tài nguyên tính toán |
| **RFC-004** | Tách biệt Prediction và Decision Contract ở Layer 6 | **Approved** | Đóng băng trong kiến trúc v1.2 |
