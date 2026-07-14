# Research Layer — XPIS v1.0

## Vai trò

Layer này **không tham gia Prediction**. Nó chỉ phục vụ việc nghiên cứu, kiểm chứng và phê duyệt ý tưởng mới trước khi đưa vào Production.

## Quy trình bắt buộc

```
Ý tưởng (Idea)
    │
    ▼
Hypothesis (Giả thuyết rõ ràng, có thể đo lường)
    │
    ▼
Backtest (trên dữ liệu out-of-sample, chưa thấy)
    │
    ▼
Validation (cross-check trên nhiều window thời gian)
    │
    ├── PASS → Đề xuất đưa vào Feature hoặc Model Layer
    │
    └── FAIL → Đóng. Ghi lại lý do. Không thử lại nếu không có lý do mới.
```

## Nguyên tắc

1. **Không backtest trên toàn bộ data** — luôn giữ ít nhất 20% data cuối cho final validation.
2. **Mọi ý tưởng phải có hypothesis đo được** — không nghiên cứu ý tưởng mơ hồ.
3. **Ghi lại mọi kết quả** — kể cả kết quả FAIL (để tránh lặp lại sau này).
4. **Không mang code thực nghiệm vào Production** — phải refactor và review trước.
5. **Mỗi feature mới phải được test isolated** — không thêm nhiều thứ cùng lúc.

## Cấu trúc thư mục Research

```
src/research/
    README.md           # File này
    experiments/        # Mỗi experiment có 1 folder
        EXP-001-delay-zscore/
            hypothesis.md
            notebook.py
            results.json
        EXP-002-markov-order3/
            hypothesis.md
            notebook.py
            results.json
```

## Template Experiment

### hypothesis.md
```markdown
# EXP-00X: [Tên ý tưởng]
**Ngày**: YYYY-MM-DD
**Người đề xuất**: [Tên]

## Hypothesis
[Giả thuyết rõ ràng, ví dụ: "Delay zscore > 2.0 sẽ tăng hit rate lên > 35%"]

## Cách đo lường
[Metric cụ thể, ví dụ: "Hit rate trong top-10 trên 365 ngày cuối"]

## Ngưỡng PASS
[Ví dụ: "Hit rate > 35% VÀ ROI > 0"]

## Kết quả
[ ] PASS / [ ] FAIL

## Lý do (nếu FAIL)
[...]
```

## Danh sách Experiments

| ID | Tên | Status | Ngày | Kết quả |
|---|---|---|---|---|
| — | Chưa có experiment nào | — | — | — |
