"""
backtest_matrix_1year.py — Backtest 1 năm (365 ngày) thuật toán Ma trận Quyết định (Matrix Decision)
Chọn 10 số/ngày | Chi phí 270k/ngày | Trúng 99k/nháy
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.methods.matrix_decision import MatrixDecisionPredictor

COST_PER_NUM   = 27
PAYOUT_PER_HIT = 99
N_TEST_DAYS    = 365
TOP_K          = 10

RULES_FILE = root_dir / "predictions" / "matrix_rules.json"

def calculate_mdd(pnl_list):
    """Tính toán Maximum Drawdown (mức sụt giảm vốn lớn nhất)"""
    cum_pnl = np.cumsum(pnl_list)
    running_max = np.maximum.accumulate(cum_pnl)
    # Nếu running_max âm ở mọi điểm, đặt baseline là 0
    running_max = np.maximum(running_max, 0)
    drawdowns = running_max - cum_pnl
    return float(np.max(drawdowns))

def main():
    # ── 1. Load Data ────────────────────────────────────────────────────────
    csv_path = root_dir / "data" / "xsmb-2-digits.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    total_days = len(df)
    prize_cols = [c for c in df.columns if c != "date"]

    # Chuyển đổi ma trận nhị phân sang np.int8 để tối ưu RAM
    arr_full = df[prize_cols].values.astype(int)
    S_full = np.zeros((total_days, 100), dtype=np.int8)
    rows_f = np.repeat(np.arange(total_days), arr_full.shape[1])
    cols_f = arr_full.flatten()
    valid  = (cols_f >= 0) & (cols_f < 100)
    S_full[rows_f[valid], cols_f[valid]] = 1

    start_idx = total_days - N_TEST_DAYS
    test_dates = list(df["date"].iloc[start_idx:])

    # ── 2. Load Rules ───────────────────────────────────────────────────────
    predictor = MatrixDecisionPredictor(rules_path=str(RULES_FILE))
    print(f"\n⚙️  Quy tắc lọc sử dụng:")
    for k, v in predictor.rules.items():
        print(f"    {k:<15}: {v}")

    # ── 3. Chạy Walk-forward Simulation ──────────────────────────────────────
    print(f"\n🔄 Đang chạy Backtest {N_TEST_DAYS} ngày...")
    
    pnl_list = []
    hits_list = []
    chosen_picks = []
    actual_results_list = []

    for step, idx in enumerate(range(start_idx, total_days)):
        S_hist = S_full[:idx]
        
        # Lấy date cực nhẹ để DayOfWeekPredictor lấy thông tin thứ trong tuần
        history_df = df[["date"]].iloc[:idx]
        picks = predictor.predict(history_df, top_k=TOP_K, S=S_hist)
        chosen_picks.append(picks)

        # Tính kết quả
        actual_row = df.iloc[idx]
        actual_lotos = actual_row[prize_cols].values.astype(int)
        actual_counts = {}
        for val in actual_lotos:
            if 0 <= val < 100:
                actual_counts[val] = actual_counts.get(val, 0) + 1
        actual_results_list.append(list(actual_counts.keys()))

        cost = TOP_K * COST_PER_NUM
        hits = sum(actual_counts.get(n, 0) for n in picks)
        rev  = hits * PAYOUT_PER_HIT
        pnl  = rev - cost

        pnl_list.append(pnl)
        hits_list.append(hits)

        if (step + 1) % 100 == 0:
            print(f"  Tiến độ: {step + 1}/{N_TEST_DAYS} ngày...")

    pnl_arr = np.array(pnl_list)
    hits_arr = np.array(hits_list)

    # ── 4. Thống kê Hiệu suất ───────────────────────────────────────────────
    total_cost = N_TEST_DAYS * TOP_K * COST_PER_NUM
    total_pnl  = pnl_arr.sum()
    roi        = total_pnl / total_cost * 100
    win_days   = np.sum(hits_arr > 0)
    win_rate   = win_days / N_TEST_DAYS * 100
    max_dd     = calculate_mdd(pnl_list)

    # Thống kê số nháy trúng tối đa/trung bình
    avg_hits = np.mean(hits_arr)
    max_hits = np.max(hits_arr)

    # Thống kê phân phối theo tháng
    df_results = pd.DataFrame({
        "date": test_dates,
        "pnl": pnl_list,
        "hits": hits_list
    })
    df_results["ym"] = df_results["date"].dt.to_period("M")
    monthly = df_results.groupby("ym").agg(
        total_pnl=("pnl", "sum"),
        days=("pnl", "count"),
        win_days=("hits", lambda x: np.sum(x > 0))
    ).reset_index()

    # ── 5. In kết quả ra Terminal ──────────────────────────────────────────
    SEP = "=" * 70
    print(SEP)
    print(f"📈  KẾT QUẢ BACKTEST 1 NĂM — MATRIX DECISION (top_k={TOP_K})")
    print(SEP)
    print(f"  Khoảng thời gian: {test_dates[0].strftime('%d/%m/%Y')} → {test_dates[-1].strftime('%d/%m/%Y')}")
    print(f"  Số ngày giao dịch: {N_TEST_DAYS} ngày")
    print(f"  Vốn đầu tư/ngày : {TOP_K * COST_PER_NUM:,}đ  (10 số)")
    print(f"  Tổng vốn bỏ ra  : {total_cost * 1000:,}đ")
    print(f"  Lãi/Lỗ ròng     : {total_pnl * 1000:+,}đ")
    print(f"  Tỷ lệ ROI       : {roi:+.2f}%")
    print(f"  Tỷ lệ thắng ngày: {win_rate:.1f}% ({win_days}/{N_TEST_DAYS} ngày trúng)")
    print(f"  Số nháy trúng TB: {avg_hits:.2f} nháy/ngày (Kỳ vọng về: {27*TOP_K/100:.2f} nháy)")
    print(f"  Trúng nhiều nhất: {max_hits} nháy/ngày")
    print(f"  Maximum Drawdown: {max_dd * 1000:,}đ (Vốn dự phòng tối thiểu)")
    print(SEP)
    print("📅  CHI TIẾT LỢI NHUẬN THEO THÁNG:")
    print(f"  {'Tháng':<8}  {'Số ngày':>8}  {'Ngày thắng':>10}  {'Lợi nhuận (đ)':>15}  {'ROI':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*15}  {'-'*8}")
    for _, row in monthly.iterrows():
        p = row["total_pnl"]
        c = row["days"] * TOP_K * COST_PER_NUM
        r = p / c * 100
        print(f"  {str(row['ym']):<8}  {row['days']:>8}  {row['win_days']:>10}  {p*1000:>+14,.0f}đ  {r:>+7.1f}%")
    print(SEP)

    # ── 6. Ghi báo cáo ra file Markdown ──────────────────────────────────────
    report_path = root_dir / "backtests" / "results" / "matrix_decision_1year_backtest.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Báo cáo Backtest 1 Năm — Thuật toán Ma trận Quyết định (top_k=10)\n\n")
        f.write(f"- **Kỳ kiểm thử**: {test_dates[0].strftime('%d/%m/%Y')} đến {test_dates[-1].strftime('%d/%m/%Y')} ({N_TEST_DAYS} ngày)\n")
        f.write(f"- **Chi phí**: {TOP_K} số × {COST_PER_NUM}k đ = {TOP_K*COST_PER_NUM}k đ/ngày\n")
        f.write(f"- **Tổng vốn đầu tư**: {total_cost * 1000:,}đ\n")
        f.write(f"- **Lợi nhuận ròng**: {total_pnl * 1000:+,}đ\n")
        f.write(f"- **ROI tổng**: **{roi:+.2f}%**\n")
        f.write(f"- **Win Rate ngày**: {win_rate:.1f}% ({win_days}/{N_TEST_DAYS} ngày)\n")
        f.write(f"- **Mức sụt giảm vốn lớn nhất (MDD)**: {max_dd * 1000:,}đ\n\n")
        
        f.write("## Phân phối hiệu suất theo tháng\n\n")
        f.write("| Tháng | Số ngày | Số ngày thắng | Lợi nhuận (đ) | ROI |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in monthly.iterrows():
            p = row["total_pnl"]
            c = row["days"] * TOP_K * COST_PER_NUM
            r = p / c * 100
            f.write(f"| {row['ym']} | {row['days']} | {row['win_days']} | {p*1000:+,}đ | {r:+.2f}% |\n")
            
        f.write("\n## Quy tắc bộ lọc ma trận được áp dụng\n\n")
        f.write("```json\n")
        f.write(json.dumps(predictor.rules, indent=2))
        f.write("\n```\n")
        
    print(f"\n✅ Đã lưu báo cáo chi tiết vào: {report_path}")

if __name__ == "__main__":
    main()
