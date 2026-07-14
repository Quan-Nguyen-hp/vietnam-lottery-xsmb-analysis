import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from datetime import datetime

# Add root folder to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

from src.feature_engine import FeatureEngine
from src.methods.lightgbm_predictor import LightGBMPredictor

COST_PER_NUM   = 27    # 27,000 VND / number
PAYOUT_PER_HIT = 99    # 99,000 VND / hit point
RETRAIN_INTERVAL = 30  # Retrain LightGBM model every 30 days

def run_lightgbm_backtest(n_test_days: int = 365, top_k: int = 10):
    # Load 2-digit loto data
    csv_path = root_dir / 'data' / 'xsmb-2-digits.csv'
    if not csv_path.exists():
        print(f"❌ Error: data file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    total_days = len(df)
    start_idx = total_days - n_test_days
    
    print(f"Total days available: {total_days}")
    print(f"Running LightGBM walk-forward backtest on the last {n_test_days} days (from day {start_idx} to {total_days})...")
    
    prize_cols = [c for c in df.columns if c != 'date']
    
    # Pre-compute full binary matrix S
    arr_full = df[prize_cols].values.astype(int)
    S_full = np.zeros((total_days, 100), dtype=np.int8)
    rows_full = np.repeat(np.arange(total_days), arr_full.shape[1])
    cols_full = arr_full.flatten()
    valid_full = (cols_full >= 0) & (cols_full < 100)
    S_full[rows_full[valid_full], cols_full[valid_full]] = 1
    
    engine = FeatureEngine()
    predictor = LightGBMPredictor()
    
    # Track statistics
    history_pnl = []
    daily_hits = []
    daily_wins = 0
    total_cost = 0
    total_revenue = 0
    
    daily_results_log = []
    
    # Walk-forward simulation
    for step, idx in enumerate(range(start_idx, total_days)):
        current_row = df.iloc[idx]
        current_date = pd.to_datetime(current_row['date'])
        
        # 1. Periodic Retraining
        if step % RETRAIN_INTERVAL == 0:
            print(f"🔄 Day {step}/{n_test_days}: Retraining LightGBM on history (up to {current_date.date()})...")
            # Train model using history up to idx-1
            history_df = df.iloc[:idx].reset_index(drop=True)
            predictor.train_on_history(history_df, S=S_full[:idx])
            
        # 2. Get Features & Predictions
        history_df = df.iloc[:idx].reset_index(drop=True)
        probs = predictor.predict_proba(history_df, S=S_full[:idx])
        
        # Sort and select top_k
        chosen_nums = np.argsort(probs)[::-1][:top_k].tolist()
        
        # 3. Evaluate results
        actual_lotos = current_row[prize_cols].values.astype(int)
        actual_counts = {}
        for val in actual_lotos:
            if 0 <= val < 100:
                actual_counts[val] = actual_counts.get(val, 0) + 1
                
        hits = sum(actual_counts.get(num, 0) for num in chosen_nums)
        cost = top_k * COST_PER_NUM
        revenue = hits * PAYOUT_PER_HIT
        pnl = revenue - cost
        
        # Append stats
        total_cost += cost
        total_revenue += revenue
        daily_hits.append(hits)
        is_win = 1 if hits > 0 else 0
        if is_win:
            daily_wins += 1
            
        history_pnl.append(pnl)
        
        daily_results_log.append({
            'date': str(current_date.date()),
            'picks': chosen_nums,
            'hits': hits,
            'pnl': pnl,
            'win': is_win
        })
        
    # Aggregate Metrics
    pnl_array = np.array(history_pnl)
    cum_pnl = np.cumsum(pnl_array)
    
    # Calculate Max Drawdown (MDD)
    peaks = np.maximum.accumulate(cum_pnl)
    drawdowns = peaks - cum_pnl
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    net_profit = total_revenue - total_cost
    roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
    win_rate = (daily_wins / n_test_days) * 100
    avg_hits = np.mean(daily_hits)
    max_hits = np.max(daily_hits)
    
    # Monthly Breakdown
    monthly_data = {}
    for r in daily_results_log:
        ym = r['date'][:7]
        if ym not in monthly_data:
            monthly_data[ym] = {'days': 0, 'wins': 0, 'cost': 0, 'pnl': 0}
        m = monthly_data[ym]
        m['days'] += 1
        m['wins'] += r['win']
        m['cost'] += top_k * COST_PER_NUM
        m['pnl'] += r['pnl']
        
    # Report writing
    results_dir = root_dir / 'backtests' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / 'lightgbm_backtest_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo Backtest — Mô hình LightGBM Meta Predictor\n\n")
        f.write(f"- **Kỳ kiểm thử**: {df.iloc[start_idx]['date'].strftime('%d-%m-%Y')} đến {df.iloc[-1]['date'].strftime('%d-%m-%Y')} ({n_test_days} ngày)\n")
        f.write(f"- **Cài đặt**: Chọn Top {top_k} số lô mỗi ngày. Giá mua: {COST_PER_NUM},000đ/số. Trúng giải: {PAYOUT_PER_HIT},000đ/nháy.\n")
        f.write(f"- **Tần suất huấn luyện lại**: {RETRAIN_INTERVAL} ngày/lần\n")
        f.write(f"- **Tổng vốn chi ra**: {total_cost*1000:,.0f}đ\n")
        f.write(f"- **Lợi nhuận ròng**: **{net_profit*1000:+,.0f}đ**\n")
        f.write(f"- **ROI tổng**: **{roi:+.2f}%**\n")
        f.write(f"- **Win Rate ngày**: **{win_rate:.1f}%** ({daily_wins}/{n_test_days} ngày)\n")
        f.write(f"- **Mức sụt giảm vốn lớn nhất (MDD)**: {max_dd*1000:,.0f}đ\n")
        f.write(f"- **Số nháy trúng trung bình/ngày**: {avg_hits:.3f} nháy (Kỳ vọng lý thuyết: {top_k * 0.27:.2f})\n")
        f.write(f"- **Số nháy trúng nhiều nhất/ngày**: {max_hits} nháy\n\n")
        
        f.write("## Phân phối hiệu suất theo tháng\n\n")
        f.write("| Tháng | Số ngày | Số ngày thắng | Lợi nhuận (đ) | ROI |\n")
        f.write("|---|---|---|---|---|\n")
        for ym in sorted(monthly_data.keys()):
            m = monthly_data[ym]
            m_roi = (m['pnl'] / m['cost']) * 100
            f.write(f"| {ym} | {m['days']} | {m['wins']} | {m['pnl']*1000:+,.0f}đ | {m_roi:+.2f}% |\n")
            
        f.write("\n## Top 15 ngày có PnL tốt nhất\n\n")
        f.write("| Ngày | Số chọn | Nháy trúng | Lợi nhuận (đ) |\n")
        f.write("|---|---|---|---|\n")
        sorted_by_pnl = sorted(daily_results_log, key=lambda x: x['pnl'], reverse=True)
        for r in sorted_by_pnl[:15]:
            picks_str = ", ".join(f"{n:02d}" for n in r['picks'])
            f.write(f"| {r['date']} | {picks_str} | {r['hits']} | {r['pnl']*1000:+,.0f}đ |\n")
            
    print(f"\n✅ Backtest hoàn thành! Báo cáo được ghi vào {report_path}")
    print(f"Lợi nhuận ròng: {net_profit*1000:+,.0f}đ | ROI: {roi:+.2f}% | Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    run_lightgbm_backtest(n_test_days=365, top_k=10)
