"""
XPIS v1.1 — Parameter Sweep Optimization Tool
Chạy quét lưới (Grid Search) tìm ngưỡng tối ưu cho min_probability và min_confidence.

Quy trình:
1. Load lịch sử 180 ngày.
2. Dùng Parquet snapshots và pre-compute 11 model probabilities để tối đa hóa tốc độ.
3. Quét lưới min_prob ∈ [0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32]
             min_conf ∈ [0.40, 0.45, 0.50, 0.55, 0.60]
4. Xuất kết quả chi tiết ra backtests/results/parameter_sweep_report.md
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Thêm src vào sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

from src.data.loader import DataLoader
from src.evidence.builder import EvidenceBuilder
from src.evidence.store import EvidenceStore
from src.features.feature_store import FeatureStore
from src.probability import get_all_models
from src.meta.fusion import MetaFusion
from src.decision.engine import DecisionEngine

# Cấu hình quét
PROB_GRID = [0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32]
CONF_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]
N_TEST_DAYS = 180

COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0


def run_parameter_sweep():
    print("=== Khởi chạy Parameter Sweep XPIS v1.1 ===")
    
    loader = DataLoader().load()
    total_days = loader.total_days
    start_idx = total_days - N_TEST_DAYS
    
    evidence_store = EvidenceStore()
    evidence_builder = EvidenceBuilder(evidence_store)
    feature_store = FeatureStore(S_history=loader.S, evidence_store=evidence_store)
    models = get_all_models()
    
    # Pre-load/pre-compute tất cả đặc trưng của 180 ngày để chạy sweep siêu tốc
    print("Đang pre-load dữ liệu & tính toán mô hình cơ sở cho 180 ngày test...")
    t0 = time.time()
    
    days_data = []
    
    for idx in range(start_idx, total_days):
        current_row = loader.df.iloc[idx]
        current_date = pd.to_datetime(current_row['date'])
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Actuals
        prize_cols = loader.prize_cols()
        actual_lotos = current_row[prize_cols].dropna().values.astype(int).tolist()
        
        # Load features
        df_hist, S_hist = loader.slice_history(idx)
        df_ev = evidence_builder.build_all(df_hist, S_hist, current_date.to_pydatetime(), save=True)
        df_feat = feature_store.build(df_ev, date_str, S=S_hist)
        
        # Tính xác suất từ 11 models
        model_probas = {}
        for m in models:
            # Lưu ý: do đây là bước kiểm thử tĩnh, chúng ta giả định mô hình LGBM đã được huấn luyện sẵn hoặc fallback
            # Để chính xác nhất, trong sweep này ta sẽ sử dụng Weighted Fusion trên 10 mô hình thống kê trước
            # hoặc nạp các mô hình đã fit. Ta lấy 10 mô hình thống kê đầu để tối đa hóa độ tin cậy tĩnh.
            if m.name != "lightgbm_classifier":
                model_probas[m.name] = m.predict_proba(df_feat, df_hist, S_hist)
        
        # Fusion tĩnh (Weighted Fusion)
        fusion = MetaFusion()
        meta_proba = fusion.fuse(model_probas)
        
        days_data.append({
            'date': date_str,
            'meta_proba': meta_proba,
            'model_probas': model_probas,
            'actual_lotos': actual_lotos
        })
        
    print(f"Pre-compute hoàn thành sau {time.time()-t0:.1f}s. Bắt đầu quét lưới...")
    
    sweep_results = []
    
    # Chạy quét
    for min_prob in PROB_GRID:
        for min_conf in CONF_GRID:
            # Khởi tạo DecisionEngine cho cặp tham số này
            engine = DecisionEngine(
                min_probability=min_prob,
                min_confidence=min_conf,
                top_k=10,
                kelly_odds=PAYOUT_PER_HIT / COST_PER_BET,
                kelly_fraction=0.20
            )
            
            flat_bets = 0
            flat_hits = 0
            flat_cost = 0.0
            flat_payout = 0.0
            
            kelly_bankroll = 10000.0
            kelly_bankroll_history = []
            
            days_with_bets = 0
            
            for day in days_data:
                day_decision = engine.decide(
                    date=day['date'],
                    meta_proba=day['meta_proba'],
                    model_probas=day['model_probas']
                )
                
                bets = day_decision.bets
                n_bets = len(bets)
                actual_lotos = day['actual_lotos']
                
                if n_bets > 0:
                    days_with_bets += 1
                    
                # Flat Betting evaluation
                for b in bets:
                    matches = actual_lotos.count(b.number)
                    flat_bets += 1
                    flat_cost += COST_PER_BET
                    if matches > 0:
                        flat_hits += matches
                        flat_payout += matches * PAYOUT_PER_HIT
                        
                # Kelly evaluation
                if n_bets > 0 and kelly_bankroll > 500.0:
                    bet_allocations = {}
                    for b in bets:
                        bet_amount = b.kelly_fraction * kelly_bankroll
                        if bet_amount >= 5.0:
                            bet_allocations[b.number] = bet_amount
                            
                    total_bet = sum(bet_allocations.values())
                    if total_bet > 0.50 * kelly_bankroll:
                        scale = (0.50 * kelly_bankroll) / total_bet
                        bet_allocations = {n: a * scale for n, a in bet_allocations.items()}
                        total_bet = sum(bet_allocations.values())
                        
                    winnings = 0.0
                    for num, amt in bet_allocations.items():
                        matches = actual_lotos.count(num)
                        winnings += matches * amt * (PAYOUT_PER_HIT / COST_PER_BET)
                        
                    kelly_bankroll += (winnings - total_bet)
                    
                kelly_bankroll_history.append(kelly_bankroll)
                
            # Tổng kết cặp tham số
            flat_pnl = flat_payout - flat_cost
            flat_roi = (flat_pnl / flat_cost * 100.0) if flat_cost > 0 else 0.0
            
            kelly_pnl = kelly_bankroll - 10000.0
            kelly_roi = (kelly_pnl / 10000.0 * 100.0)
            
            sweep_results.append({
                'min_prob': min_prob,
                'min_conf': min_conf,
                'days_traded': days_with_bets,
                'flat_bets': flat_bets,
                'flat_hits': flat_hits,
                'flat_roi': flat_roi,
                'kelly_roi': kelly_roi,
                'kelly_final_pts': kelly_bankroll
            })
            
    # Tạo báo cáo Sweep
    df_sweep = pd.DataFrame(sweep_results)
    
    results_dir = root_dir / 'backtests' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / 'parameter_sweep_report.md'
    
    # Sắp xếp theo Kelly ROI để tìm bộ tham số tốt nhất
    df_sorted = df_sweep.sort_values('kelly_roi', ascending=False)
    best_config = df_sorted.iloc[0]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo Quét tối ưu hóa tham số (Parameter Sweep) — XPIS v1.1\n\n")
        f.write(f"- **Kỳ kiểm thử**: 180 ngày qua\n")
        f.write(f"- **Số lượng tổ hợp thử nghiệm**: {len(df_sweep)}\n")
        f.write(f"- **Cấu hình tốt nhất đề xuất**: Probability: **{best_config['min_prob']:.2f}** | Confidence: **{best_config['min_conf']:.2f}** (Kelly ROI: **{best_config['kelly_roi']:+.2f}%**)\n\n")
        
        f.write("## Bảng so sánh hiệu suất quét lưới\n\n")
        f.write("| Ngưỡng Prob | Ngưỡng Conf | Số ngày cược | Số lượt cược | Win rate cược | ROI Cược Phẳng | ROI Kelly (Vốn) | Vốn cuối kỳ (điểm) |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
        for _, r in df_sweep.iterrows():
            win_rate = (r['flat_hits'] / r['flat_bets'] * 100.0) if r['flat_bets'] > 0 else 0.0
            f.write(
                f"| {r['min_prob']:.2f} | {r['min_conf']:.2f} | {r['days_traded']} | {r['flat_bets']} | "
                f"{win_rate:.1f}% | {r['flat_roi']:+.2f}% | {r['kelly_roi']:+.2f}% | {r['kelly_final_pts']:.1f} |\n"
            )
            
    print(f"\n✅ Sweep hoàn tất! Báo cáo tối ưu hóa đã được ghi vào {report_path}")
    print(f"Cấu hình tối ưu nhất: Prob={best_config['min_prob']:.2f}, Conf={best_config['min_conf']:.2f} -> Kelly ROI: {best_config['kelly_roi']:+.2f}%")


if __name__ == "__main__":
    run_parameter_sweep()
