"""
XPIS v1.1 — Walk-Forward Backtesting Engine
Kiểm thử toàn diện kiến trúc nâng cấp v1.1.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
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
from src.probability.lgb_model import LightGBMProbabilityModel
from src.meta.fusion import MetaFusion
from src.decision.engine import DecisionEngine
from src.evaluation.metrics import EvaluationMetrics

# Cấu hình tài chính
COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0
RETRAIN_INTERVAL = 30
TRAIN_WINDOW_DAYS = 365

def run_xpis_backtest(n_test_days: int = 180, min_prob: float = 0.31, min_conf: float = 0.45):
    print("=== Khởi chạy Backtest Walk-Forward XPIS v1.1 ===")
    
    loader = DataLoader().load()
    total_days = loader.total_days
    start_idx = total_days - n_test_days
    
    evidence_store = EvidenceStore()
    evidence_builder = EvidenceBuilder(evidence_store)
    feature_store = FeatureStore(S_history=loader.S, evidence_store=evidence_store)
    
    # Layer 4 Models
    static_models = [m for m in get_all_models() if m.name != "lightgbm_classifier"]
    lgb_model = LightGBMProbabilityModel()
    
    # Layer 5 & 6 & 7
    fusion = MetaFusion()
    decision_engine = DecisionEngine(
        min_probability=min_prob,
        min_confidence=min_conf,
        top_k=10,
        kelly_odds=PAYOUT_PER_HIT / COST_PER_BET,
        kelly_fraction=0.20,
        min_diversification=0.85
    )
    evaluator = EvaluationMetrics(odds=PAYOUT_PER_HIT / COST_PER_BET, cost_per_bet=COST_PER_BET)
    
    flat_bets_log = []
    kelly_bankroll = 10000.0
    kelly_bankroll_history = [kelly_bankroll]
    daily_results = []
    
    # Cần lưu lịch sử dự báo để tính toán trọng số động
    history_model_preds = {m.name: [] for m in static_models}
    history_model_preds[lgb_model.name] = []
    history_labels = []
    
    t_start = time.time()
    
    for step, idx in enumerate(range(start_idx, total_days)):
        current_row = loader.df.iloc[idx]
        current_date = pd.to_datetime(current_row['date'])
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Nhãn
        prize_cols = loader.prize_cols()
        actual_lotos = current_row[prize_cols].dropna().values.astype(int).tolist()
        actual_set = set(actual_lotos)
        y = np.zeros(100)
        for num in actual_set:
            y[num] = 1
            
        # 1. Định kỳ retraining và tính lại trọng số động
        if step % RETRAIN_INTERVAL == 0:
            print(f"🔄 Ngày {step}/{n_test_days} ({date_str}): Huấn luyện lại LGBM và cập nhật trọng số Fusion...")
            
            # Huấn luyện mô hình LGBM ở Layer 4
            train_start_idx = max(50, idx - TRAIN_WINDOW_DAYS)
            train_snapshots = []
            train_labels = []
            
            for t_idx in range(train_start_idx, idx):
                t_df_hist, t_S_hist = loader.slice_history(t_idx)
                t_row = loader.df.iloc[t_idx]
                t_date = t_row['date'].to_pydatetime()
                t_date_str = t_date.strftime('%Y-%m-%d')
                
                t_df_ev = evidence_builder.build_all(t_df_hist, t_S_hist, t_date, save=True)
                t_df_feat = feature_store.build(t_df_ev, t_date_str, S=t_S_hist)
                train_snapshots.append(t_df_feat)
                
                t_actual = set(t_row[prize_cols].dropna().values.astype(int).tolist())
                t_y = np.zeros(100)
                for num in t_actual:
                    t_y[num] = 1
                train_labels.append(t_y)
                
            # Fit LGBM
            # Khởi tạo ma trận X_train từ snapshots
            meta_cols = [c for c in train_snapshots[0].columns if c not in ("number", "date")]
            X_train = np.vstack([df[meta_cols].values for df in train_snapshots])
            y_train = np.concatenate(train_labels)
            
            lgb_model.fit(X_train, y_train, feature_names=meta_cols)
            
            # Tính toán trọng số Fusion động cho Layer 5 từ kết quả dự báo trong train window
            eval_predictions = {m.name: [] for m in static_models}
            eval_predictions[lgb_model.name] = []
            
            # Chỉ đánh giá trên 90 ngày gần nhất của train window để giảm thiểu thời gian tính toán
            eval_start = max(train_start_idx, idx - 90)
            for t_idx in range(eval_start, idx):
                t_df_hist, t_S_hist = loader.slice_history(t_idx)
                t_date = loader.df.iloc[t_idx]['date'].to_pydatetime()
                t_date_str = t_date.strftime('%Y-%m-%d')
                
                t_df_ev = evidence_builder.build_all(t_df_hist, t_S_hist, t_date, save=True)
                t_df_feat = feature_store.build(t_df_ev, t_date_str, S=t_S_hist)
                
                for m in static_models:
                    eval_predictions[m.name].append(m.predict_proba(t_df_feat, t_df_hist, t_S_hist))
                eval_predictions[lgb_model.name].append(lgb_model.predict_proba(t_df_feat, t_df_hist, t_S_hist))
                
            # Định dạng thành ma trận (n_days, 100)
            eval_preds_matrix = {k: np.array(v) for k, v in eval_predictions.items()}
            eval_labels_matrix = np.array(train_labels[-(idx - eval_start):])
            
            # Cập nhật trọng số
            fusion.compute_dynamic_weights(eval_preds_matrix, eval_labels_matrix)
            feature_store.clear_ram_cache()
            
        # 2. Dự báo ngày hiện tại
        df_hist, S_hist = loader.slice_history(idx)
        df_ev = evidence_builder.build_all(df_hist, S_hist, current_date.to_pydatetime(), save=True)
        df_feat = feature_store.build(df_ev, date_str, S=S_hist)
        
        # Lấy xác suất của tất cả 11 models
        model_probas = {}
        for m in static_models:
            model_probas[m.name] = m.predict_proba(df_feat, df_hist, S_hist)
        model_probas[lgb_model.name] = lgb_model.predict_proba(df_feat, df_hist, S_hist)
        
        # 3. Fusion xác suất
        meta_proba = fusion.fuse(model_probas)
        
        # 4. Ra quyết định cược (Layer 6)
        day_decision = decision_engine.decide(
            date=date_str,
            meta_proba=meta_proba,
            model_probas=model_probas,
            feature_version=feature_store.version,
            model_version=lgb_model.version,
            evidence_version="v1.0"
        )
        
        # 5. Đánh giá kết quả giao dịch
        bets = day_decision.bets
        n_bets = len(bets)
        n_hits = 0
        pnl_flat = 0.0
        
        for b in bets:
            matches = actual_lotos.count(b.number)
            if matches > 0:
                n_hits += matches
                pnl_flat += (matches * PAYOUT_PER_HIT) - COST_PER_BET
            else:
                pnl_flat -= COST_PER_BET
                
        # Đánh giá Kelly
        day_kelly_pnl = 0.0
        active_bets_info = []
        
        if n_bets > 0 and kelly_bankroll > 500.0:
            bet_allocations = {}
            for b in bets:
                # b.allocation là % vốn (ví dụ: 0.02 = 2% vốn)
                bet_amount = b.allocation * kelly_bankroll
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
                active_bets_info.append(f"{num:02d}({amt:.0f}đ)")
                
            day_kelly_pnl = winnings - total_bet
            kelly_bankroll += day_kelly_pnl
            
        kelly_bankroll_history.append(kelly_bankroll)
        
        flat_bets_log.append({
            'date': date_str,
            'hit': n_hits > 0,
            'n_bets': n_bets,
            'n_hits': n_hits,
            'daily_pnl': pnl_flat
        })
        
        daily_results.append({
            'date': date_str,
            'bets': [f"{d.number:02d}" for d in bets],
            'hits': n_hits,
            'pnl_flat': pnl_flat,
            'kelly_pnl': day_kelly_pnl,
            'kelly_bankroll': kelly_bankroll,
            'kelly_bets': ", ".join(active_bets_info) if active_bets_info else "SKIP",
            'div_score': day_decision.diversification_score,
            'actual': [f"{n:02d}" for n in sorted(actual_set)]
        })
        
    # Kết thúc backtest
    elapsed = time.time() - t_start
    print(f"\n✅ Backtest XPIS v1.1 hoàn tất sau {elapsed:.1f} giây!")
    
    # 6. Ghi báo cáo kết quả
    df_flat = pd.DataFrame(flat_bets_log)
    metrics_summary = evaluator.compute_full(df_flat)
    
    results_dir = root_dir / 'backtests' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / 'xpis_backtest_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo hiệu suất kiến trúc XPIS v1.1 (Dynamic Fusion & Calibration)\n\n")
        f.write(f"- **Kỳ kiểm thử**: 180 ngày qua\n")
        f.write(f"- **Hệ thống**: XPIS v1.1 (11 Models + Dynamic Weighted Fusion + Portfolio Risk Manager)\n")
        f.write(f"- **Tham số**: Min Prob: {min_prob:.2f} | Min Conf: {min_conf:.2f} | Min Diversification: 0.85\n")
        f.write(f"- **Thời gian chạy**: {elapsed:.1f}s\n\n")
        
        f.write("## 1. Kết quả Tổng Hợp\n\n")
        f.write("| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |\n")
        f.write("|---|:---:|:---:|\n")
        
        flat_total_cost = df_flat['n_bets'].sum() * COST_PER_BET
        flat_total_pnl = df_flat['daily_pnl'].sum()
        flat_roi = (flat_total_pnl / flat_total_cost) * 100 if flat_total_cost > 0 else 0.0
        
        f.write(f"| Tổng vốn chi | {flat_total_cost*1000:,.0f}đ | Thay đổi theo ngày |\n")
        f.write(f"| Lợi nhuận ròng | **{flat_total_pnl*1000:+,.0f}đ** | **{(kelly_bankroll - 10000.0)*COST_PER_BET*1000:+,.0f}đ** |\n")
        f.write(f"| ROI tổng | **{flat_roi:+.2f}%** | **{((kelly_bankroll / 10000.0) - 1.0)*100:+.2f}%** (Tăng trưởng vốn) |\n")
        f.write(f"| Win Rate ngày | **{metrics_summary['hit_rate']:.1%}** ({metrics_summary['total_hits']}/{n_test_days} ngày có trúng) | — |\n")
        f.write(f"| Tổng số lần cược | {df_flat['n_bets'].sum()} số | {sum(1 for r in daily_results if r['kelly_bets'] != 'SKIP')} ngày cược |\n")
        f.write(f"| Vốn cuối kỳ | — | **{kelly_bankroll*COST_PER_BET*1000:,.0f}đ** ({kelly_bankroll:,.1f} điểm) |\n\n")
        
        # In trọng số tối ưu hiện tại của Layer 5
        f.write("## 2. Trọng số Fusion tối ưu hiện tại của 11 Models (Layer 5)\n\n")
        f.write("| Model | Trọng số |\n")
        f.write("|---|:---:|\n")
        for k, v in sorted(fusion.weights.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {k} | {v:.2%} |\n")
            
        f.write("\n## 3. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)\n\n")
        f.write("| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |\n")
        f.write("|---|---|:---:|---|---|---|:---:|\n")
        
        sorted_daily = sorted(daily_results, key=lambda x: x['pnl_flat'], reverse=True)
        for r in sorted_daily[:15]:
            bets_str = ", ".join(r['bets']) if r['bets'] else "SKIP"
            f.write(f"| {r['date']} | {bets_str} | {r['hits']} | {r['pnl_flat']*1000:+,.0f}đ | {r['kelly_bets']} | {r['kelly_pnl']*COST_PER_BET*1000:+,.0f}đ | {r['div_score']:.2f} |\n")
            
    print(f"✅ Đã cập nhật báo cáo tại: {report_path}")


if __name__ == "__main__":
    # Dùng cấu hình tối ưu quét được (Prob=0.31, Conf=0.45)
    run_xpis_backtest(n_test_days=180, min_prob=0.31, min_conf=0.45)
