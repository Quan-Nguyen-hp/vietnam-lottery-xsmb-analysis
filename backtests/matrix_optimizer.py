"""
matrix_optimizer.py — Bộ tối ưu hóa ma trận quyết định tự động.
Sử dụng Random Search qua cache ma trận đặc trưng lịch sử để tìm các ngưỡng lọc
đạt ROI cao nhất và lưu vào predictions/matrix_rules.json.
"""
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.methods.matrix_decision import MatrixDecisionPredictor

COST_PER_NUM   = 27
PAYOUT_PER_HIT = 99
N_TEST_DAYS    = 180      # tối ưu dựa trên 180 ngày gần nhất
N_ITERATIONS   = 2000     # số lượt thử ngẫu nhiên
TOP_K          = 10

RULES_FILE = root_dir / "predictions" / "matrix_rules.json"

def main():
    csv_path = root_dir / "data" / "xsmb-2-digits.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    total_days = len(df)
    prize_cols = [c for c in df.columns if c != "date"]

    arr_full = df[prize_cols].values.astype(int)
    S_full = np.zeros((total_days, 100), dtype=np.int8)
    rows_f = np.repeat(np.arange(total_days), arr_full.shape[1])
    cols_f = arr_full.flatten()
    valid  = (cols_f >= 0) & (cols_f < 100)
    S_full[rows_f[valid], cols_f[valid]] = 1

    start_idx = total_days - N_TEST_DAYS

    # 1. Sinh cache ma trận đặc trưng và lưu kết quả thực tế cho mỗi ngày
    print(f"🔄 Đang tạo bộ nhớ đệm (cache) ma trận đặc trưng cho {N_TEST_DAYS} ngày...")
    predictor = MatrixDecisionPredictor()
    
    cache_matrices = []  # list of dict[num][feature]
    cache_actuals = []   # list of dict {num: count}
    
    for step, idx in enumerate(range(start_idx, total_days)):
        S_hist = S_full[:idx]
        matrix = predictor.build_matrix(None, S=S_hist)
        cache_matrices.append(matrix)
        
        actual_row = df.iloc[idx]
        actual_lotos = actual_row[prize_cols].values.astype(int)
        actual_counts = {}
        for val in actual_lotos:
            if 0 <= val < 100:
                actual_counts[val] = actual_counts.get(val, 0) + 1
        cache_actuals.append(actual_counts)
        
        if (step + 1) % 100 == 0:
            print(f"  Đã tạo: {step + 1}/{N_TEST_DAYS} ngày...")

    # 2. Định nghĩa hàm tính ROI nhanh từ cache
    def evaluate_rules(rules: dict) -> tuple[float, float, int]:
        total_pnl = 0
        win_days = 0
        
        for day_i, matrix in enumerate(cache_matrices):
            actual_counts = cache_actuals[day_i]
            
            # Lọc candidates
            candidates = []
            for num in range(100):
                f = matrix[num]
                if not (rules["delay_min"] <= f["delay"] <= rules["delay_max"]):
                    continue
                if f["poisson"] < rules["poisson_min"]: continue
                if f["markov"] < rules["markov_min"]: continue
                if f["momentum"] < rules["momentum_min"]: continue
                if f["repeat"] < rules["repeat_min"]: continue
                if f["pairs"] < rules["pairs_min"]: continue
                if f["cond_prob"] < rules["cond_prob_min"]: continue
                candidates.append(num)
                
            # Thuật toán nới lỏng động nhanh
            curr_rules = dict(rules)
            iter_count = 0
            while len(candidates) < TOP_K and iter_count < 10:
                iter_count += 1
                curr_rules["delay_min"] = max(0.0, curr_rules["delay_min"] - 0.1)
                curr_rules["delay_max"] = min(1.0, curr_rules["delay_max"] + 0.1)
                for k in ["poisson_min", "markov_min", "momentum_min", "repeat_min", "pairs_min", "cond_prob_min"]:
                    curr_rules[k] = max(0.0, curr_rules[k] - 0.1)
                
                candidates = []
                for num in range(100):
                    f = matrix[num]
                    if not (curr_rules["delay_min"] <= f["delay"] <= curr_rules["delay_max"]):
                        continue
                    if f["poisson"] < curr_rules["poisson_min"]: continue
                    if f["markov"] < curr_rules["markov_min"]: continue
                    if f["momentum"] < curr_rules["momentum_min"]: continue
                    if f["repeat"] < curr_rules["repeat_min"]: continue
                    if f["pairs"] < curr_rules["pairs_min"]: continue
                    if f["cond_prob"] < curr_rules["cond_prob_min"]: continue
                    candidates.append(num)

            # Sắp xếp và chọn top_k
            def get_avg_score(n):
                return np.mean(list(matrix[n].values()))
                
            sorted_cand = sorted(candidates, key=get_avg_score, reverse=True)
            if len(sorted_cand) < TOP_K:
                all_sorted = sorted(list(range(100)), key=get_avg_score, reverse=True)
                for num in all_sorted:
                    if num not in sorted_cand:
                        sorted_cand.append(num)
                    if len(sorted_cand) >= TOP_K:
                        break
                        
            chosen = sorted_cand[:TOP_K]
            
            cost = TOP_K * COST_PER_NUM
            hits = sum(actual_counts.get(n, 0) for n in chosen)
            rev = hits * PAYOUT_PER_HIT
            total_pnl += (rev - cost)
            if hits > 0:
                win_days += 1
                
        total_cost = N_TEST_DAYS * TOP_K * COST_PER_NUM
        roi = total_pnl / total_cost * 100
        win_rate = win_days / N_TEST_DAYS * 100
        return roi, total_pnl, win_rate

    # 3. Tìm kiếm ngẫu nhiên (Random Search)
    print(f"\n🚀 Đang chạy tối ưu hoá {N_ITERATIONS} lượt thử...")
    best_roi = -999.0
    best_rules = {}
    best_pnl = 0
    best_wr = 0.0

    # Lượt thử 0: Dùng tập quy tắc mặc định làm baseline
    default_rules = {
        "delay_min"     : 0.1,
        "delay_max"     : 0.9,
        "poisson_min"   : 0.2,
        "markov_min"    : 0.1,
        "momentum_min"  : 0.2,
        "repeat_min"    : 0.0,
        "pairs_min"     : 0.0,
        "cond_prob_min" : 0.0,
    }
    best_roi, best_pnl, best_wr = evaluate_rules(default_rules)
    best_rules = default_rules
    print(f"🟢 Baseline ROI (Default): {best_roi:+.2f}% | Lãi ròng: {best_pnl*1000:+,}đ")

    for i in range(N_ITERATIONS):
        # Generate random thresholds
        d_min = round(random.uniform(0.0, 0.4), 2)
        d_max = round(random.uniform(0.6, 1.0), 2)
        
        trial = {
            "delay_min"     : d_min,
            "delay_max"     : d_max,
            "poisson_min"   : round(random.uniform(0.0, 0.6), 2),
            "markov_min"    : round(random.uniform(0.0, 0.6), 2),
            "momentum_min"  : round(random.uniform(0.0, 0.6), 2),
            "repeat_min"    : round(random.uniform(0.0, 0.4), 2),
            "pairs_min"     : round(random.uniform(0.0, 0.4), 2),
            "cond_prob_min" : round(random.uniform(0.0, 0.3), 2),
        }
        
        roi, pnl, wr = evaluate_rules(trial)
        if roi > best_roi:
            best_roi = roi
            best_pnl = pnl
            best_wr = wr
            best_rules = trial
            print(f"  🏆 Vòng {i:4d}: ROI mới = {best_roi:>+6.2f}% | Lãi ròng = {best_pnl*1000:>+11,.0f}đ (Win rate {best_wr:.1f}%)")

    print(f"\n🔥 HOÀN THÀNH TỐI ƯU HOÁ!")
    print(f"  ROI tốt nhất  : {best_roi:+.2f}%")
    print(f"  Lãi ròng tốt nhất: {(best_pnl*1000):+,}đ")
    print(f"  Quy tắc tối ưu:")
    for k, v in best_rules.items():
        print(f"    {k:<15}: {v}")

    # Ghi lại file json
    rules_payload = {
        "updated_at": pd.Timestamp.now().isoformat(),
        "backtest_days": N_TEST_DAYS,
        "roi_achieved": best_roi,
        "pnl_achieved": best_pnl,
        "win_rate": best_wr,
        "rules": best_rules
    }
    
    RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules_payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu quy tắc tối ưu vào {RULES_FILE}")

if __name__ == "__main__":
    main()
