"""
daily_predict.py — Dự đoán XSMB hàng ngày (Weighted Ensemble, top_k=4)

Cách dùng:
  uv run daily_predict.py                      # dự đoán ngày hôm nay
  uv run daily_predict.py --date 2026-07-14    # ngày cụ thể
  uv run daily_predict.py --top-k 4            # số lượng số chọn
  uv run daily_predict.py --dry-run            # chỉ in, không lưu log
"""
import sys
import json
import argparse
from datetime import datetime, date, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

root_dir = Path(__file__).parent
sys.path.append(str(root_dir / "src"))

import numpy as np
import pandas as pd

from methods.max_delay import MaxDelayPredictor
from methods.conditional_prob import ConditionalProbabilityPredictor
from methods.markov_chain import MarkovChainPredictor
from methods.frequency_momentum import FrequencyMomentumPredictor
from methods.poisson_estimator import PoissonEstimatorPredictor
from methods.ensemble import EnsemblePredictor

# ── Đường dẫn ────────────────────────────────────────────────────────────────
DATA_CSV     = root_dir / "data" / "xsmb-2-digits.csv"
PRED_LOG     = root_dir / "predictions" / "prediction_log.json"
WEIGHTS_FILE = root_dir / "predictions" / "adaptive_weights.json"

COST_PER_NUM   = 27    # nghìn đồng
PAYOUT_PER_HIT = 99    # nghìn đồng
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_weights() -> dict:
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("weights", {})
    return {}

def load_log() -> list:
    if PRED_LOG.exists():
        with open(PRED_LOG, encoding="utf-8") as f:
            return json.load(f)
    return []

def save_log(log: list) -> None:
    with open(PRED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def fetch_today_data():
    """Tự động fetch dữ liệu mới nhất từ web (dùng Lottery class)."""
    try:
        from lottery import Lottery
        from datetime import timedelta

        print("🌐 Đang cập nhật dữ liệu mới nhất từ web…")
        lottery = Lottery()
        lottery.load()
        tz = ZoneInfo("Asia/Ho_Chi_Minh")
        now = datetime.now(tz)
        last_date = now.date()
        if now.time() < dtime(18, 35):
            last_date -= timedelta(days=1)

        begin = lottery.get_last_date()
        delta = (last_date - begin).days
        if delta > 0:
            for i in range(1, delta + 1):
                d = begin + timedelta(days=i)
                print(f"  Fetch ngày {d}…")
                lottery.fetch(d)
            lottery.generate_dataframes()
            lottery.dump()
            print(f"✅ Đã cập nhật đến {last_date}")
        else:
            print("✅ Dữ liệu đã cập nhật mới nhất")
    except Exception as e:
        print(f"⚠️  Không thể fetch dữ liệu mới: {e}")
        print("   Tiếp tục dùng dữ liệu CSV hiện có")

def build_ensemble(weights: dict) -> EnsemblePredictor:
    base_preds = [
        MaxDelayPredictor(),
        ConditionalProbabilityPredictor(),
        MarkovChainPredictor(),
        FrequencyMomentumPredictor(window_size=30),
        PoissonEstimatorPredictor(window_size=180),
    ]
    return EnsemblePredictor(base_preds, weights=weights)

def run_predict(target_date: date, top_k: int, weights: dict) -> dict:
    """Chạy dự đoán cho target_date, trả về dict entry."""
    df = pd.read_csv(DATA_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    prize_cols = [c for c in df.columns if c != "date"]

    # Chỉ dùng lịch sử trước target_date
    history = df[df["date"].dt.date < target_date].reset_index(drop=True)
    if len(history) < 30:
        raise ValueError(f"Không đủ dữ liệu lịch sử trước {target_date}")

    # Build binary matrix
    arr  = history[prize_cols].values.astype(int)
    S    = np.zeros((len(history), 100), dtype=int)
    rows = np.repeat(np.arange(len(history)), arr.shape[1])
    cols = arr.flatten()
    valid = (cols >= 0) & (cols < 100)
    S[rows[valid], cols[valid]] = 1

    # Dự đoán từng phương pháp
    per_method = {}
    methods = [
        ("Max Delay",          MaxDelayPredictor()),
        ("ConditionalProb",    ConditionalProbabilityPredictor()),
        ("MarkovChain",        MarkovChainPredictor()),
        ("FrequencyMomentum",  FrequencyMomentumPredictor(window_size=30)),
        ("PoissonEstimator",   PoissonEstimatorPredictor(window_size=180)),
    ]
    for name, pred in methods:
        try:
            preds = pred.predict(history, top_k=20, S=S)
            per_method[name] = preds[:20]
        except Exception as e:
            per_method[name] = []

    # Ensemble
    ensemble = build_ensemble(weights)
    ensemble_picks = ensemble.predict(history, top_k=top_k, S=S)

    return {
        "date"             : str(target_date),
        "predicted_at"     : datetime.now(TZ).isoformat(),
        "top_k"            : top_k,
        "ensemble_picks"   : ensemble_picks,
        "per_method"       : per_method,
        "weights_used"     : weights,
        "actual_results"   : None,
        "ensemble_hits"    : None,
        "per_method_hits"  : None,
        "cost_k"           : top_k * COST_PER_NUM,
        "revenue_k"        : None,
        "pnl_k"            : None,
    }

def print_prediction(entry: dict) -> None:
    target_date = entry["date"]
    picks       = entry["ensemble_picks"]
    top_k       = entry["top_k"]
    cost        = entry["cost_k"]

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  🎯  DỰ ĐOÁN XSMB — {target_date}                    ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Phương pháp : Weighted Ensemble (5 mô hình)         ║")
    print(f"║  Chọn {top_k} số    : {', '.join(f'{n:02d}' for n in picks):<44}║")
    print(f"║  Chi phí     : {top_k} × {COST_PER_NUM},000đ = {cost:,},000đ{' '*20}║")
    print(f"║  Trúng 1 số  : {PAYOUT_PER_HIT},000đ/nháy{' '*32}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Gợi ý từng phương pháp (top 4):                     ║")
    for method, preds in entry["per_method"].items():
        top4 = ", ".join(f"{n:02d}" for n in preds[:4]) if preds else "—"
        print(f"║    {method:<20}: {top4:<28}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dự đoán XSMB hàng ngày")
    parser.add_argument("--date",    type=str, default=None, help="Ngày dự đoán (YYYY-MM-DD), mặc định là hôm nay")
    parser.add_argument("--top-k",  type=int, default=4,    help="Số lượng số chọn (mặc định 4)")
    parser.add_argument("--dry-run", action="store_true",   help="Chỉ in kết quả, không lưu log")
    parser.add_argument("--no-fetch", action="store_true",  help="Bỏ qua bước fetch dữ liệu mới")
    args = parser.parse_args()

    # Xác định ngày
    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        now = datetime.now(TZ)
        target_date = now.date()

    # Fetch dữ liệu mới
    if not args.no_fetch:
        fetch_today_data()

    # Load trọng số thích ứng
    weights = load_weights()
    print(f"\n📦 Trọng số hiện tại: {weights}")

    # Kiểm tra đã dự đoán ngày này chưa
    log = load_log()
    existing = [e for e in log if e["date"] == str(target_date)]
    if existing and not args.dry_run:
        print(f"\n⚠️  Đã có dự đoán cho ngày {target_date}. Dùng --dry-run để xem lại.")
        print_prediction(existing[0])
        return

    # Chạy dự đoán
    print(f"\n🔮 Đang tính dự đoán cho {target_date}…")
    entry = run_predict(target_date, args.top_k, weights)

    # In kết quả
    print_prediction(entry)

    # Lưu vào log
    if not args.dry_run:
        log.append(entry)
        save_log(log)
        print(f"✅ Đã lưu dự đoán vào {PRED_LOG}")
    else:
        print("ℹ️  Dry-run: không lưu log")

if __name__ == "__main__":
    main()
