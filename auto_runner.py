"""
auto_runner.py — Script chạy tự động cả 2 việc:
  - Nếu 01:00–18:34 UTC (8:00–01:34 VN): dự đoán HÔM NAY
  - Nếu sau 12:05 UTC (19:05 VN)        : cập nhật kết quả + dự đoán NGÀY MAI

Được gọi bởi GitHub Actions lúc 01:00 UTC và 12:05 UTC mỗi ngày.
Hoặc bởi Windows Task Scheduler lúc 8:00 và 19:05 VN.
"""
import sys
import json
import shutil
import subprocess
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

root_dir = Path(__file__).parent
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

LOG_FILE = root_dir / "predictions" / "auto_runner.log"
PRED_LOG = root_dir / "predictions" / "prediction_log.json"

def log(msg: str):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_script(script: str, args: list[str] = []) -> bool:
    # Ưu tiên dùng uv run (GitHub Actions / venv), fallback sang python
    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [uv_bin, "run", script] + args
    else:
        cmd = [sys.executable, str(root_dir / script)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root_dir))
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log(f"  {line}")
    if result.returncode != 0 and result.stderr:
        log(f"  ❌ LỖI: {result.stderr.strip()[:200]}")
    return result.returncode == 0

def already_predicted(target_date: date) -> bool:
    if not PRED_LOG.exists():
        return False
    with open(PRED_LOG, encoding="utf-8") as f:
        log_data = json.load(f)
    for e in log_data:
        d_val = e["pipeline_metadata"]["date"] if "pipeline_metadata" in e else e.get("date")
        # Bản v1.2 APPROVED lưu top_k trong bets hoặc metadata, chỉ cần so ngày là đủ
        if d_val == str(target_date):
            return True
    return False

def already_updated(target_date: date) -> bool:
    if not PRED_LOG.exists():
        return False
    with open(PRED_LOG, encoding="utf-8") as f:
        log_data = json.load(f)
    for e in log_data:
        d_val = e["pipeline_metadata"]["date"] if "pipeline_metadata" in e else e.get("date")
        if d_val == str(target_date) and e.get("actual_results") is not None:
            return True
    return False

def main():
    now  = datetime.now(TZ)
    today = now.date()
    yesterday = today - timedelta(days=1)
    tomorrow  = today + timedelta(days=1)

    log("=" * 55)
    log(f"🚀 Auto runner khởi động — {now.strftime('%d/%m/%Y %H:%M')}")

    after_lottery = now.time() >= dtime(18, 35)

    if after_lottery:
        # ── Sau 18:35: cập nhật kết quả hôm nay + dự đoán ngày mai ──────
        log(f"📋 Giai đoạn: SAU XỔ SỐ ({now.strftime('%H:%M')})")

        # 1. Cập nhật kết quả hôm nay (nếu chưa)
        if not already_updated(today):
            log(f"🔄 Cập nhật kết quả ngày {today}…")
            ok = run_script("daily_update.py", ["--date", str(today)])
            if ok:
                log(f"✅ Đã cập nhật kết quả {today}")
            else:
                log(f"⚠️  Không thể cập nhật {today}")
        else:
            log(f"✅ Kết quả {today} đã được cập nhật rồi")

        # 2. Dự đoán ngày mai (nếu chưa)
        if not already_predicted(tomorrow):
            log(f"🔮 Dự đoán ngày mai {tomorrow}…")
            ok = run_script("daily_predict.py", ["--date", str(tomorrow), "--no-fetch"])
            if ok:
                log(f"✅ Đã lưu dự đoán cho {tomorrow}")
            else:
                log(f"⚠️  Không thể dự đoán {tomorrow}")
        else:
            log(f"✅ Đã có dự đoán cho {tomorrow} rồi")

    else:
        # ── Trước 18:35: dự đoán hôm nay (nếu chưa có) ──────────────────
        log(f"📋 Giai đoạn: TRƯỚC XỔ SỐ ({now.strftime('%H:%M')})")

        if not already_predicted(today):
            log(f"🔮 Dự đoán hôm nay {today}…")
            ok = run_script("daily_predict.py", ["--date", str(today)])
            if ok:
                log(f"✅ Đã lưu dự đoán cho {today}")
            else:
                log(f"⚠️  Không thể dự đoán {today}")
        else:
            log(f"✅ Đã có dự đoán cho {today} rồi — chờ kết quả sau 18:35")

    log("🏁 Auto runner hoàn thành")
    log("=" * 55)

if __name__ == "__main__":
    main()
