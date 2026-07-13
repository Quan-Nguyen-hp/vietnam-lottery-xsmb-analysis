@echo off
:: setup_scheduler.bat — Cài đặt Task Scheduler chạy tự động hàng ngày
:: Chạy file này 1 lần với quyền Administrator

set PROJECT_DIR=F:\MR_BOM\PYTHON\vietnam-lottery-xsmb-analysis
set PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe
set SCRIPT=%PROJECT_DIR%\auto_runner.py

:: Nếu không có .venv, dùng python hệ thống
if not exist "%PYTHON%" (
    set PYTHON=python
)

echo ============================================================
echo  Cai dat Task Scheduler cho XSMB Auto Predict
echo ============================================================

:: Task 1: Chạy lúc 8:00 sáng (dự đoán hôm nay)
schtasks /create /tn "XSMB_Morning_Predict" ^
    /tr "\"%PYTHON%\" \"%SCRIPT%\"" ^
    /sc daily /st 08:00 ^
    /ru "%USERNAME%" ^
    /f
if %errorlevel%==0 (
    echo [OK] Task XSMB_Morning_Predict - chay luc 08:00 moi ngay
) else (
    echo [LOI] Khong the tao task sang
)

:: Task 2: Chạy lúc 19:00 tối (cập nhật kết quả + dự đoán ngày mai)
schtasks /create /tn "XSMB_Evening_Update" ^
    /tr "\"%PYTHON%\" \"%SCRIPT%\"" ^
    /sc daily /st 19:00 ^
    /ru "%USERNAME%" ^
    /f
if %errorlevel%==0 (
    echo [OK] Task XSMB_Evening_Update - chay luc 19:00 moi ngay
) else (
    echo [LOI] Khong the tao task toi
)

echo.
echo ============================================================
echo  Ket qua: 2 task da duoc cai dat!
echo  - 08:00 sang: du doan hom nay
echo  - 19:00 toi : cap nhat ket qua + du doan ngay mai
echo.
echo  Xem log tai: %PROJECT_DIR%\predictions\auto_runner.log
echo  Xem lich task: schtasks /query /tn XSMB_Morning_Predict
echo ============================================================
pause
