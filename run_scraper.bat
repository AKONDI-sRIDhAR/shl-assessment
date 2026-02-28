@echo off
echo ============================================
echo   SHL Assessment Scraper (Resumable + Fast)
echo ============================================
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

python -u resumable_fast_scraper.py

echo.
echo ============================================
echo   Done! Now run:  streamlit run app.py
echo ============================================
pause
