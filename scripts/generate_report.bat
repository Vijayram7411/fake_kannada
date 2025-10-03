@echo off
setlocal
cd /d "C:\fakee"

if exist venv\Scripts\activate (
  call venv\Scripts\activate
)

echo Generating evaluation report from saved results
python src\generate_report.py

echo Report generated at results\final_evaluation_report.md
