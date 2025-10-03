@echo off
setlocal
cd /d "C:\fakee"

if exist venv\Scripts\activate (
  call venv\Scripts\activate
)

echo Running k-fold cross-validation using %1 (CSV)
set DATA=%1
if "%DATA%"=="" (
  if exist data\processed_dataset.csv (
    set DATA=data\processed_dataset.csv
  ) else if exist data\train_processed.csv (
    set DATA=data\train_processed.csv
  ) else (
    echo Dataset CSV not found. Provide path: scripts\run_crossval.bat ^<path_to_csv^>
    exit /b 1
  )
)

python main.py --mode crossval --data "%DATA%" --k_folds 5 --batch_size 8 --epochs 2
