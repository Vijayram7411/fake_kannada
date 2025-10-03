@echo off
setlocal
cd /d "C:\fakee"

REM Optionally activate venv if present
if exist venv\Scripts\activate (
  call venv\Scripts\activate
)

set DATA=%1
if "%DATA%"=="" (
  if exist data\processed_dataset.csv (
    set DATA=data\processed_dataset.csv
  ) else if exist data\train_processed.csv (
    set DATA=data\train_processed.csv
  ) else (
    echo Dataset CSV not found. Provide path: train_transformer.bat ^<path_to_csv^>
    exit /b 1
  )
)

echo Training transformer model on %DATA%
python main.py --mode train --data "%DATA%" --epochs 2 --batch_size 8

echo If training succeeded, the model should be saved to models\multilingual_fake_news_model.pth
