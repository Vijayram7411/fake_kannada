@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Windows quick start: create venv, install, download NLTK data, run API
cd /d "C:\fakee"

if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
)

call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Download required NLTK data (silent)
python - <<PY
import nltk
for p in ['punkt','stopwords','wordnet']:
    try:
        nltk.data.find(p)
    except LookupError:
        nltk.download(p, quiet=True)
print('NLTK data ready')
PY

set PORT=5000
python main.py --mode api --port %PORT%
