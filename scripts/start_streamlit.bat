@echo off
setlocal
cd /d "C:\fakee"

if exist venv\Scripts\activate (
  call venv\Scripts\activate
)

streamlit run src\app.py
