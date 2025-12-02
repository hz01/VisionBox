@echo off
echo Starting VisionBox Backend...
cd backend
python -m venv venv 2>nul
call venv\Scripts\activate
pip install -r requirements.txt >nul 2>&1
python main.py

