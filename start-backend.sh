#!/bin/bash
echo "Starting VisionBox Backend..."
cd backend
python3 -m venv venv 2>/dev/null
source venv/bin/activate
pip install -r requirements.txt >/dev/null 2>&1
python main.py

