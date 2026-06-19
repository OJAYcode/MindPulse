@echo off
cd /d "c:\Users\HP\Documents\Stressor"
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
