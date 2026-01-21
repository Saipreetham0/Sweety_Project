@echo off
echo Starting Resume AI Detector...

:: Start Backend in a new window
echo Starting Backend (Port 8000)...
start "Sweety Backend" cmd /k "cd backend && if not exist venv (python -m venv venv && call venv\Scripts\activate && pip install -r requirements.txt && python -m spacy download en_core_web_sm) else (call venv\Scripts\activate) && uvicorn main:app --reload"

:: Start Frontend in a new window
echo Starting Frontend (Port 3000)...
start "Sweety Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo.
echo Application keys:
echo  - Backend: http://localhost:8000
echo  - Frontend: http://localhost:3000
echo.
echo Close the popup windows to stop the servers.
pause
