@echo off
echo ============================================================
echo   GeneRx AI — Setup ^& Launch
echo ============================================================
echo.

:: Install dependencies
echo [1/5] Installing Python dependencies...
pip install fastapi uvicorn pandas scikit-learn xgboost requests 2>nul

:: Fetch data
echo.
echo [2/5] Fetching FAERS data from FDA...
python backend\fetch_faers.py

echo.
echo [3/5] Fetching SIDER side effects data...
python backend\fetch_sider.py

echo.
echo [4/5] Building training dataset and training model...
python backend\build_dataset.py
python backend\train_model.py

:: Launch
echo.
echo [5/5] Starting servers...
echo.
start "GeneRx API" cmd /c "python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 >nul
start "GeneRx Frontend" cmd /c "cd frontend && npx -y serve -l 3000 ."

echo.
echo ============================================================
echo   Backend API:  http://localhost:8000/docs
echo   Frontend:     http://localhost:3000
echo ============================================================
echo.
pause