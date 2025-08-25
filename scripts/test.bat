@echo off
echo 🧪 Testing Gemma 3 Resume Scorer System...
echo ==================================================

REM Activate virtual environment
if exist "venv" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo ❌ Virtual environment not found. Run setup.py first.
    pause
    exit /b 1
)

REM Run API tests
echo 🧪 Running API tests...
python tests\test_api.py

echo.
echo ✅ Testing completed!
pause
