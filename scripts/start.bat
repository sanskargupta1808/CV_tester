@echo off
echo 🚀 Starting Gemma 3 Resume Scorer System...
echo ==================================================

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

REM Install/update dependencies
echo 📚 Installing dependencies...
pip install -r requirements_gemma3.txt

REM Start the system
echo 🚀 Deploying Gemma 3 system...
python deployment\deploy_gemma3_complete.py

echo.
echo ✅ System started successfully!
echo 🌐 Access the web interface at: http://localhost:8006/
echo 📚 API documentation at: http://localhost:8006/docs
echo.
echo Press Ctrl+C to stop the server
pause
