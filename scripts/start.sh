#!/bin/bash
echo "🚀 Starting Gemma 3 Resume Scorer System..."
echo "=" * 50

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install -r requirements_gemma3.txt

# Start the system
echo "🚀 Deploying Gemma 3 system..."
python deployment/deploy_gemma3_complete.py

echo ""
echo "✅ System started successfully!"
echo "🌐 Access the web interface at: http://localhost:8006/"
echo "📚 API documentation at: http://localhost:8006/docs"
echo ""
echo "Press Ctrl+C to stop the server"
