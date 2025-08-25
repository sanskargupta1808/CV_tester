#!/bin/bash
echo "🧪 Testing Gemma 3 Resume Scorer System..."
echo "=" * 50

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run setup.py first."
    exit 1
fi

# Run API tests
echo "🧪 Running API tests..."
python tests/test_api.py

echo ""
echo "✅ Testing completed!"
