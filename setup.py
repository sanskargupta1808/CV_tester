#!/usr/bin/env python3
"""
GEMMA 3 RESUME SCORER - SETUP SCRIPT
===================================

Easy setup script for the complete Gemma 3 Resume Scorer system.
This script handles environment setup, dependency installation, and initial deployment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("🚀 GEMMA 3 RESUME SCORER - SETUP")
    print("=" * 50)
    print("Setting up the complete AI-powered resume scoring system")
    print("Based on Google's Gemma 3 model with LoRA fine-tuning")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n📦 Setting up virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies"""
    print("\n📚 Installing dependencies...")
    
    # Determine pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    try:
        # Upgrade pip first
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements_gemma3.txt"], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\n📄 Creating sample data...")
    
    sample_resume = """JOHN DOE
Senior Software Engineer

EXPERIENCE:
Senior Software Engineer at TechCorp (2019-2024)
- Developed scalable web applications using Python, Django, and React
- Led team of 5 developers and mentored junior engineers
- Implemented microservices architecture on AWS
- Optimized PostgreSQL databases for high-performance applications

Software Engineer at StartupXYZ (2017-2019)
- Built full-stack applications using Python and JavaScript
- Worked with MySQL databases and Redis caching
- Implemented RESTful APIs and CI/CD pipelines

SKILLS:
- Programming: Python (Expert), JavaScript, TypeScript, SQL
- Frameworks: Django, React, Flask, Node.js
- Databases: PostgreSQL, MySQL, Redis, MongoDB
- Cloud: AWS (EC2, S3, RDS, Lambda), Docker, Kubernetes
- Leadership: Team management, mentoring, technical architecture

EDUCATION:
Master of Science in Computer Science, Stanford University (2017)
Bachelor of Science in Software Engineering, UC Berkeley (2015)"""

    sample_jd = """Senior Full-Stack Developer
We are seeking an experienced Senior Full-Stack Developer to join our growing team.

REQUIREMENTS:
- 5+ years of experience in full-stack development
- Strong proficiency in Python and Django framework
- Experience with React and modern JavaScript
- Knowledge of PostgreSQL and database optimization
- AWS cloud services experience (EC2, S3, RDS)
- Leadership and mentoring experience
- Experience with microservices architecture
- Strong problem-solving and communication skills

PREFERRED:
- Master's degree in Computer Science or related field
- Experience with Docker and Kubernetes
- CI/CD pipeline implementation experience
- Agile/Scrum methodology experience"""

    # Create data directory and sample files
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "sample_resume.txt", "w") as f:
        f.write(sample_resume)
    
    with open(data_dir / "sample_job_description.txt", "w") as f:
        f.write(sample_jd)
    
    print("✅ Sample data created")
    return True

def create_test_scripts():
    """Create test scripts"""
    print("\n🧪 Creating test scripts...")
    
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)
    
    # Create API test script
    api_test = '''#!/usr/bin/env python3
"""Test script for Gemma 3 API"""
import requests
import json

def test_api():
    """Test the Gemma 3 API"""
    print("🧪 Testing Gemma 3 API...")
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8006/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test scoring endpoint
    try:
        data = {
            "resume_text": "Software Engineer with Python experience",
            "jd_text": "Looking for Software Engineer with Python and web development experience. 3+ years required."
        }
        
        response = requests.post("http://localhost:8006/score", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Scoring test passed: {result['score']:.1f}/100")
        else:
            print("❌ Scoring test failed")
    except Exception as e:
        print(f"❌ Scoring test error: {e}")

if __name__ == "__main__":
    test_api()
'''
    
    with open(test_dir / "test_api.py", "w") as f:
        f.write(api_test)
    
    print("✅ Test scripts created")
    return True

def create_run_scripts():
    """Create convenient run scripts"""
    print("\n📜 Creating run scripts...")
    
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Create start script
    if platform.system() == "Windows":
        start_script = '''@echo off
echo Starting Gemma 3 Resume Scorer...
call venv\\Scripts\\activate
python deployment\\deploy_gemma3_complete.py
pause
'''
        with open(scripts_dir / "start.bat", "w") as f:
            f.write(start_script)
    else:
        start_script = '''#!/bin/bash
echo "Starting Gemma 3 Resume Scorer..."
source venv/bin/activate
python deployment/deploy_gemma3_complete.py
'''
        with open(scripts_dir / "start.sh", "w") as f:
            f.write(start_script)
        os.chmod(scripts_dir / "start.sh", 0o755)
    
    # Create test script
    if platform.system() == "Windows":
        test_script = '''@echo off
echo Testing Gemma 3 System...
call venv\\Scripts\\activate
python tests\\test_api.py
pause
'''
        with open(scripts_dir / "test.bat", "w") as f:
            f.write(test_script)
    else:
        test_script = '''#!/bin/bash
echo "Testing Gemma 3 System..."
source venv/bin/activate
python tests/test_api.py
'''
        with open(scripts_dir / "test.sh", "w") as f:
            f.write(test_script)
        os.chmod(scripts_dir / "test.sh", 0o755)
    
    print("✅ Run scripts created")
    return True

def print_completion_message():
    """Print setup completion message"""
    activation_cmd = get_activation_command()
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"\n🚀 To start the Gemma 3 Resume Scorer:")
    if platform.system() == "Windows":
        print(f"   • Double-click: scripts\\start.bat")
        print(f"   • Or manually: {activation_cmd} && python deployment\\deploy_gemma3_complete.py")
    else:
        print(f"   • Run: ./scripts/start.sh")
        print(f"   • Or manually: {activation_cmd} && python deployment/deploy_gemma3_complete.py")
    
    print(f"\n🧪 To test the system:")
    if platform.system() == "Windows":
        print(f"   • Double-click: scripts\\test.bat")
        print(f"   • Or manually: {activation_cmd} && python tests\\test_api.py")
    else:
        print(f"   • Run: ./scripts/test.sh")
        print(f"   • Or manually: {activation_cmd} && python tests/test_api.py")
    
    print(f"\n🌐 Once running, access:")
    print(f"   • Web Interface: http://localhost:8006/")
    print(f"   • API Docs: http://localhost:8006/docs")
    print(f"   • Health Check: http://localhost:8006/health")
    
    print(f"\n📁 Project Structure:")
    print(f"   • Backend: backend/gemma3_api.py")
    print(f"   • Frontend: frontend/gemma3_interface.html")
    print(f"   • Training: training/gemma3_trainer.py")
    print(f"   • Tests: tests/test_api.py")
    print(f"   • Sample Data: data/")
    
    print(f"\n📚 Documentation:")
    print(f"   • README.md - Complete system documentation")
    print(f"   • requirements_gemma3.txt - Python dependencies")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create sample data
    if not create_sample_data():
        return False
    
    # Create test scripts
    if not create_test_scripts():
        return False
    
    # Create run scripts
    if not create_run_scripts():
        return False
    
    # Print completion message
    print_completion_message()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
