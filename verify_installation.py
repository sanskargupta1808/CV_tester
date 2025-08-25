#!/usr/bin/env python3
"""
Installation Verification Script for Gemma 3 Resume Scorer
Checks that all necessary files and components are present
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and report status"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ MISSING {description}: {file_path}")
        return False

def check_directory_exists(dir_path, description):
    """Check if a directory exists and report status"""
    if Path(dir_path).is_dir():
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"❌ MISSING {description}: {dir_path}")
        return False

def verify_installation():
    """Verify complete installation"""
    print("🔍 GEMMA 3 RESUME SCORER - INSTALLATION VERIFICATION")
    print("=" * 60)
    
    all_good = True
    
    # Core documentation
    print("\n📚 Core Documentation:")
    all_good &= check_file_exists("README.md", "Main README")
    all_good &= check_file_exists("PROJECT_STRUCTURE.md", "Project Structure")
    all_good &= check_file_exists("SYSTEM_COMPLETE.md", "System Complete Guide")
    all_good &= check_file_exists("requirements_gemma3.txt", "Requirements File")
    all_good &= check_file_exists("setup.py", "Setup Script")
    
    # Backend system
    print("\n🖥️ Backend System:")
    all_good &= check_directory_exists("backend", "Backend Directory")
    all_good &= check_file_exists("backend/gemma3_api.py", "Main API Server")
    
    # Frontend system
    print("\n🌐 Frontend System:")
    all_good &= check_directory_exists("frontend", "Frontend Directory")
    all_good &= check_file_exists("frontend/gemma3_interface.html", "Web Interface")
    
    # Training system
    print("\n🧠 Training System:")
    all_good &= check_directory_exists("training", "Training Directory")
    all_good &= check_file_exists("training/gemma3_trainer.py", "Model Trainer")
    all_good &= check_file_exists("training/benchmark_models.py", "Benchmarking System")
    
    # Models and data
    print("\n🤖 Models & Data:")
    all_good &= check_directory_exists("models", "Models Directory")
    all_good &= check_directory_exists("data", "Data Directory")
    all_good &= check_file_exists("data/sample_resume.txt", "Sample Resume")
    all_good &= check_file_exists("data/sample_job_description.txt", "Sample Job Description")
    
    # Testing
    print("\n🧪 Testing System:")
    all_good &= check_directory_exists("tests", "Tests Directory")
    all_good &= check_file_exists("tests/test_api.py", "API Test Suite")
    
    # Deployment
    print("\n🚀 Deployment System:")
    all_good &= check_directory_exists("deployment", "Deployment Directory")
    all_good &= check_file_exists("deployment/deploy_gemma3_complete.py", "Deployment Script")
    
    # Scripts
    print("\n📜 Utility Scripts:")
    all_good &= check_directory_exists("scripts", "Scripts Directory")
    all_good &= check_file_exists("scripts/start.sh", "Linux/Mac Start Script")
    all_good &= check_file_exists("scripts/start.bat", "Windows Start Script")
    all_good &= check_file_exists("scripts/test.sh", "Linux/Mac Test Script")
    all_good &= check_file_exists("scripts/test.bat", "Windows Test Script")
    
    # Documentation
    print("\n📖 Documentation:")
    all_good &= check_directory_exists("docs", "Documentation Directory")
    all_good &= check_file_exists("docs/deployment_guide.md", "Deployment Guide")
    all_good &= check_file_exists("docs/training_guide.md", "Training Guide")
    all_good &= check_file_exists("docs/benchmarking_guide.md", "Benchmarking Guide")
    
    # Virtual environment
    print("\n🐍 Python Environment:")
    all_good &= check_directory_exists("venv", "Virtual Environment")
    
    # Results
    print("\n📊 Results & Reports:")
    all_good &= check_directory_exists("benchmark_results", "Benchmark Results Directory")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    if all_good:
        print("🎉 INSTALLATION COMPLETE!")
        print("✅ All necessary files and folders are present")
        print("✅ System is ready for use")
        
        print(f"\n🚀 Quick Start:")
        print(f"   1. Run: python setup.py")
        print(f"   2. Run: ./scripts/start.sh (Linux/Mac) or scripts\\start.bat (Windows)")
        print(f"   3. Open: http://localhost:8006/")
        
    else:
        print("❌ INSTALLATION INCOMPLETE!")
        print("⚠️ Some files or folders are missing")
        print("Please check the missing items above")
    
    return all_good

def main():
    """Main verification function"""
    return verify_installation()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
