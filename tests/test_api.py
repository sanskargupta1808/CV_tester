#!/usr/bin/env python3
"""
Comprehensive test suite for Gemma 3 Resume Scorer API
"""

import requests
import json
import time
import os
from pathlib import Path

class Gemma3APITester:
    """Test suite for Gemma 3 API"""
    
    def __init__(self, base_url="http://localhost:8006"):
        self.base_url = base_url
        self.test_results = []
    
    def log_test(self, test_name, success, details=""):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
    
    def test_health_endpoint(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                details = f"Status: {data.get('status')}, Model: {data.get('model_loaded')}"
                self.log_test("Health Endpoint", True, details)
                return True
            else:
                self.log_test("Health Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Health Endpoint", False, str(e))
            return False
    
    def test_score_endpoint(self):
        """Test scoring endpoint with text input"""
        try:
            test_data = {
                "resume_text": "Software Engineer with 5 years Python experience, Django framework, React frontend development, PostgreSQL databases, AWS cloud services, team leadership experience.",
                "jd_text": "Senior Software Engineer position requiring Python, Django, React, PostgreSQL, AWS experience. 5+ years required. Leadership and mentoring experience preferred."
            }
            
            response = requests.post(
                f"{self.base_url}/score", 
                json=test_data, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                score = result.get("score", 0)
                category = result.get("category", "Unknown")
                processing_time = result.get("processing_time", 0)
                
                details = f"Score: {score:.1f}/100, Category: {category}, Time: {processing_time*1000:.1f}ms"
                self.log_test("Score Endpoint", True, details)
                return True
            else:
                self.log_test("Score Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Score Endpoint", False, str(e))
            return False
    
    def test_file_upload(self):
        """Test file upload endpoint"""
        try:
            # Create test resume file
            test_resume = """JANE SMITH
Senior Full-Stack Developer

EXPERIENCE:
Senior Full-Stack Developer at TechCorp (2019-2024)
- Led development of microservices using Python, Django, and React
- Managed PostgreSQL databases and AWS infrastructure
- Mentored team of 6 developers
- Implemented CI/CD pipelines and DevOps practices

Full-Stack Developer at StartupXYZ (2017-2019)
- Built scalable web applications using modern technologies
- Worked with cross-functional teams in Agile environment

SKILLS:
- Languages: Python (Expert), JavaScript, TypeScript, SQL
- Frameworks: Django, React, Flask, Node.js
- Databases: PostgreSQL, MySQL, Redis
- Cloud: AWS (EC2, S3, RDS, Lambda), Docker, Kubernetes
- Leadership: Team management, mentoring, architecture decisions

EDUCATION:
Master of Science in Computer Science, MIT (2017)"""

            job_description = "Senior Full-Stack Developer position requiring Python, Django, React, PostgreSQL, AWS experience. 5+ years required. Leadership and mentoring experience preferred."
            
            # Test with text file
            files = {'file': ('test_resume.txt', test_resume, 'text/plain')}
            data = {'job_description': job_description}
            
            response = requests.post(
                f"{self.base_url}/upload-score",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                score = result.get("score", 0)
                file_info = result.get("file_info", {})
                filename = file_info.get("filename", "Unknown")
                
                details = f"Score: {score:.1f}/100, File: {filename}"
                self.log_test("File Upload", True, details)
                return True
            else:
                self.log_test("File Upload", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("File Upload", False, str(e))
            return False
    
    def test_web_interface(self):
        """Test web interface accessibility"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            
            if response.status_code == 200:
                content = response.text
                
                # Check for key elements
                checks = [
                    ("Gemma 3" in content, "Gemma 3 branding"),
                    ("upload" in content.lower(), "Upload functionality"),
                    ("analyze" in content.lower(), "Analysis functionality"),
                    ("<script>" in content, "JavaScript present")
                ]
                
                passed_checks = sum(1 for check, _ in checks if check)
                total_checks = len(checks)
                
                details = f"UI checks: {passed_checks}/{total_checks} passed"
                self.log_test("Web Interface", passed_checks == total_checks, details)
                return passed_checks == total_checks
            else:
                self.log_test("Web Interface", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Web Interface", False, str(e))
            return False
    
    def test_api_documentation(self):
        """Test API documentation endpoint"""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=10)
            
            if response.status_code == 200:
                self.log_test("API Documentation", True, "Swagger docs accessible")
                return True
            else:
                self.log_test("API Documentation", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("API Documentation", False, str(e))
            return False
    
    def test_error_handling(self):
        """Test API error handling"""
        try:
            # Test with invalid data
            invalid_data = {
                "resume_text": "",  # Empty resume
                "jd_text": "Short JD"  # Too short job description
            }
            
            response = requests.post(
                f"{self.base_url}/score",
                json=invalid_data,
                timeout=10
            )
            
            # Should return 400 error for invalid input
            if response.status_code == 400:
                self.log_test("Error Handling", True, "Properly handles invalid input")
                return True
            else:
                self.log_test("Error Handling", False, f"Expected 400, got {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Error Handling", False, str(e))
            return False
    
    def test_performance(self):
        """Test API performance"""
        try:
            test_data = {
                "resume_text": "Software Engineer with Python and JavaScript experience",
                "jd_text": "Looking for Software Engineer with Python, JavaScript, and web development experience. Must have 3+ years of experience."
            }
            
            # Run multiple requests to test performance
            times = []
            for _ in range(5):
                start_time = time.time()
                response = requests.post(f"{self.base_url}/score", json=test_data, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
                else:
                    self.log_test("Performance Test", False, f"Request failed: {response.status_code}")
                    return False
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            
            # Performance criteria: average < 5s, max < 10s
            performance_ok = avg_time < 5.0 and max_time < 10.0
            
            details = f"Avg: {avg_time:.2f}s, Max: {max_time:.2f}s"
            self.log_test("Performance Test", performance_ok, details)
            return performance_ok
            
        except Exception as e:
            self.log_test("Performance Test", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 GEMMA 3 API TEST SUITE")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Score Endpoint", self.test_score_endpoint),
            ("File Upload", self.test_file_upload),
            ("Web Interface", self.test_web_interface),
            ("API Documentation", self.test_api_documentation),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            success = test_func()
            if success:
                passed += 1
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{status} {result['test']}")
            if result["details"]:
                print(f"     {result['details']}")
        
        print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! System is fully functional.")
        elif passed >= total * 0.8:
            print("✅ MOSTLY SUCCESSFUL! Minor issues detected.")
        else:
            print("⚠️ MULTIPLE FAILURES! System needs attention.")
        
        # Save results
        self.save_test_results()
        
        return passed == total
    
    def save_test_results(self):
        """Save test results to file"""
        results_file = f"test_results_{int(time.time())}.json"
        
        summary = {
            "timestamp": time.time(),
            "total_tests": len(self.test_results),
            "passed_tests": sum(1 for r in self.test_results if r["success"]),
            "test_details": self.test_results
        }
        
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Test results saved to: {results_file}")

def main():
    """Main test function"""
    tester = Gemma3APITester()
    success = tester.run_all_tests()
    
    return success

if __name__ == "__main__":
    main()
