#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL BENCHMARKING SYSTEM
======================================

Benchmarks multiple models for resume-job description matching:
1. Trained Gemma 3 model
2. Ridge regression baseline (current system)
3. GPT-based models (if available)
4. Traditional ML approaches

Provides detailed performance metrics and comparison analysis.
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import scipy.stats as stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results from model benchmarking"""
    model_name: str
    scores: List[float]
    predictions: List[float]
    ground_truth: List[float]
    processing_times: List[float]
    accuracy_metrics: Dict[str, float]
    error_metrics: Dict[str, float]
    metadata: Dict[str, any]

class TestDataset:
    """Comprehensive test dataset for benchmarking"""
    
    def __init__(self):
        self.test_cases = self._create_comprehensive_dataset()
    
    def _create_comprehensive_dataset(self) -> List[Dict]:
        """Create comprehensive test dataset with ground truth scores"""
        return [
            {
                "id": "test_001",
                "resume": """SARAH JOHNSON
Senior Software Engineer

EXPERIENCE:
Senior Software Engineer at TechCorp (2019-2024)
- Led development of microservices architecture using Python, Django, and PostgreSQL
- Built React-based frontend applications serving 100K+ users
- Implemented CI/CD pipelines using Docker and AWS services (EC2, S3, RDS, Lambda)
- Managed team of 5 developers and mentored junior engineers
- Designed and optimized database schemas for high-performance applications
- Architected scalable systems handling 1M+ requests per day

Software Engineer at StartupXYZ (2017-2019)
- Developed full-stack web applications using Python, Flask, and JavaScript
- Worked with MySQL databases and Redis caching
- Implemented RESTful APIs and integrated third-party services
- Collaborated in Agile development environment

SKILLS:
- Programming: Python (Expert), JavaScript, TypeScript, SQL
- Frameworks: Django (Expert), Flask, React (Advanced), Node.js
- Databases: PostgreSQL (Advanced), MySQL, Redis, MongoDB
- Cloud: AWS (EC2, S3, RDS, Lambda, ECS), Docker, Kubernetes
- Leadership: Team management, mentoring, technical architecture
- Tools: Git, Jenkins, JIRA, Agile/Scrum

EDUCATION:
Master of Science in Computer Science, Stanford University (2017)
Bachelor of Science in Software Engineering, UC Berkeley (2015)""",
                
                "job_description": """Senior Full-Stack Developer
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
- Agile/Scrum methodology experience""",
                
                "ground_truth_score": 92,
                "category": "excellent",
                "reasoning": "Perfect match with 7+ years experience, expert Python/Django, React skills, PostgreSQL knowledge, comprehensive AWS experience, proven leadership, microservices architecture, advanced education."
            },
            
            {
                "id": "test_002",
                "resume": """MIKE CHEN
Full-Stack Developer

EXPERIENCE:
Full-Stack Developer at WebSolutions Inc (2021-2024)
- Developed web applications using Python, Flask, and Vue.js
- Worked with MySQL databases and implemented REST APIs
- Used Git for version control and participated in code reviews
- Collaborated with design team to implement responsive UI/UX
- Built e-commerce platforms serving 10K+ customers

Junior Developer at LocalTech (2020-2021)
- Built simple web applications using HTML, CSS, JavaScript
- Learned Python programming and basic database concepts
- Assisted senior developers with bug fixes and feature implementation

SKILLS:
- Programming: Python (Intermediate), JavaScript, HTML, CSS
- Frameworks: Flask, Vue.js, Bootstrap
- Databases: MySQL, SQLite
- Tools: Git, VS Code, Chrome DevTools
- Basic knowledge of AWS (EC2, S3)

EDUCATION:
Bachelor of Science in Computer Science, Local University (2020)""",
                
                "job_description": """Senior Full-Stack Developer
We are seeking an experienced Senior Full-Stack Developer to join our growing team.

REQUIREMENTS:
- 5+ years of experience in full-stack development
- Strong proficiency in Python and Django framework
- Experience with React and modern JavaScript
- Knowledge of PostgreSQL and database optimization
- AWS cloud services experience (EC2, S3, RDS)
- Leadership and mentoring experience
- Experience with microservices architecture
- Strong problem-solving and communication skills""",
                
                "ground_truth_score": 58,
                "category": "fair",
                "reasoning": "Moderate match with 4 years experience (below 5+ requirement), Python skills but Flask instead of Django, Vue.js instead of React, MySQL instead of PostgreSQL, basic AWS, no leadership or microservices experience."
            },
            
            {
                "id": "test_003",
                "resume": """ALEX RODRIGUEZ
Data Scientist

EXPERIENCE:
Senior Data Scientist at Analytics Corp (2020-2024)
- Developed machine learning models using Python, scikit-learn, and TensorFlow
- Analyzed large datasets using pandas, NumPy, and SQL
- Created data visualizations using matplotlib, seaborn, and Tableau
- Implemented statistical analysis and A/B testing frameworks
- Collaborated with product teams to drive data-driven decisions

Data Analyst at Research Institute (2018-2020)
- Performed statistical analysis on research data using R and Python
- Created reports and presentations for stakeholders
- Managed databases and ensured data quality

SKILLS:
- Programming: Python (Expert), R, SQL, MATLAB
- ML/AI: scikit-learn, TensorFlow, PyTorch, pandas, NumPy
- Visualization: Tableau, matplotlib, seaborn, ggplot2
- Databases: SQL Server, PostgreSQL, MongoDB
- Statistics: Hypothesis testing, regression analysis, time series

EDUCATION:
PhD in Statistics, MIT (2018)
Master of Science in Mathematics, Harvard University (2015)""",
                
                "job_description": """Senior Full-Stack Developer
We are seeking an experienced Senior Full-Stack Developer to join our growing team.

REQUIREMENTS:
- 5+ years of experience in full-stack development
- Strong proficiency in Python and Django framework
- Experience with React and modern JavaScript
- Knowledge of PostgreSQL and database optimization
- AWS cloud services experience (EC2, S3, RDS)
- Leadership and mentoring experience
- Experience with microservices architecture
- Strong problem-solving and communication skills""",
                
                "ground_truth_score": 25,
                "category": "poor",
                "reasoning": "Poor match - data scientist with no web development experience. Has Python and PostgreSQL but lacks Django, React, JavaScript, AWS, web development, and full-stack experience. Different career focus."
            },
            
            {
                "id": "test_004",
                "resume": """JENNIFER WANG
Senior Full-Stack Engineer

EXPERIENCE:
Senior Full-Stack Engineer at CloudTech Solutions (2018-2024)
- Architected and developed scalable web applications using Python, Django, and React
- Designed and optimized PostgreSQL databases for high-traffic applications
- Implemented microservices architecture on AWS using Docker and Kubernetes
- Led cross-functional team of 8 engineers and established development best practices
- Built CI/CD pipelines using Jenkins and AWS CodePipeline
- Mentored junior developers and conducted technical interviews

Full-Stack Developer at InnovateLab (2016-2018)
- Developed responsive web applications using Django REST framework and React
- Integrated third-party APIs and payment processing systems
- Optimized application performance and implemented caching strategies
- Worked in Agile environment with continuous deployment

SKILLS:
- Languages: Python (Expert), JavaScript (Advanced), TypeScript, Go, SQL
- Backend: Django (Expert), Django REST Framework, Flask, Node.js
- Frontend: React (Expert), Redux, HTML5, CSS3, SASS
- Databases: PostgreSQL (Expert), MySQL, Redis, Elasticsearch
- Cloud: AWS (EC2, S3, RDS, Lambda, ECS, CloudFormation), Docker, Kubernetes
- DevOps: Jenkins, GitLab CI, Terraform, Monitoring (Prometheus, Grafana)
- Leadership: Team management, mentoring, technical architecture decisions

EDUCATION:
Master of Science in Computer Science, Carnegie Mellon University (2016)
Bachelor of Engineering in Software Engineering, UC San Diego (2014)""",
                
                "job_description": """Senior Full-Stack Developer
We are seeking an experienced Senior Full-Stack Developer to join our growing team.

REQUIREMENTS:
- 5+ years of experience in full-stack development
- Strong proficiency in Python and Django framework
- Experience with React and modern JavaScript
- Knowledge of PostgreSQL and database optimization
- AWS cloud services experience (EC2, S3, RDS)
- Leadership and mentoring experience
- Experience with microservices architecture
- Strong problem-solving and communication skills""",
                
                "ground_truth_score": 98,
                "category": "excellent",
                "reasoning": "Perfect match with 8+ years experience, expert Python/Django skills, extensive React experience, PostgreSQL optimization expertise, comprehensive AWS knowledge, proven leadership and mentoring, microservices architecture expertise, advanced education. Exceeds all requirements."
            },
            
            {
                "id": "test_005",
                "resume": """DAVID KUMAR
Frontend Developer

EXPERIENCE:
Frontend Developer at DesignStudio (2022-2024)
- Built responsive web interfaces using React, HTML5, and CSS3
- Collaborated with UX designers to implement pixel-perfect designs
- Optimized web applications for performance and accessibility
- Used JavaScript and TypeScript for interactive features

Junior Frontend Developer at WebAgency (2021-2022)
- Developed static websites using HTML, CSS, and JavaScript
- Learned React framework and modern frontend tools
- Worked with designers to create user-friendly interfaces

SKILLS:
- Languages: JavaScript, TypeScript, HTML5, CSS3
- Frameworks: React, Vue.js, Angular (basic)
- Styling: SASS, Bootstrap, Tailwind CSS
- Tools: Git, Webpack, npm, VS Code
- Design: Figma, Adobe XD (basic)

EDUCATION:
Bachelor of Arts in Graphic Design, Art Institute (2021)""",
                
                "job_description": """Senior Full-Stack Developer
We are seeking an experienced Senior Full-Stack Developer to join our growing team.

REQUIREMENTS:
- 5+ years of experience in full-stack development
- Strong proficiency in Python and Django framework
- Experience with React and modern JavaScript
- Knowledge of PostgreSQL and database optimization
- AWS cloud services experience (EC2, S3, RDS)
- Leadership and mentoring experience
- Experience with microservices architecture
- Strong problem-solving and communication skills""",
                
                "ground_truth_score": 35,
                "category": "poor",
                "reasoning": "Poor match - frontend-only developer with 3 years experience (below 5+ requirement). Has React and JavaScript but completely lacks Python, Django, PostgreSQL, AWS, backend development, leadership, and microservices experience."
            },
            
            {
                "id": "test_006",
                "resume": """LISA THOMPSON
Software Engineer

EXPERIENCE:
Software Engineer at TechStartup (2020-2024)
- Developed web applications using Python, Django, and JavaScript
- Built REST APIs and integrated with PostgreSQL databases
- Worked with React for frontend development
- Deployed applications on AWS using EC2 and S3
- Participated in code reviews and Agile development process

Software Developer Intern at BigCorp (2019-2020)
- Assisted with web development projects using Python and Django
- Learned database design and SQL optimization
- Contributed to team projects and documentation

SKILLS:
- Programming: Python, JavaScript, SQL, HTML, CSS
- Frameworks: Django, React, Flask (basic)
- Databases: PostgreSQL, MySQL
- Cloud: AWS (EC2, S3, basic RDS)
- Tools: Git, Docker (basic), Jenkins (basic)
- Methodologies: Agile, Scrum

EDUCATION:
Bachelor of Science in Computer Science, State University (2019)""",
                
                "job_description": """Senior Full-Stack Developer
We are seeking an experienced Senior Full-Stack Developer to join our growing team.

REQUIREMENTS:
- 5+ years of experience in full-stack development
- Strong proficiency in Python and Django framework
- Experience with React and modern JavaScript
- Knowledge of PostgreSQL and database optimization
- AWS cloud services experience (EC2, S3, RDS)
- Leadership and mentoring experience
- Experience with microservices architecture
- Strong problem-solving and communication skills""",
                
                "ground_truth_score": 72,
                "category": "good",
                "reasoning": "Good match with 4+ years experience (close to 5+ requirement), solid Python/Django skills, React experience, PostgreSQL knowledge, basic AWS experience. Missing leadership, microservices experience, and advanced AWS skills."
            }
        ]
    
    def get_test_cases(self) -> List[Dict]:
        """Get all test cases"""
        return self.test_cases
    
    def get_ground_truth_scores(self) -> List[float]:
        """Get ground truth scores"""
        return [case["ground_truth_score"] for case in self.test_cases]

class ModelBenchmarker:
    """Main benchmarking class"""
    
    def __init__(self):
        self.test_dataset = TestDataset()
        self.results = {}
        
    def benchmark_gemma3_api(self, api_url: str = "http://localhost:8006") -> BenchmarkResult:
        """Benchmark Gemma 3 API"""
        logger.info("Benchmarking Gemma 3 API...")
        
        test_cases = self.test_dataset.get_test_cases()
        predictions = []
        processing_times = []
        
        for case in test_cases:
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{api_url}/score",
                    json={
                        "resume_text": case["resume"],
                        "jd_text": case["job_description"],
                        "model_version": "gemma3"
                    },
                    timeout=30
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    predictions.append(result["score"])
                    processing_times.append(processing_time)
                else:
                    logger.error(f"API error for case {case['id']}: {response.status_code}")
                    predictions.append(50.0)  # Default score
                    processing_times.append(processing_time)
                    
            except Exception as e:
                logger.error(f"Error testing case {case['id']}: {e}")
                predictions.append(50.0)  # Default score
                processing_times.append(1.0)  # Default time
        
        ground_truth = self.test_dataset.get_ground_truth_scores()
        
        return BenchmarkResult(
            model_name="Gemma 3 API",
            scores=predictions,
            predictions=predictions,
            ground_truth=ground_truth,
            processing_times=processing_times,
            accuracy_metrics=self._calculate_accuracy_metrics(predictions, ground_truth),
            error_metrics=self._calculate_error_metrics(predictions, ground_truth),
            metadata={"api_url": api_url, "model_type": "transformer"}
        )
    
    def benchmark_ridge_baseline(self, api_url: str = "http://localhost:8005") -> BenchmarkResult:
        """Benchmark Ridge regression baseline (current system)"""
        logger.info("Benchmarking Ridge regression baseline...")
        
        test_cases = self.test_dataset.get_test_cases()
        predictions = []
        processing_times = []
        
        for case in test_cases:
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{api_url}/score",
                    json={
                        "resume_text": case["resume"],
                        "jd_text": case["job_description"]
                    },
                    timeout=30
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    predictions.append(result["score"])
                    processing_times.append(processing_time)
                else:
                    logger.error(f"API error for case {case['id']}: {response.status_code}")
                    predictions.append(50.0)
                    processing_times.append(processing_time)
                    
            except Exception as e:
                logger.error(f"Error testing case {case['id']}: {e}")
                predictions.append(50.0)
                processing_times.append(1.0)
        
        ground_truth = self.test_dataset.get_ground_truth_scores()
        
        return BenchmarkResult(
            model_name="Ridge Baseline",
            scores=predictions,
            predictions=predictions,
            ground_truth=ground_truth,
            processing_times=processing_times,
            accuracy_metrics=self._calculate_accuracy_metrics(predictions, ground_truth),
            error_metrics=self._calculate_error_metrics(predictions, ground_truth),
            metadata={"api_url": api_url, "model_type": "traditional_ml"}
        )
    
    def _calculate_accuracy_metrics(self, predictions: List[float], ground_truth: List[float]) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        # Convert to categorical for classification metrics
        pred_categories = [self._score_to_category(score) for score in predictions]
        true_categories = [self._score_to_category(score) for score in ground_truth]
        
        # Classification accuracy
        accuracy = accuracy_score(true_categories, pred_categories)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_categories, pred_categories, average='weighted', zero_division=0
        )
        
        # Correlation
        correlation = np.corrcoef(predictions, ground_truth)[0, 1] if len(predictions) > 1 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "correlation": correlation
        }
    
    def _calculate_error_metrics(self, predictions: List[float], ground_truth: List[float]) -> Dict[str, float]:
        """Calculate error metrics"""
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(ground_truth, predictions)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((np.array(ground_truth) - np.array(predictions)) / np.array(ground_truth))) * 100
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "mape": mape
        }
    
    def _score_to_category(self, score: float) -> str:
        """Convert score to category"""
        if score >= 80:
            return "excellent"
        elif score >= 65:
            return "good"
        elif score >= 45:
            return "fair"
        else:
            return "poor"
    
    def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive benchmark of all models"""
        logger.info("Starting comprehensive model benchmarking...")
        
        results = {}
        
        # Benchmark Gemma 3
        try:
            results["gemma3"] = self.benchmark_gemma3_api()
            logger.info("✅ Gemma 3 benchmarking completed")
        except Exception as e:
            logger.error(f"❌ Gemma 3 benchmarking failed: {e}")
        
        # Benchmark Ridge baseline
        try:
            results["ridge"] = self.benchmark_ridge_baseline()
            logger.info("✅ Ridge baseline benchmarking completed")
        except Exception as e:
            logger.error(f"❌ Ridge baseline benchmarking failed: {e}")
        
        self.results = results
        return results
    
    def generate_comparison_report(self, results: Dict[str, BenchmarkResult]) -> Dict:
        """Generate comprehensive comparison report"""
        report = {
            "benchmark_date": datetime.now().isoformat(),
            "test_dataset_size": len(self.test_dataset.get_test_cases()),
            "models_compared": list(results.keys()),
            "detailed_results": {},
            "summary": {},
            "recommendations": []
        }
        
        # Detailed results for each model
        for model_name, result in results.items():
            report["detailed_results"][model_name] = {
                "accuracy_metrics": result.accuracy_metrics,
                "error_metrics": result.error_metrics,
                "avg_processing_time": np.mean(result.processing_times),
                "predictions": result.predictions,
                "metadata": result.metadata
            }
        
        # Summary comparison
        if len(results) > 1:
            best_accuracy = max(results.values(), key=lambda x: x.accuracy_metrics["accuracy"])
            best_f1 = max(results.values(), key=lambda x: x.accuracy_metrics["f1_score"])
            best_correlation = max(results.values(), key=lambda x: x.accuracy_metrics["correlation"])
            fastest = min(results.values(), key=lambda x: np.mean(x.processing_times))
            
            report["summary"] = {
                "best_accuracy": {"model": best_accuracy.model_name, "score": best_accuracy.accuracy_metrics["accuracy"]},
                "best_f1": {"model": best_f1.model_name, "score": best_f1.accuracy_metrics["f1_score"]},
                "best_correlation": {"model": best_correlation.model_name, "score": best_correlation.accuracy_metrics["correlation"]},
                "fastest": {"model": fastest.model_name, "time": np.mean(fastest.processing_times)}
            }
        
        # Generate recommendations
        if "gemma3" in results and "ridge" in results:
            gemma3_result = results["gemma3"]
            ridge_result = results["ridge"]
            
            if gemma3_result.accuracy_metrics["accuracy"] > ridge_result.accuracy_metrics["accuracy"]:
                report["recommendations"].append("Gemma 3 shows superior accuracy compared to Ridge baseline")
            
            if gemma3_result.accuracy_metrics["correlation"] > ridge_result.accuracy_metrics["correlation"]:
                report["recommendations"].append("Gemma 3 demonstrates better correlation with ground truth scores")
            
            if np.mean(gemma3_result.processing_times) < np.mean(ridge_result.processing_times):
                report["recommendations"].append("Gemma 3 provides faster inference times")
            else:
                report["recommendations"].append("Ridge baseline offers faster processing for real-time applications")
        
        return report
    
    def save_results(self, results: Dict[str, BenchmarkResult], report: Dict, output_dir: str = "../benchmark_results"):
        """Save benchmark results and report"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed report
        report_file = f"{output_dir}/benchmark_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save CSV with predictions
        df_data = []
        ground_truth = self.test_dataset.get_ground_truth_scores()
        
        for i, case in enumerate(self.test_dataset.get_test_cases()):
            row = {
                "test_id": case["id"],
                "ground_truth": ground_truth[i],
                "category": case["category"]
            }
            
            for model_name, result in results.items():
                row[f"{model_name}_prediction"] = result.predictions[i]
                row[f"{model_name}_processing_time"] = result.processing_times[i]
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_file = f"{output_dir}/benchmark_predictions_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to: {report_file} and {csv_file}")
        return report_file, csv_file

def main():
    """Main benchmarking function"""
    print("🚀 COMPREHENSIVE MODEL BENCHMARKING")
    print("=" * 60)
    print("Comparing:")
    print("• Trained Gemma 3 model")
    print("• Ridge regression baseline")
    print("• Performance metrics and analysis")
    print()
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker()
    
    # Run comprehensive benchmark
    results = benchmarker.run_comprehensive_benchmark()
    
    if not results:
        print("❌ No models could be benchmarked. Please ensure APIs are running.")
        return False
    
    # Generate comparison report
    report = benchmarker.generate_comparison_report(results)
    
    # Save results
    report_file, csv_file = benchmarker.save_results(results, report)
    
    # Print summary
    print("\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 40)
    
    for model_name, result in results.items():
        print(f"\n🤖 {model_name}:")
        print(f"   Accuracy: {result.accuracy_metrics['accuracy']:.3f}")
        print(f"   F1 Score: {result.accuracy_metrics['f1_score']:.3f}")
        print(f"   Correlation: {result.accuracy_metrics['correlation']:.3f}")
        print(f"   RMSE: {result.error_metrics['rmse']:.2f}")
        print(f"   Avg Processing Time: {np.mean(result.processing_times):.3f}s")
    
    if "summary" in report:
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Accuracy: {report['summary']['best_accuracy']['model']} ({report['summary']['best_accuracy']['score']:.3f})")
        print(f"   F1 Score: {report['summary']['best_f1']['model']} ({report['summary']['best_f1']['score']:.3f})")
        print(f"   Correlation: {report['summary']['best_correlation']['model']} ({report['summary']['best_correlation']['score']:.3f})")
        print(f"   Speed: {report['summary']['fastest']['model']} ({report['summary']['fastest']['time']:.3f}s)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report.get("recommendations", []):
        print(f"   • {rec}")
    
    print(f"\n📁 Detailed results saved to:")
    print(f"   • {report_file}")
    print(f"   • {csv_file}")
    
    return True

if __name__ == "__main__":
    main()
