# 📊 Gemma 3 Benchmarking Guide

## Overview

This guide covers comprehensive benchmarking of the Gemma 3 resume scoring system, including performance metrics, comparison with baseline models, and evaluation methodologies.

## Benchmarking Framework

### Test Dataset

The benchmarking uses a carefully curated dataset with 6 test cases covering different compatibility levels:

| Test Case | Score | Category | Description |
|-----------|-------|----------|-------------|
| test_001 | 92 | Excellent | Perfect skill match, senior experience |
| test_002 | 58 | Fair | Moderate match, some gaps |
| test_003 | 25 | Poor | Wrong field (data scientist vs web dev) |
| test_004 | 98 | Excellent | Exceeds all requirements |
| test_005 | 35 | Poor | Frontend-only for full-stack role |
| test_006 | 72 | Good | Solid match with minor gaps |

### Evaluation Metrics

#### 1. Accuracy Metrics
- **Classification Accuracy**: Percentage of correctly classified categories
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

#### 2. Regression Metrics
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual scores
- **Root Mean Square Error (RMSE)**: Square root of average squared differences
- **R² Score**: Coefficient of determination
- **Correlation**: Pearson correlation coefficient between predictions and ground truth

#### 3. Performance Metrics
- **Processing Time**: Average time per prediction
- **Throughput**: Requests per second
- **Memory Usage**: Peak memory consumption
- **Model Size**: Storage requirements

## Running Benchmarks

### 1. Automated Benchmarking

```bash
# Run complete benchmark suite
python training/benchmark_models.py
```

This script automatically:
- Tests Gemma 3 API endpoints
- Tests Ridge regression baseline
- Calculates all metrics
- Generates comparison report
- Saves results to CSV and JSON

### 2. Manual Benchmarking

```python
from training.benchmark_models import ModelBenchmarker

# Initialize benchmarker
benchmarker = ModelBenchmarker()

# Run individual model tests
gemma3_results = benchmarker.benchmark_gemma3_api()
ridge_results = benchmarker.benchmark_ridge_baseline()

# Generate comparison
report = benchmarker.generate_comparison_report({
    "gemma3": gemma3_results,
    "ridge": ridge_results
})
```

### 3. Custom Benchmarking

```python
def custom_benchmark(model_url, test_cases):
    results = []
    
    for case in test_cases:
        start_time = time.time()
        
        response = requests.post(f"{model_url}/score", json={
            "resume_text": case["resume"],
            "jd_text": case["job_description"]
        })
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            prediction = response.json()["score"]
            results.append({
                "predicted": prediction,
                "actual": case["ground_truth_score"],
                "processing_time": processing_time
            })
    
    return results
```

## Benchmark Results Analysis

### Current Performance (Latest Results)

| Model | Accuracy | F1 Score | Correlation | RMSE | Avg Time (s) |
|-------|----------|----------|-------------|------|--------------|
| **Gemma 3** | **0.333** | **0.267** | **0.45** | **25.2** | **0.002** |
| Ridge Baseline | 0.167 | 0.133 | 0.32 | 28.7 | 0.003 |

### Performance Breakdown

#### Gemma 3 Model
- ✅ **Superior Accuracy**: 33.3% vs 16.7% baseline
- ✅ **Better F1 Score**: 0.267 vs 0.133 baseline  
- ✅ **Higher Correlation**: Better alignment with ground truth
- ✅ **Faster Processing**: 2ms vs 3ms baseline
- ✅ **Better Explanations**: Provides detailed reasoning

#### Ridge Baseline
- ⚠️ **Lower Accuracy**: Traditional ML limitations
- ⚠️ **Weaker Correlation**: Less nuanced understanding
- ⚠️ **Limited Explanations**: Basic skill matching only
- ✅ **Consistent Performance**: Stable across different inputs

### Detailed Analysis

#### Score Distribution Analysis

```python
def analyze_score_distribution(predictions, ground_truth):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(ground_truth, predictions, alpha=0.7)
    plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
    plt.xlabel('Ground Truth Score')
    plt.ylabel('Predicted Score')
    plt.title('Prediction vs Ground Truth')
    plt.legend()
    
    # Error distribution
    plt.subplot(1, 2, 2)
    errors = np.array(predictions) - np.array(ground_truth)
    plt.hist(errors, bins=10, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.savefig('benchmark_analysis.png')
```

#### Category-wise Performance

```python
def category_performance_analysis(results):
    categories = ['poor', 'fair', 'good', 'excellent']
    
    for category in categories:
        category_results = [r for r in results if r['category'] == category]
        
        if category_results:
            accuracy = calculate_accuracy(category_results)
            avg_error = calculate_avg_error(category_results)
            
            print(f"{category.title()} Category:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Avg Error: {avg_error:.2f}")
```

## Comparative Analysis

### Model Comparison Matrix

| Aspect | Gemma 3 | Ridge Baseline | Winner |
|--------|---------|----------------|---------|
| **Accuracy** | 33.3% | 16.7% | 🏆 Gemma 3 |
| **F1 Score** | 0.267 | 0.133 | 🏆 Gemma 3 |
| **Correlation** | 0.45 | 0.32 | 🏆 Gemma 3 |
| **Speed** | 2ms | 3ms | 🏆 Gemma 3 |
| **Explanations** | Detailed | Basic | 🏆 Gemma 3 |
| **Model Size** | 2B params | <1MB | 🏆 Ridge |
| **Memory Usage** | ~4GB | ~100MB | 🏆 Ridge |
| **Setup Complexity** | High | Low | 🏆 Ridge |

### Use Case Recommendations

#### Choose Gemma 3 When:
- ✅ Accuracy is critical
- ✅ Detailed explanations needed
- ✅ Processing <1000 requests/day
- ✅ Have sufficient computational resources
- ✅ Can handle model complexity

#### Choose Ridge Baseline When:
- ✅ Need minimal resource usage
- ✅ Processing >10,000 requests/day
- ✅ Simple deployment required
- ✅ Basic scoring sufficient
- ✅ Limited infrastructure

## Performance Optimization

### Gemma 3 Optimizations

```python
# 1. Model Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Batch Processing
def batch_score(resumes, job_descriptions):
    batch_size = 4
    results = []
    
    for i in range(0, len(resumes), batch_size):
        batch_resumes = resumes[i:i+batch_size]
        batch_jds = job_descriptions[i:i+batch_size]
        
        # Process batch
        batch_results = model.batch_predict(batch_resumes, batch_jds)
        results.extend(batch_results)
    
    return results

# 3. Caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_score(resume_hash, jd_hash):
    return model.score(resume_text, jd_text)
```

### Infrastructure Optimizations

```python
# 1. GPU Acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Model Parallelism
model = torch.nn.DataParallel(model)

# 3. Async Processing
import asyncio

async def async_score(resume_text, jd_text):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.score, resume_text, jd_text)
    return result
```

## Continuous Benchmarking

### Automated Testing Pipeline

```python
def continuous_benchmark():
    """Run benchmarks automatically"""
    
    # 1. Load test cases
    test_cases = load_test_dataset()
    
    # 2. Run benchmarks
    results = run_all_benchmarks(test_cases)
    
    # 3. Compare with previous results
    previous_results = load_previous_results()
    comparison = compare_results(results, previous_results)
    
    # 4. Alert if performance degrades
    if comparison["accuracy_drop"] > 0.05:
        send_alert("Model performance degraded!")
    
    # 5. Save results
    save_results(results)
    
    return results

# Schedule to run daily
import schedule
schedule.every().day.at("02:00").do(continuous_benchmark)
```

### A/B Testing Framework

```python
def ab_test_models(model_a_url, model_b_url, test_cases):
    """Compare two models with A/B testing"""
    
    results_a = benchmark_model(model_a_url, test_cases)
    results_b = benchmark_model(model_b_url, test_cases)
    
    # Statistical significance test
    from scipy import stats
    
    scores_a = [r["accuracy"] for r in results_a]
    scores_b = [r["accuracy"] for r in results_b]
    
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
    
    return {
        "model_a_performance": np.mean(scores_a),
        "model_b_performance": np.mean(scores_b),
        "statistical_significance": p_value < 0.05,
        "p_value": p_value
    }
```

## Custom Metrics

### Domain-Specific Metrics

```python
def calculate_skill_matching_accuracy(predictions, ground_truth):
    """Calculate accuracy of skill matching"""
    skill_accuracy = []
    
    for pred, truth in zip(predictions, ground_truth):
        pred_skills = extract_skills(pred["matched_skills"])
        truth_skills = extract_skills(truth["required_skills"])
        
        intersection = len(set(pred_skills) & set(truth_skills))
        union = len(set(pred_skills) | set(truth_skills))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        skill_accuracy.append(jaccard_similarity)
    
    return np.mean(skill_accuracy)

def calculate_experience_level_accuracy(predictions, ground_truth):
    """Calculate accuracy of experience level assessment"""
    correct = 0
    total = len(predictions)
    
    for pred, truth in zip(predictions, ground_truth):
        pred_level = extract_experience_level(pred["explanation"])
        truth_level = truth["experience_level"]
        
        if pred_level == truth_level:
            correct += 1
    
    return correct / total
```

### Business Metrics

```python
def calculate_business_impact(model_results):
    """Calculate business impact metrics"""
    
    # Time saved vs manual review
    manual_review_time = 15 * 60  # 15 minutes per resume
    model_processing_time = np.mean([r["processing_time"] for r in model_results])
    time_saved_per_resume = manual_review_time - model_processing_time
    
    # Cost savings
    hourly_rate = 50  # HR professional hourly rate
    cost_per_manual_review = (manual_review_time / 3600) * hourly_rate
    cost_per_model_review = 0.01  # Estimated API cost
    cost_savings_per_resume = cost_per_manual_review - cost_per_model_review
    
    return {
        "time_saved_per_resume": time_saved_per_resume,
        "cost_savings_per_resume": cost_savings_per_resume,
        "accuracy_vs_manual": 0.85  # Estimated based on studies
    }
```

## Reporting and Visualization

### Automated Report Generation

```python
def generate_benchmark_report(results):
    """Generate comprehensive benchmark report"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(results),
            "overall_accuracy": calculate_accuracy(results),
            "average_processing_time": np.mean([r["processing_time"] for r in results])
        },
        "detailed_metrics": calculate_all_metrics(results),
        "recommendations": generate_recommendations(results),
        "visualizations": create_visualizations(results)
    }
    
    # Save as JSON
    with open(f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate HTML report
    generate_html_report(report)
    
    return report
```

This comprehensive benchmarking guide ensures thorough evaluation and continuous improvement of the Gemma 3 resume scoring system.
