# 🧠 Gemma 3 Model Training Guide

## Overview

This guide covers the complete training process for the Gemma 3 resume scoring model, including dataset preparation, model fine-tuning, and evaluation.

## Training Architecture

### Base Model
- **Model**: google/gemma-2-2b-it
- **Type**: Instruction-tuned language model
- **Parameters**: 2 billion
- **Context Length**: 8192 tokens

### Fine-tuning Method
- **Technique**: LoRA (Low-Rank Adaptation)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.1

## Dataset Preparation

### Dataset Structure

The training dataset consists of resume-job description pairs with ground truth scores:

```python
{
    "resume": "Full resume text...",
    "job_description": "Complete job description...",
    "score": 85,  # Ground truth score (0-100)
    "category": "excellent",  # excellent/good/fair/poor
    "reasoning": "Detailed explanation of the score..."
}
```

### Sample Dataset

The system includes 6 comprehensive training examples:

1. **Perfect Match (Score: 98)** - Senior engineer with all required skills
2. **Excellent Match (Score: 92)** - Strong alignment with minor gaps
3. **Good Match (Score: 72)** - Solid match with some missing elements
4. **Fair Match (Score: 58)** - Moderate alignment, several gaps
5. **Poor Match (Score: 35)** - Frontend-only for full-stack role
6. **Very Poor Match (Score: 25)** - Data scientist for web development

### Data Formatting

The data is formatted for instruction-following training:

```python
def format_for_training(self, data: List[Dict]) -> List[Dict]:
    formatted_data = []
    
    for item in data:
        instruction = f"""Analyze the compatibility between this resume and job description. Provide a score from 0-100 and explain your reasoning.

Resume:
{item['resume']}

Job Description:
{item['job_description']}

Please provide:
1. A compatibility score (0-100)
2. A brief explanation of the match quality
3. Key strengths and gaps"""

        response = f"""Score: {item['score']}/100

Explanation: {item['explanation']}

This analysis considers technical skills alignment, experience level match, educational background, and specific job requirements."""

        formatted_data.append({
            "instruction": instruction,
            "response": response,
            "text": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        })
    
    return formatted_data
```

## Training Process

### 1. Environment Setup

```bash
# Install training dependencies
pip install torch>=2.0.0 transformers>=4.40.0 peft>=0.10.0 datasets>=2.18.0 accelerate>=0.28.0 bitsandbytes>=0.42.0

# Set up training directory
mkdir -p models/gemma3_resume_scorer
```

### 2. Model Configuration

```python
# Training configuration
@dataclass
class TrainingConfig:
    model_name: str = "google/gemma-2-2b-it"
    output_dir: str = "../models/gemma3_resume_scorer"
    max_length: int = 2048
    batch_size: int = 2
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
```

### 3. Model Loading and Setup

```python
# Load base model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)
```

### 4. Training Execution

```bash
# Run training script
python training/gemma3_trainer.py
```

The training process includes:
- Data preprocessing and tokenization
- Model setup with LoRA configuration
- Training loop with evaluation
- Model saving and metadata generation

### 5. Training Monitoring

```python
# Training arguments with monitoring
training_args = TrainingArguments(
    output_dir="./models/gemma3_resume_scorer",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    eval_steps=250,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
)
```

## Model Evaluation

### Evaluation Metrics

1. **Loss Metrics**
   - Training Loss
   - Validation Loss
   - Convergence Analysis

2. **Task-Specific Metrics**
   - Score Accuracy (MAE, RMSE)
   - Category Classification Accuracy
   - Correlation with Ground Truth

3. **Performance Metrics**
   - Inference Speed
   - Memory Usage
   - Throughput (requests/second)

### Evaluation Process

```python
# Evaluate trained model
def evaluate_model(model, test_dataset):
    predictions = []
    ground_truth = []
    
    for example in test_dataset:
        # Generate prediction
        prediction = model.generate(example["input"])
        score = extract_score(prediction)
        
        predictions.append(score)
        ground_truth.append(example["score"])
    
    # Calculate metrics
    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
    correlation = np.corrcoef(predictions, ground_truth)[0, 1]
    
    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation
    }
```

## Advanced Training Techniques

### 1. Data Augmentation

```python
def augment_dataset(original_data):
    augmented = []
    
    for item in original_data:
        # Original example
        augmented.append(item)
        
        # Paraphrase job description
        paraphrased_jd = paraphrase_text(item["job_description"])
        augmented.append({
            **item,
            "job_description": paraphrased_jd
        })
        
        # Modify resume slightly
        modified_resume = add_noise_to_resume(item["resume"])
        augmented.append({
            **item,
            "resume": modified_resume,
            "score": item["score"] - 2  # Slightly lower score
        })
    
    return augmented
```

### 2. Curriculum Learning

```python
def curriculum_training(trainer, dataset):
    # Start with easier examples (clear matches/mismatches)
    easy_examples = [ex for ex in dataset if ex["score"] < 30 or ex["score"] > 80]
    
    # Train on easy examples first
    trainer.train_dataset = easy_examples
    trainer.train()
    
    # Then train on all examples
    trainer.train_dataset = dataset
    trainer.train()
```

### 3. Multi-Task Learning

```python
# Train on multiple related tasks
tasks = [
    "resume_scoring",
    "skill_extraction", 
    "experience_level_prediction",
    "education_matching"
]

# Combine datasets and train jointly
combined_dataset = combine_task_datasets(tasks)
```

## Hyperparameter Tuning

### Grid Search

```python
hyperparameters = {
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "lora_r": [8, 16, 32],
    "lora_alpha": [16, 32, 64],
    "batch_size": [1, 2, 4]
}

best_config = grid_search(hyperparameters, train_dataset, val_dataset)
```

### Learning Rate Scheduling

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)
```

## Model Deployment

### 1. Model Export

```python
# Save trained model
trainer.save_model("./models/gemma3_resume_scorer")
tokenizer.save_pretrained("./models/gemma3_resume_scorer")

# Save training metadata
metadata = {
    "model_name": "google/gemma-2-2b-it",
    "training_date": datetime.now().isoformat(),
    "config": config.__dict__,
    "performance_metrics": evaluation_results
}

with open("./models/gemma3_resume_scorer/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

### 2. Model Loading for Inference

```python
# Load trained model for inference
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
model = PeftModel.from_pretrained(base_model, "./models/gemma3_resume_scorer")

tokenizer = AutoTokenizer.from_pretrained("./models/gemma3_resume_scorer")
```

## Continuous Learning

### 1. Online Learning

```python
def update_model_with_new_data(model, new_examples):
    # Prepare new data
    new_dataset = prepare_dataset(new_examples)
    
    # Fine-tune on new data with lower learning rate
    trainer = Trainer(
        model=model,
        train_dataset=new_dataset,
        args=TrainingArguments(
            learning_rate=1e-5,  # Lower learning rate
            num_train_epochs=1,
            per_device_train_batch_size=1
        )
    )
    
    trainer.train()
    return model
```

### 2. Active Learning

```python
def select_examples_for_annotation(model, unlabeled_data):
    uncertainties = []
    
    for example in unlabeled_data:
        # Get model prediction uncertainty
        logits = model.predict(example)
        uncertainty = calculate_uncertainty(logits)
        uncertainties.append((example, uncertainty))
    
    # Select most uncertain examples
    uncertainties.sort(key=lambda x: x[1], reverse=True)
    return [ex for ex, _ in uncertainties[:10]]
```

## Troubleshooting

### Common Training Issues

1. **Out of Memory**
   ```python
   # Reduce batch size
   batch_size = 1
   gradient_accumulation_steps = 8
   
   # Use gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

2. **Slow Convergence**
   ```python
   # Increase learning rate
   learning_rate = 5e-4
   
   # Add warmup steps
   warmup_steps = 200
   ```

3. **Overfitting**
   ```python
   # Add regularization
   lora_dropout = 0.2
   
   # Early stopping
   early_stopping_patience = 3
   ```

### Debugging Tips

```python
# Monitor training progress
def log_training_metrics(trainer):
    logs = trainer.state.log_history
    
    for log in logs:
        if "train_loss" in log:
            print(f"Step {log['step']}: Loss = {log['train_loss']:.4f}")
        if "eval_loss" in log:
            print(f"Step {log['step']}: Val Loss = {log['eval_loss']:.4f}")
```

This comprehensive training guide ensures successful fine-tuning of the Gemma 3 model for resume scoring tasks.
