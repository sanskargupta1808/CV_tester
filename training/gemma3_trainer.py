#!/usr/bin/env python3
"""
GEMMA 3 MODEL TRAINING FOR RESUME-JD MATCHING
============================================

Trains a Gemma 3 model on resume-job description dataset for precise analysis.
Implements LoRA fine-tuning for efficient training and deployment.

Requirements:
- transformers>=4.40.0
- torch>=2.0.0
- peft>=0.10.0
- datasets>=2.18.0
- accelerate>=0.28.0
"""

import os
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# Transformers and PEFT imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, DatasetDict
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for Gemma 3 training"""
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

class ResumeJDDataset:
    """Dataset handler for resume-job description pairs"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.tokenizer = None
        
    def create_sample_dataset(self) -> List[Dict]:
        """Create a comprehensive sample dataset for training"""
        return [
            {
                "resume": """SARAH JOHNSON
Senior Software Engineer

EXPERIENCE:
Senior Software Engineer at TechCorp (2019-2024)
- Led development of microservices architecture using Python, Django, and PostgreSQL
- Built React-based frontend applications serving 100K+ users
- Implemented CI/CD pipelines using Docker and AWS services
- Managed team of 5 developers and mentored junior engineers
- Designed and optimized database schemas for high-performance applications

Software Engineer at StartupXYZ (2017-2019)
- Developed full-stack web applications using Python, Flask, and JavaScript
- Worked with MySQL databases and Redis caching
- Implemented RESTful APIs and integrated third-party services
- Collaborated in Agile development environment

SKILLS:
- Programming: Python, JavaScript, TypeScript, SQL
- Frameworks: Django, Flask, React, Node.js
- Databases: PostgreSQL, MySQL, Redis, MongoDB
- Cloud: AWS (EC2, S3, RDS, Lambda), Docker, Kubernetes
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
                
                "score": 92,
                "explanation": "Excellent match with 7+ years experience, strong Python/Django skills, React expertise, PostgreSQL knowledge, AWS experience, leadership background, and microservices architecture experience. Education and all technical requirements fully met."
            },
            
            {
                "resume": """MIKE CHEN
Full-Stack Developer

EXPERIENCE:
Full-Stack Developer at WebSolutions Inc (2021-2024)
- Developed web applications using Python, Flask, and Vue.js
- Worked with MySQL databases and implemented REST APIs
- Used Git for version control and participated in code reviews
- Collaborated with design team to implement responsive UI/UX

Junior Developer at LocalTech (2020-2021)
- Built simple web applications using HTML, CSS, JavaScript
- Learned Python programming and basic database concepts
- Assisted senior developers with bug fixes and feature implementation

SKILLS:
- Programming: Python, JavaScript, HTML, CSS
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
                
                "score": 58,
                "explanation": "Moderate match with 4 years experience (below 5+ requirement), Python skills present but Flask instead of Django, Vue.js instead of React, MySQL instead of PostgreSQL, basic AWS knowledge, no leadership or microservices experience mentioned."
            },
            
            {
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
- Programming: Python, R, SQL, MATLAB
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
                
                "score": 25,
                "explanation": "Poor match - candidate is a data scientist with no web development experience. Has Python and PostgreSQL knowledge but lacks Django, React, JavaScript, AWS, web development, and full-stack experience. Different career focus entirely."
            },
            
            {
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
- Languages: Python, JavaScript, TypeScript, Go, SQL
- Backend: Django, Django REST Framework, Flask, Node.js
- Frontend: React, Redux, HTML5, CSS3, SASS
- Databases: PostgreSQL, MySQL, Redis, Elasticsearch
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
                
                "score": 98,
                "explanation": "Perfect match with 8+ years experience, expert-level Python/Django skills, extensive React experience, PostgreSQL optimization expertise, comprehensive AWS knowledge, proven leadership and mentoring experience, microservices architecture expertise, and advanced education. Exceeds all requirements."
            }
        ]
    
    def format_for_training(self, data: List[Dict]) -> List[Dict]:
        """Format data for instruction-following training"""
        formatted_data = []
        
        for item in data:
            # Create instruction-following format
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
                "input": "",  # Empty for instruction-following format
                "text": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            })
        
        return formatted_data
    
    def prepare_dataset(self, tokenizer, max_length: int = 2048) -> DatasetDict:
        """Prepare dataset for training"""
        self.tokenizer = tokenizer
        
        # Get sample data
        raw_data = self.create_sample_dataset()
        formatted_data = self.format_for_training(raw_data)
        
        # Split data (80% train, 10% validation, 10% test)
        train_size = int(0.8 * len(formatted_data))
        val_size = int(0.1 * len(formatted_data))
        
        train_data = formatted_data[:train_size] if train_size > 0 else formatted_data[:1]
        val_data = formatted_data[train_size:train_size + val_size] if val_size > 0 else formatted_data[:1]
        test_data = formatted_data[train_size + val_size:] if len(formatted_data) > train_size + val_size else formatted_data[:1]
        
        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data).map(tokenize_function, batched=True)
        val_dataset = Dataset.from_list(val_data).map(tokenize_function, batched=True)
        test_dataset = Dataset.from_list(test_data).map(tokenize_function, batched=True)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })

class Gemma3Trainer:
    """Main trainer class for Gemma 3 resume scoring model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset_handler = ResumeJDDataset()
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Configure quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Configure LoRA if enabled
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def prepare_training_data(self):
        """Prepare training dataset"""
        logger.info("Preparing training dataset...")
        return self.dataset_handler.prepare_dataset(self.tokenizer, self.config.max_length)
    
    def train(self):
        """Execute training process"""
        logger.info("Starting Gemma 3 training process...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Prepare dataset
        dataset = self.prepare_training_data()
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Beginning training...")
        trainer.train()
        
        # Save final model
        logger.info("Saving trained model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metadata
        metadata = {
            "model_name": self.config.model_name,
            "training_date": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "dataset_size": {
                "train": len(dataset["train"]),
                "validation": len(dataset["validation"]),
                "test": len(dataset["test"])
            }
        }
        
        with open(f"{self.config.output_dir}/training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training completed! Model saved to: {self.config.output_dir}")
        return trainer

def main():
    """Main training function"""
    print("🚀 GEMMA 3 RESUME SCORER TRAINING")
    print("=" * 50)
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = Gemma3Trainer(config)
    
    try:
        # Execute training
        trained_model = trainer.train()
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📁 Model saved to: {config.output_dir}")
        print(f"🎯 Ready for deployment and benchmarking")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ TRAINING FAILED: {e}")
        return False

if __name__ == "__main__":
    main()
