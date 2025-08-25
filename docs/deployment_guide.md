# 🚀 Gemma 3 Resume Scorer - Deployment Guide

## Overview

This guide covers the complete deployment process for the Gemma 3 Resume Scorer system, from initial setup to production deployment.

## Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (recommended for model loading)
- 5GB+ free disk space
- Internet connection (for model downloads)

## Quick Deployment

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
python setup.py

# Start the system
./scripts/start.sh  # Linux/Mac
# or
scripts\start.bat   # Windows
```

### Option 2: Manual Deployment

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements_gemma3.txt

# 3. Deploy the system
python deployment/deploy_gemma3_complete.py
```

## Deployment Process

The deployment script performs the following steps:

1. **Dependency Check**: Verifies all required packages are installed
2. **Model Setup**: Creates mock training environment and model structure
3. **API Deployment**: Starts the FastAPI server on port 8006
4. **System Verification**: Tests all endpoints and functionality
5. **Benchmarking**: Compares performance with baseline models
6. **Report Generation**: Creates comprehensive deployment report

## Configuration Options

### Environment Variables

```bash
# Optional: Set custom configuration
export GEMMA3_PORT=8006
export GEMMA3_HOST=0.0.0.0
export GEMMA3_MODEL_PATH=./models/gemma3_resume_scorer
export GEMMA3_LOG_LEVEL=INFO
```

### Model Configuration

Edit `backend/gemma3_api.py` to customize:

```python
# Model settings
MODEL_PATH = "../models/gemma3_resume_scorer"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = [".pdf", ".doc", ".docx", ".txt"]

# API settings
API_PORT = 8006
API_HOST = "0.0.0.0"
CORS_ORIGINS = ["*"]  # Restrict in production
```

## Production Deployment

### Docker Deployment (Recommended for Production)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_gemma3.txt .
RUN pip install -r requirements_gemma3.txt

COPY . .
EXPOSE 8006

CMD ["python", "backend/gemma3_api.py"]
```

```bash
# Build and run
docker build -t gemma3-resume-scorer .
docker run -p 8006:8006 gemma3-resume-scorer
```

### Cloud Deployment

#### AWS EC2

```bash
# Launch EC2 instance (t3.large or larger recommended)
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Clone/upload your code
# Follow manual deployment steps
```

#### Google Cloud Platform

```bash
# Use Cloud Run for serverless deployment
gcloud run deploy gemma3-resume-scorer \
  --source . \
  --port 8006 \
  --memory 4Gi \
  --cpu 2
```

### Reverse Proxy Setup (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8006;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring and Logging

### Health Monitoring

```bash
# Check system health
curl http://localhost:8006/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "gemma3-trained",
  "version": "1.0.0",
  "timestamp": "2025-08-25T12:00:00"
}
```

### Log Management

```bash
# View API logs
tail -f logs/gemma3_api.log

# View system logs
tail -f logs/system.log
```

### Performance Monitoring

```python
# Add to your monitoring system
import requests
import time

def monitor_performance():
    start_time = time.time()
    response = requests.post("http://localhost:8006/score", json={
        "resume_text": "Test resume",
        "jd_text": "Test job description with sufficient length for validation"
    })
    end_time = time.time()
    
    return {
        "response_time": end_time - start_time,
        "status_code": response.status_code,
        "success": response.status_code == 200
    }
```

## Scaling Considerations

### Horizontal Scaling

```bash
# Run multiple instances
python backend/gemma3_api.py --port 8006 &
python backend/gemma3_api.py --port 8007 &
python backend/gemma3_api.py --port 8008 &

# Use load balancer (nginx, HAProxy, etc.)
```

### Vertical Scaling

- **Memory**: 8GB+ recommended for model loading
- **CPU**: Multi-core beneficial for concurrent requests
- **Storage**: SSD recommended for faster model loading

## Security Considerations

### Production Security

```python
# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add authentication
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/score")
async def score_resume(request: ScoreRequest, token: str = Depends(security)):
    # Validate token
    pass
```

### File Upload Security

```python
# Validate file types and sizes
ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(file: UploadFile):
    # Check file extension
    if not any(file.filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Check file size
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   lsof -i :8006
   kill -9 <PID>
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   free -h
   
   # Reduce model precision if needed
   torch_dtype=torch.float16  # Instead of float32
   ```

3. **Model Loading Errors**
   ```bash
   # Check model files
   ls -la models/gemma3_resume_scorer/
   
   # Verify permissions
   chmod -R 755 models/
   ```

### Debug Mode

```bash
# Run with debug logging
export GEMMA3_LOG_LEVEL=DEBUG
python backend/gemma3_api.py
```

## Backup and Recovery

### Model Backup

```bash
# Backup trained models
tar -czf gemma3_models_backup.tar.gz models/

# Restore models
tar -xzf gemma3_models_backup.tar.gz
```

### Configuration Backup

```bash
# Backup configuration
cp backend/gemma3_api.py backend/gemma3_api.py.backup
cp requirements_gemma3.txt requirements_gemma3.txt.backup
```

## Performance Optimization

### Model Optimization

```python
# Use quantization for faster inference
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

### Caching

```python
# Add response caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_score(resume_hash: str, jd_hash: str):
    # Cache scoring results
    pass
```

## Maintenance

### Regular Tasks

1. **Log Rotation**: Set up logrotate for log files
2. **Model Updates**: Periodically retrain with new data
3. **Dependency Updates**: Keep packages up to date
4. **Performance Monitoring**: Track response times and accuracy
5. **Backup Verification**: Test backup and restore procedures

### Update Process

```bash
# 1. Backup current system
tar -czf gemma3_backup_$(date +%Y%m%d).tar.gz .

# 2. Update code
git pull origin main

# 3. Update dependencies
pip install -r requirements_gemma3.txt --upgrade

# 4. Test system
python tests/test_api.py

# 5. Restart services
./scripts/start.sh
```

This deployment guide ensures a robust, scalable, and maintainable Gemma 3 Resume Scorer system.
