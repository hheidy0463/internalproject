# 🚀 Cursor + Colab Pro Workflow Guide

## Overview
This setup lets you develop in Cursor with superior code editing experience, then seamlessly sync to Colab Pro for GPU training.

## 📋 Workflow Steps

### 1. Development in Cursor (Local)
```bash
# Open project in Cursor
# Choose "Jupyter Server" when prompted
# Work on your notebook with full IDE features
```

### 2. Test Logic on CPU
```bash
# Test your training logic locally first
python test_cpu_training.py
```

### 3. Sync to Version Control
```bash
# Commit your changes
git add .
git commit -m "Update training code"
git push origin main
```

### 4. Train in Colab Pro
```python
# In Colab, run the setup cell:
exec(open('colab_setup.py').read())

# Clone your repository:
!git clone https://github.com/yourusername/your-repo-name.git
%cd your-repo-name

# Install requirements:
!pip install -r requirements.txt

# Run your training!
```

### 5. Download Results
```python
# In Colab, save models to Drive:
from google.colab import files
files.download('trained_model.safetensors')

# Or sync back via Git:
!git add models/
!git commit -m "Add trained models"
!git push origin main
```

## 🔧 Setup Commands

### Initial Setup (Run Once)
```bash
# In your project directory
git init
git remote add origin https://github.com/yourusername/your-repo-name.git
```

### Daily Workflow
```bash
# 1. Develop in Cursor
# 2. Test locally (optional)
python test_cpu_training.py

# 3. Commit and push
git add .
git commit -m "Your changes"
git push origin main

# 4. Train in Colab Pro
# 5. Download/sync results
```

## 🎯 Key Benefits

- **Best of both worlds**: Cursor's IDE features + Colab's GPU power
- **Version controlled**: All changes tracked in Git
- **Cost effective**: Only pay for GPU when training
- **Flexible**: Develop offline, train online
- **Reproducible**: Requirements.txt ensures consistent environments

## 📁 File Structure
```
internalproject/
├── stable_diffusion_experimentation.ipynb  # Main notebook
├── requirements.txt                         # Dependencies
├── colab_setup.py                          # Colab optimization
├── test_cpu_training.py                    # Local testing
├── WORKFLOW_GUIDE.md                       # This guide
└── .gitignore                              # Git exclusions
```

## 🚨 Important Notes

1. **Never commit large files** (models, datasets) - use .gitignore
2. **Test locally first** - catch bugs before expensive GPU time
3. **Use version control** - always commit before major changes
4. **Monitor GPU usage** - Colab Pro has limits
5. **Save frequently** - Colab sessions can disconnect

## 🛠️ Troubleshooting

### If Colab can't find your repo:
```python
# Make sure repo is public or you're authenticated
!git config --global user.email "your-email@gmail.com"
!git config --global user.name "Your Name"
```

### If packages won't install:
```python
# Force reinstall
!pip install --force-reinstall -r requirements.txt
```

### If GPU runs out of memory:
```python
# Reduce batch size in training_args
training_args["batch_size"] = 2  # Smaller batch
training_args["gradient_accumulation_steps"] = 16  # Compensate
```
