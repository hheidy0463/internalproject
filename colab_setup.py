"""
COLAB PRO GPU OPTIMIZATION SETUP
Copy this cell to the beginning of your Colab notebook
"""

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("🚀 Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally")

if IN_COLAB:
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Clone your repository (replace with your actual repo URL)
    # !git clone https://github.com/yourusername/your-repo-name.git
    # %cd your-repo-name
    
    # Install requirements
    !pip install -r requirements.txt

import torch
import os

# GPU Configuration
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🎯 GPU: {device_name}")
    print(f"💾 Memory: {total_memory:.1f} GB")
    
    # Optimize memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Clear cache
    torch.cuda.empty_cache()
    print("✅ GPU optimizations applied")
else:
    print("❌ No GPU available!")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")
