# Img-to-Img Style Transfer with LoRA Training

This project provides a complete workflow for training LoRA (Low-Rank Adaptation) models on Stable Diffusion for **img-to-img style transfer** - transforming input images into specific artistic styles while preserving content.

## Quick Start

### 1. Test the System
```bash
# Test with dummy data first
python img2img_style_training.py
```

### 2. Set Up Your Training Data
```bash
# Create training folders and set up data
python style_transfer_example.py
# Choose option 1 to create training folders
```

### 3. Use Universal Dataset Handler (Recommended)
```bash
# Works with ANY dataset structure automatically
python universal_dataset_handler.py
# Choose option 2 to clone and analyze repositories
# Choose option 3 to adapt any dataset for training
```

### 4. Find and Prepare Datasets (Alternative)
```bash
# Find datasets and set up training data
python dataset_finder.py
# Choose option 3 to build custom dataset
```

### 4. Train Your Style Model
```bash
# Train on your own images
python style_transfer_example.py
# Choose option 2 to train on your data
```

## 📁 Project Structure

```
internalproject/
├── img2img_style_training.py      # Core img-to-img LoRA trainer
├── style_transfer_example.py      # Practical setup and training script
├── universal_dataset_handler.py    # Universal dataset handler (NEW!)
├── dataset_finder.py              # Dataset discovery and setup tool
├── IMG2IMG_STYLE_GUIDE.md         # Comprehensive style transfer guide
├── DATASET_GUIDE.md               # Complete dataset guide
├── stable_diffusion_experimentation.ipynb  # Main notebook
├── requirements.txt                # Dependencies
├── .gitignore                     # Git exclusions
└── README.md                      # This guide
```

## What You Can Do

### Universal Dataset Support 🌐
- **Any Dataset Structure**: Automatically detects and adapts any dataset format
- **Repository Integration**: Clone and analyze GitHub repositories directly
- **Automatic Adaptation**: Converts any dataset structure to our training format
- **Smart Detection**: Identifies dataset type and suggests optimal training approach

### Img-to-Img Style Transfer 🎨
- **Transform Photos**: Convert photos into artistic styles
- **Style Learning**: Train models to apply specific visual styles
- **Batch Processing**: Apply styles to multiple images automatically
- **Content Preservation**: Maintain original image content while changing style

### Training Options
- **Lightweight**: `r=8` for fast training with minimal resources
- **Balanced**: `r=16` for good quality/speed balance (recommended)
- **High Quality**: `r=32` for maximum quality (more memory required)

## 🔧 Key Features

- **Automatic Checkpointing**: Saves progress during training
- **Progress Monitoring**: Real-time loss and learning rate tracking
- **Flexible Data**: Use your own images or start with dummy data
- **Memory Efficient**: Only trains LoRA parameters, not full model
- **Style Consistency**: Learns consistent artistic style across all examples

## Documentation

- **[IMG2IMG_STYLE_GUIDE.md](IMG2IMG_STYLE_GUIDE.md)**: Complete guide with examples and troubleshooting
- **[DATASET_GUIDE.md](DATASET_GUIDE.md)**: Complete guide to finding and preparing training data
- **[universal_dataset_handler.py](universal_dataset_handler.py)**: Universal dataset handler for any format
- **[style_transfer_example.py](style_transfer_example.py)**: Interactive setup and training script
- **[dataset_finder.py](dataset_finder.py)**: Tool to find and set up datasets

## Example Use Cases

### Style Training
```python
from img2img_style_training import Img2ImgStyleTrainer

# Initialize trainer
trainer = Img2ImgStyleTrainer()

# Prepare your training data
source_images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]  # Original photos
target_images = ["style1.jpg", "style2.jpg", "style3.jpg"]  # Stylized versions
style_prompts = ["impressionist style", "impressionist style", "impressionist style"]

# Train the model
training_data = trainer.prepare_style_training_data(source_images, target_images, style_prompts)
trainer.train(training_data, num_epochs=20)
```

### Common Styles
- **Artistic**: Impressionist, watercolor, oil painting, digital art
- **Historical**: Renaissance, Baroque, Art Nouveau
- **Modern**: Minimalist, abstract, pop art

## 📊 Training Tips

### 1. **Start Small**
- Begin with `r=16` and `num_epochs=20`
- Test with 10-20 training pairs first
- Gradually increase complexity

### 2. **Data Quality is Critical**
- Use high-quality source and target images
- Ensure consistent style across all targets
- Match source and target content closely

### 3. **Monitor Progress**
- Loss should decrease steadily
- Save checkpoints frequently
- Test generated samples during training

## Complete Workflow

### Step 1: Prepare Data
```bash
# Create training folders
python style_transfer_example.py
# Choose option 1

# Add your images:
# - style_training_data/source/ (original photos)
# - style_training_data/target/ (stylized versions)
```

### Step 2: Train Model
```bash
# Train your style model
python style_transfer_example.py
# Choose option 2
```

### Step 3: Use Model
```bash
# Test on new images
python style_transfer_example.py
# Choose option 3

# Batch process multiple images
python style_transfer_example.py
# Choose option 4
```

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size or image size
2. **Style Not Transferring**: Increase epochs or LoRA rank
3. **Poor Results**: Improve training data quality
4. **Slow Training**: Use smaller rank or base model

### Getting Help:
- Check [IMG2IMG_STYLE_GUIDE.md](IMG2IMG_STYLE_GUIDE.md) for detailed explanations
- Start with dummy data to test the system
- Ensure consistent style across all training examples

## Dependencies

All required packages are in `requirements.txt`:
- PyTorch, Diffusers, Transformers
- PEFT for LoRA implementation
- PIL, NumPy for data handling

## Next Steps

1. **Test the system** with dummy data first
2. **Set up training folders** for your data
3. **Collect training pairs** (source + target images)
4. **Train your style model** with real data
5. **Apply styles** to new images

---

**Happy Style Transfer Training!**

For detailed explanations, see [IMG2IMG_STYLE_GUIDE.md](IMG2IMG_STYLE_GUIDE.md)
