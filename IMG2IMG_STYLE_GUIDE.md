# Img-to-Img Style Transfer with LoRA Training

## What is Img-to-Img Style Transfer?

Img-to-img style transfer is a technique where you train a model to transform input images into a specific artistic style. Unlike text-to-image generation, this approach:

- **Takes an input image** (e.g., a photo)
- **Applies a specific style** (e.g., impressionist painting)
- **Outputs a transformed image** that maintains the original content but with the new style

## Quick Start for Style Transfer

### 1. Test the System
```bash
# Test with dummy data first
python img2img_style_training.py
```

### 2. Prepare Your Real Data
```python
from img2img_style_training import Img2ImgStyleTrainer

trainer = Img2ImgStyleTrainer()

# Your training data
source_images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]  # Original photos
target_images = ["style1.jpg", "style2.jpg", "style3.jpg"]  # Stylized versions
style_prompts = ["impressionist style", "impressionist style", "impressionist style"]

# Train the model
training_data = trainer.prepare_style_training_data(source_images, target_images, style_prompts)
trainer.train(training_data, num_epochs=20)
```

## Understanding the Training Process

### Training Data Structure
For img-to-img style transfer, you need **pairs** of images:

```
Training Pair 1:
├── Source: photo1.jpg (original landscape photo)
├── Target: style1.jpg (same landscape in impressionist style)
└── Prompt: "impressionist painting style"

Training Pair 2:
├── Source: photo2.jpg (original portrait photo)
├── Target: style2.jpg (same portrait in impressionist style)
└── Prompt: "impressionist painting style"
```

### Key Differences from Text-to-Image
1. **Input**: Source image + text prompt (not just text)
2. **Output**: Target image in desired style
3. **Training**: Model learns to transform source → target
4. **Inference**: Can apply style to any new input image

## Preparing Your Training Data

### 1. Collect Source Images
- **Photos** that represent your content (landscapes, portraits, objects)
- **Variety** in composition, lighting, subjects
- **High quality** (at least 512x512 pixels)
- **Consistent style** (all photos, all paintings, etc.)

### 2. Create Target Images
- **Apply the style** you want to learn to each source image
- **Use the same style** consistently across all targets
- **Maintain content** - same composition, same subject
- **Professional quality** - the better your targets, the better the results

### 3. Write Style Prompts
- **Consistent language** across all training examples
- **Descriptive** but not overly specific
- **Examples**:
  - "impressionist painting style with visible brushstrokes"
  - "watercolor art style with flowing colors"
  - "oil painting technique in renaissance style"

## Training Configuration

### LoRA Settings for Style Transfer
```python
# Balanced approach (recommended starting point)
trainer._setup_lora(
    r=16,           # Good balance of quality/speed
    lora_alpha=32,  # 2 * r
    lora_dropout=0.1
)

# High quality (more memory required)
trainer._setup_lora(
    r=32,           # Higher capacity for complex styles
    lora_alpha=64,  # 2 * r
    lora_dropout=0.1
)

# Lightweight (faster training)
trainer._setup_lora(
    r=8,            # Lower capacity, faster training
    lora_alpha=16,  # 2 * r
    lora_dropout=0.05
)
```

### Training Parameters
```python
trainer.train(
    training_data=training_data,
    num_epochs=20,           # Start with 20, increase if needed
    batch_size=1,            # Safe for most GPUs
    learning_rate=5e-5,      # Good starting point for LoRA
    save_steps=50,          # Save every 100 steps
    output_dir="./my_style_lora"
)
```

## Common Style Transfer Use Cases

### 1. Artistic Styles
```python
# Impressionist
style_prompts = ["impressionist painting style with visible brushstrokes"]

# Watercolor
style_prompts = ["watercolor art style with flowing colors and transparency"]

# Oil Painting
style_prompts = ["oil painting technique with rich textures and depth"]

# Digital Art
style_prompts = ["digital art style with clean lines and modern aesthetics"]
```

### 2. Historical Periods
```python
# Renaissance
style_prompts = ["renaissance painting style with classical composition"]

# Baroque
style_prompts = ["baroque art style with dramatic lighting and movement"]

# Art Nouveau
style_prompts = ["art nouveau style with flowing organic forms"]
```

### 3. Modern Styles
```python
# Minimalist
style_prompts = ["minimalist art style with simple forms and limited colors"]

# Abstract
style_prompts = ["abstract expressionism with bold colors and gestural brushwork"]

# Pop Art
style_prompts = ["pop art style with bright colors and bold graphic elements"]
```

## Training Tips for Style Transfer

### 1. **Data Quality is Critical**
- Use high-quality source and target images
- Ensure consistent style across all targets
- Match source and target content closely

### 2. **Start Small, Scale Up**
- Begin with 10-20 training pairs
- Test with simple styles first
- Gradually increase complexity

### 3. **Monitor Training Progress**
- Loss should decrease steadily
- Check generated samples during training
- Save checkpoints frequently

### 4. **Style Consistency**
- Use the same style prompt for all examples
- Ensure target images have consistent visual characteristics
- Avoid mixing different styles in one training run

## Troubleshooting Style Transfer

### Common Issues:

1. **Style Not Transferring**
   ```python
   # Increase training epochs
   trainer.train(num_epochs=50)
   
   # Increase LoRA rank
   trainer._setup_lora(r=32, lora_alpha=64)
   
   # Improve training data quality
   # Ensure target images are consistently styled
   ```

2. **Content Loss (image becomes unrecognizable)**
   ```python
   # Reduce learning rate
   trainer.train(learning_rate=1e-5)
   
   # Increase training data
   # Add more diverse source images
   
   # Check target image quality
   # Ensure targets maintain original content
   ```

3. **Overfitting to Training Data**
   ```python
   # Reduce LoRA rank
   trainer._setup_lora(r=8, lora_alpha=16)
   
   # Add dropout
   trainer._setup_lora(lora_dropout=0.2)
   
   # Use more training data
   ```

4. **Memory Issues**
   ```python
   # Reduce batch size
   trainer.train(batch_size=1)
   
   # Use smaller images
   training_data = trainer.prepare_style_training_data(
       source_images, target_images, style_prompts, image_size=256
   )
   
   # Use smaller base model
   trainer = Img2ImgStyleTrainer(base_model_id="runwayml/stable-diffusion-v1-5")
   ```

## Using Your Trained Style Transfer Model

### Generate Style Transfers
```python
# Load your trained model
trainer = Img2ImgStyleTrainer()
# Load LoRA weights (implement loading logic)

# Apply style to new images
result = trainer.generate_style_transfer(
    source_image_path="new_photo.jpg",
    style_prompt="impressionist painting style",
    output_path="styled_result.png",
    strength=0.75  # How much to transform (0.0 = no change, 1.0 = complete transformation)
)
```

### Batch Processing
```python
import os
from pathlib import Path

input_folder = "input_photos/"
output_folder = "styled_results/"

for image_file in Path(input_folder).glob("*.jpg"):
    output_path = Path(output_folder) / f"styled_{image_file.name}"
    
    trainer.generate_style_transfer(
        source_image_path=str(image_file),
        style_prompt="impressionist painting style",
        output_path=str(output_path)
    )
```

## Complete Workflow Example

### Step 1: Prepare Your Data
```bash
# Organize your training data
mkdir style_training_data
mkdir style_training_data/source    # Original photos
mkdir style_training_data/target    # Stylized versions
mkdir style_training_data/output    # Training results
```

### Step 2: Train the Model
```python
from img2img_style_training import Img2ImgStyleTrainer

# Initialize trainer
trainer = Img2ImgStyleTrainer()

# Prepare training data
source_images = [f"style_training_data/source/{f}" for f in os.listdir("style_training_data/source")]
target_images = [f"style_training_data/target/{f}" for f in os.listdir("style_training_data/target")]
style_prompts = ["impressionist painting style"] * len(source_images)

# Train
training_data = trainer.prepare_style_training_data(source_images, target_images, style_prompts)
trainer.train(
    training_data=training_data,
    num_epochs=30,
    output_dir="style_training_data/output"
)
```

### Step 3: Test and Use
```python
# Test on new images
test_images = ["test1.jpg", "test2.jpg", "test3.jpg"]

for test_img in test_images:
    result = trainer.generate_style_transfer(
        test_img,
        "impressionist painting style",
        f"styled_{test_img}"
    )
```

## Advanced Techniques

### 1. **Multi-Style Training**
Train one model on multiple styles:
```python
# Mix different styles in one training run
style_prompts = [
    "impressionist style",
    "watercolor style", 
    "oil painting style",
    "impressionist style",  # Repeat for balance
    "watercolor style"
]
```

### 2. **Style Strength Control**
```python
# During inference, control transformation strength
result = trainer.generate_style_transfer(
    source_image_path="photo.jpg",
    style_prompt="impressionist style",
    strength=0.5  # Subtle transformation
)

result = trainer.generate_style_transfer(
    source_image_path="photo.jpg", 
    style_prompt="impressionist style",
    strength=0.9  # Strong transformation
)
```

### 3. **Conditional Style Transfer**
```python
# Train with multiple style prompts
style_prompts = [
    "impressionist style, bright colors",
    "impressionist style, muted tones",
    "impressionist style, warm palette"
]
```

## Best Practices Summary

1. **Data Preparation**
   - High-quality source and target images
   - Consistent style across all targets
   - Good variety in source content

2. **Training Strategy**
   - Start with simple styles
   - Use balanced LoRA settings (r=16)
   - Monitor loss and save checkpoints

3. **Testing and Iteration**
   - Test on diverse input images
   - Adjust parameters based on results
   - Collect feedback and refine

4. **Production Use**
   - Batch process multiple images
   - Control style strength as needed
   - Maintain consistent quality

---

**Remember**: Img-to-img style transfer requires high-quality training data and careful attention to style consistency. Start simple, test often, and gradually build up to more complex styles!
