"""
CPU Testing Script for Style Finetuning Logic
Test your training loop with small models before moving to Colab GPU
"""

import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image

def test_cpu_training_logic():
    """Test training logic on CPU with minimal resources"""
    print("🧪 Testing training logic on CPU...")
    
    # Use a smaller model for testing
    model_id = "runwayml/stable-diffusion-v1-5"  # Smaller than SDXL
    
    try:
        # Load with CPU and low precision for testing
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for CPU
            use_safetensors=True,
        )
        
        print("✅ Model loaded successfully")
        
        # Test basic generation (very small image)
        prompt = "a simple red apple, artistic style"
        
        # Generate tiny image for testing
        image = pipe(
            prompt=prompt,
            height=64,  # Very small for CPU testing
            width=64,
            num_inference_steps=5,  # Few steps for speed
            guidance_scale=3.0
        ).images[0]
        
        print("✅ Basic generation works")
        
        # Test noise scheduling (core of training)
        from diffusers import DDPMScheduler
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Simulate training step
        batch_size = 1
        height, width = 64, 64
        
        # Create dummy batch
        images = torch.randn(batch_size, 3, height, width)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,))
        
        # Add noise (this is what happens in training)
        noise = torch.randn_like(images)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
        
        print("✅ Noise scheduling works")
        
        # Test UNet forward pass (simplified)
        with torch.no_grad():
            # Dummy text embeddings
            text_embeddings = torch.randn(batch_size, 77, 768)
            
            # This would be your training forward pass
            noise_pred = pipe.unet(noisy_images, timesteps, encoder_hidden_states=text_embeddings).sample
            
            # Calculate loss (MSE between predicted and actual noise)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            print(f"✅ Training step works - Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_lora_setup():
    """Test LoRA configuration"""
    print("🔧 Testing LoRA setup...")
    
    try:
        from peft import LoraConfig
        
        lora_config = LoraConfig(
            r=4,  # Small rank for testing
            lora_alpha=8,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
        )
        
        print("✅ LoRA configuration works")
        return True
        
    except ImportError:
        print("❌ PEFT not installed - install with: pip install peft")
        return False
    except Exception as e:
        print(f"❌ LoRA error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting CPU training logic tests...\n")
    
    # Test LoRA first (lightweight)
    lora_ok = test_lora_setup()
    print()
    
    # Test training logic
    if lora_ok:
        training_ok = test_cpu_training_logic()
        
        if training_ok:
            print("\n🎉 All tests passed! Ready for GPU training in Colab.")
            print("\n📋 Next steps:")
            print("1. Push code to GitHub")
            print("2. Open Colab Pro")
            print("3. Clone repository")
            print("4. Run full training with GPU")
        else:
            print("\n⚠️  Training logic needs fixes before GPU training")
    else:
        print("\n⚠️  Install missing dependencies first")
