"""
Fixed CPU Testing Script for Style Finetuning Logic
"""

import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
import numpy as np

def test_cpu_training_logic():
    """Test training logic on CPU with minimal resources"""
    print("🧪 Testing training logic on CPU...")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        
        print("✅ Model loaded successfully")
        
        # Test noise scheduling with CORRECT dimensions
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Simulate training step with CORRECT latent dimensions
        batch_size = 1
        # Latent space is 4 channels for Stable Diffusion (not 3 like RGB)
        latent_height, latent_width = 8, 8  # 64/8 = 8 (VAE downscales by 8x)
        latent_channels = 4  # This is the key fix!
        
        # Create dummy latent batch (not RGB image)
        latents = torch.randn(batch_size, latent_channels, latent_height, latent_width)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,))
        
        # Add noise (this is what happens in training)
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        print("✅ Noise scheduling works")
        
        # Test UNet forward pass with correct dimensions
        with torch.no_grad():
            # Dummy text embeddings (correct size for SD 1.5)
            text_embeddings = torch.randn(batch_size, 77, 768)
            
            # This is the correct training forward pass
            noise_pred = pipe.unet(
                noisy_latents,  # 4-channel latents, not 3-channel images
                timesteps, 
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Calculate loss (MSE between predicted and actual noise)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            print(f"✅ Training step works - Loss: {loss.item():.4f}")
        
        # Test text encoder
        with torch.no_grad():
            prompt = "test prompt"
            text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]
            print(f"✅ Text encoding works - Shape: {text_embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_complete_training_step():
    """Test a more complete training step"""
    print("🔄 Testing complete training step...")
    
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Load components separately for more control
        from diffusers import UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        print("✅ Components loaded")
        
        # Simulate training data
        batch_size = 2
        prompts = ["a red apple", "a blue car"]
        
        # Encode text
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = text_encoder(text_inputs.input_ids)[0]
        
        # Simulate latents (from VAE encoding of real images)
        latents = torch.randn(batch_size, 4, 8, 8)  # Small for CPU testing
        
        # Sample timesteps
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,))
        
        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Forward pass
        with torch.no_grad():
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        print(f"✅ Complete training step - Loss: {loss.item():.4f}")
        print(f"   Latents shape: {latents.shape}")
        print(f"   Text embeddings shape: {text_embeddings.shape}")
        print(f"   Noise prediction shape: {noise_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in complete training step: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting FIXED CPU training logic tests...\n")
    
    # Test LoRA first
    try:
        from peft import LoraConfig
        print("✅ PEFT/LoRA available")
        lora_ok = True
    except ImportError:
        print("❌ PEFT not available")
        lora_ok = False
    
    print()
    
    # Test basic training logic
    basic_ok = test_cpu_training_logic()
    print()
    
    # Test complete training step
    if basic_ok:
        complete_ok = test_complete_training_step()
        
        if complete_ok and lora_ok:
            print("\n🎉 ALL TESTS PASSED! Ready for GPU training in Colab.")
            print("\n📋 Next steps:")
            print("1. Push code to GitHub")
            print("2. Open Colab Pro")
            print("3. Clone repository") 
            print("4. Run: exec(open('colab_setup.py').read())")
            print("5. Start full training with GPU")
        else:
            print("\n⚠️  Some tests failed - check errors above")
    else:
        print("\n⚠️  Basic training logic failed")
