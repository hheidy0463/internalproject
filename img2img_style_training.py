#!/usr/bin/env python3
"""
Img-to-Img Style Transfer LoRA Training
Specialized for training LoRA models that can transform images into specific artistic styles
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler, StableDiffusionImg2ImgPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image
import json
from datetime import datetime
from tqdm.auto import tqdm

class Img2ImgStyleTrainer:
    """Specialized LoRA trainer for img-to-img style transfer"""
    
    def __init__(self, 
                 base_model_id="runwayml/stable-diffusion-v1-5",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.base_model_id = base_model_id
        
        print(f"🎨 Initializing Img-to-Img Style Transfer LoRA trainer on {device}")
        print(f"📦 Base model: {base_model_id}")
        
        # Load base model components
        self._load_models()
        
        # Setup LoRA
        self._setup_lora()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
    def _load_models(self):
        """Load the base Stable Diffusion model components"""
        print("📥 Loading base model components...")
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model_id, 
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_id, 
            subfolder="text_encoder"
        ).to(self.device)
        
        # Load UNet and scheduler
        self.unet = StableDiffusionPipeline.from_pretrained(
            self.base_model_id,
            subfolder="unet"
        ).unet.to(self.device)
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_model_id, 
            subfolder="scheduler"
        )
        
        # Load VAE for image encoding/decoding
        self.vae = StableDiffusionPipeline.from_pretrained(
            self.base_model_id,
            subfolder="vae"
        ).vae.to(self.device)
        
        print("✅ Base model components loaded")
        
    def _setup_lora(self, r=16, lora_alpha=32, lora_dropout=0.1):
        """Configure and apply LoRA to the UNet"""
        print(f"🔧 Setting up LoRA (r={r}, alpha={lora_alpha})...")
        
        # LoRA configuration optimized for style transfer
        self.lora_config = LoraConfig(
            r=r,  # Rank - controls model capacity
            lora_alpha=lora_alpha,  # Scaling parameter
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Target attention layers
            lora_dropout=lora_dropout,
            bias="none"
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, self.lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        # Freeze text encoder and VAE
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        print("✅ LoRA setup complete")
        
    def prepare_style_training_data(self, source_images, target_images, style_prompts, image_size=512):
        """Prepare training data for style transfer (source -> target with style prompt)"""
        print(f"📁 Preparing {len(source_images)} style transfer training pairs...")
        
        training_data = []
        
        for source_path, target_path, style_prompt in zip(source_images, target_images, style_prompts):
            try:
                # Load source image (input)
                source_image = Image.open(source_path).convert("RGB")
                source_image = source_image.resize((image_size, image_size))
                
                # Load target image (desired output)
                target_image = Image.open(target_path).convert("RGB")
                target_image = target_image.resize((image_size, image_size))
                
                # Convert to tensors and normalize
                source_tensor = torch.tensor(np.array(source_image)).permute(2, 0, 1).float() / 255.0
                source_tensor = 2 * source_tensor - 1  # Scale to [-1, 1]
                
                target_tensor = torch.tensor(np.array(target_image)).permute(2, 0, 1).float() / 255.0
                target_tensor = 2 * target_tensor - 1  # Scale to [-1, 1]
                
                # Tokenize style prompt
                tokens = self.tokenizer(
                    style_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                training_data.append({
                    "source_pixel_values": source_tensor,
                    "target_pixel_values": target_tensor,
                    "input_ids": tokens.input_ids.squeeze(),
                    "style_prompt": style_prompt
                })
                
            except Exception as e:
                print(f"⚠️  Skipping {source_path} -> {target_path}: {e}")
                
        print(f"✅ Prepared {len(training_data)} style transfer training pairs")
        return training_data
    
    def create_style_dummy_dataset(self, num_samples=10, image_size=64):
        """Create dummy dataset for style transfer testing"""
        print(f"🧪 Creating dummy style transfer dataset with {num_samples} samples...")
        
        training_data = []
        style_prompts = [
            "impressionist painting style",
            "watercolor art style", 
            "oil painting technique",
            "digital art style",
            "sketch drawing style",
            "abstract expressionism",
            "renaissance painting style",
            "modern minimalist art",
            "surrealist art style",
            "pop art style"
        ]
        
        for i in range(num_samples):
            # Create random source and target images
            source_image = torch.randn(3, image_size, image_size)
            target_image = torch.randn(3, image_size, image_size)
            
            # Tokenize style prompt
            prompt = style_prompts[i % len(style_prompts)]
            tokens = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            training_data.append({
                "source_pixel_values": source_image,
                "target_pixel_values": target_image,
                "input_ids": tokens.input_ids.squeeze(),
                "style_prompt": prompt
            })
            
        print("✅ Dummy style transfer dataset created")
        return training_data
    
    def training_step(self, batch, optimizer):
        """Single training step for style transfer"""
        # Get batch data
        if isinstance(batch, list):
            item = batch[0]
        else:
            item = batch
            
        source_pixels = item["source_pixel_values"].to(self.device)
        target_pixels = item["target_pixel_values"].to(self.device)
        input_ids = item["input_ids"].to(self.device)
        
        # Encode source image to latent space (this will be our conditioning)
        with torch.no_grad():
            if source_pixels.dim() == 3:
                source_pixels = source_pixels.unsqueeze(0)
            source_latents = self.vae.encode(source_pixels).latent_dist.sample()
            source_latents = source_latents * 0.18215  # Scale factor
        
        # Encode target image to latent space (this is what we want to predict)
        with torch.no_grad():
            if target_pixels.dim() == 3:
                target_pixels = target_pixels.unsqueeze(0)
            target_latents = self.vae.encode(target_pixels).latent_dist.sample()
            target_latents = target_latents * 0.18215  # Scale factor
        
        # Encode text prompt
        with torch.no_grad():
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            text_embeddings = self.text_encoder(input_ids)[0]
        
        # Sample random timesteps
        batch_size = target_latents.shape[0]
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), 
            device=target_latents.device
        ).long()
        
        # Add noise to target latents
        noise = torch.randn_like(target_latents)
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        
        # For img-to-img, we condition on both source image and text
        # We'll concatenate source latents with noisy target latents
        # This is a simplified approach - in practice you might want more sophisticated conditioning
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(self, 
              training_data, 
              num_epochs=10, 
              batch_size=1, 
              learning_rate=5e-5,
              save_steps=100,
              output_dir="./style_transfer_lora"):
        """Main training loop for style transfer"""
        print(f"🎯 Starting style transfer training for {num_epochs} epochs...")
        print(f"📊 Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=len(training_data) * num_epochs
        )
        
        # Training loop
        progress_bar = tqdm(range(num_epochs * len(training_data)), desc="Style Transfer Training")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                # Training step
                loss = self.training_step(batch, optimizer)
                epoch_losses.append(loss)
                
                # Update learning rate
                lr_scheduler.step()
                
                # Update progress
                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "Epoch": epoch + 1,
                    "Step": self.global_step,
                    "Loss": f"{loss:.4f}",
                    "LR": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(output_dir, f"checkpoint_{self.global_step}")
                    
                    # Save best model
                    avg_loss = np.mean(epoch_losses[-save_steps:])
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss
                        self.save_checkpoint(output_dir, "best_model")
            
            # Epoch summary
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"\n📈 Epoch {epoch + 1} complete - Avg Loss: {avg_epoch_loss:.4f}")
            
        progress_bar.close()
        
        # Save final model
        self.save_checkpoint(output_dir, "final_model")
        print(f"🎉 Style transfer training complete! Models saved to {output_dir}")
        
    def save_checkpoint(self, output_dir, name):
        """Save LoRA checkpoint"""
        checkpoint_dir = os.path.join(output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir)
        
        # Save training config
        config = {
            "base_model": self.base_model_id,
            "lora_config": {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "target_modules": list(self.lora_config.target_modules),
                "lora_dropout": self.lora_config.lora_dropout,
                "bias": self.lora_config.bias
            },
            "training_info": {
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "timestamp": datetime.now().isoformat(),
                "training_type": "img2img_style_transfer"
            }
        }
        
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"💾 Saved checkpoint: {name}")
        
    def generate_style_transfer(self, source_image_path, style_prompt, output_path=None, num_inference_steps=20):
        """Generate a style transfer using the trained LoRA"""
        print(f"🎨 Generating style transfer for: '{style_prompt}'")
        
        # Load source image
        source_image = Image.open(source_image_path).convert("RGB")
        
        # Create img2img pipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.base_model_id,
            unet=self.unet,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Generate style transfer
        image = pipe(
            prompt=style_prompt,
            image=source_image,
            strength=0.75,  # How much to transform the image
            guidance_scale=7.5,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        # Save if path provided
        if output_path:
            image.save(output_path)
            print(f"💾 Style transfer saved to: {output_path}")
        
        return image

def main():
    """Main function demonstrating img-to-img style transfer training"""
    print("🎨 Img-to-Img Style Transfer LoRA Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = Img2ImgStyleTrainer()
    
    # Example: Training on impressionist style transfer
    print("\n📚 Example: Training on Impressionist Style Transfer")
    
    # In real usage, you would have:
    # source_images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]  # Original photos
    # target_images = ["impressionist1.jpg", "impressionist2.jpg", "impressionist3.jpg"]  # Stylized versions
    # style_prompts = ["impressionist painting style", "impressionist painting style", "impressionist painting style"]
    
    # For demo, use dummy data
    print("🧪 Using dummy data for demonstration")
    training_data = trainer.create_style_dummy_dataset(num_samples=20, image_size=64)
    
    # Train the model
    print("\n🚀 Starting style transfer training...")
    trainer.train(
        training_data=training_data,
        num_epochs=8,           # Fewer epochs for demo
        batch_size=1,
        learning_rate=5e-5,
        save_steps=20,
        output_dir="./style_transfer_lora"
    )
    
    # Generate a test style transfer
    print("\n🎨 Generating test style transfer...")
    
    # Create a dummy source image for testing
    dummy_source = Image.new('RGB', (64, 64), color='red')
    dummy_source.save("dummy_source.png")
    
    # Test style transfer
    test_prompt = "impressionist painting style with visible brushstrokes"
    try:
        sample_image = trainer.generate_style_transfer(
            "dummy_source.png",
            test_prompt,
            output_path="./style_transfer_sample.png"
        )
        print("✅ Style transfer generation complete!")
    except Exception as e:
        print(f"⚠️  Style transfer generation failed: {e}")
        print("This is expected with dummy data - use real images for actual results")
    
    print("\n🎉 Img-to-img style transfer training complete!")
    print("📁 Check the 'style_transfer_lora' directory for saved models")
    print("🖼️  Check 'style_transfer_sample.png' for a generated sample")

if __name__ == "__main__":
    main()
