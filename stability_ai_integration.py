#!/usr/bin/env python3
"""
Stability AI API Integration for Style Transfer Training
Generate training data using Stability AI's API
"""

import os
import requests
import json
from pathlib import Path
from PIL import Image
import io
import time
from typing import List, Dict, Optional
import base64
from image_preprocessor import ImagePreprocessor

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Or set environment variables manually")

class StabilityAIIntegration:
    """Integration with Stability AI API for dataset generation"""
    
    def __init__(self, api_key: str):
        self.engine_id = "stable-diffusion-xl-1024-v1-0"  # Default to working engine
        self.api_host = os.getenv('API_HOST', 'https://api.stability.ai')
        self.api_key = api_key
        self.preprocessor = ImagePreprocessor(target_size=(1024, 1024))

        if api_key is None:
            raise Exception("Missing Stability API key.")
        
        # Available engines for different use cases (only working ones)
        self.available_engines = {
            "sdxl": "stable-diffusion-xl-1024-v1-0",  # High quality, strict dimensions
            "sdxl-turbo": "stable-diffusion-xl-turbo",  # Fast, good quality
            "sdxl-09": "stable-diffusion-xl-1024-v0-9"  # Alternative SDXL version
        }
    
    def set_engine(self, engine_name: str):
        """Change the engine being used"""
        if engine_name in self.available_engines:
            self.engine_id = self.available_engines[engine_name]
            print(f"✅ Switched to engine: {self.engine_id}")
        else:
            print(f"❌ Unknown engine: {engine_name}")
            print(f"Available engines: {list(self.available_engines.keys())}")
    
    def get_engine_info(self):
        """Get information about the current engine"""
        engine_info = {
            "sdxl": {
                "name": "Stable Diffusion XL 1.0",
                "dimensions": "Fixed dimensions only (1024x1024, 1152x896, etc.)",
                "quality": "Excellent",
                "speed": "Slow",
                "best_for": "High quality, when you can control dimensions"
            },
            "sdxl-turbo": {
                "name": "Stable Diffusion XL Turbo",
                "dimensions": "Fixed dimensions only",
                "quality": "Very Good",
                "speed": "Fast",
                "best_for": "Quick high-quality results"
            },
            "sdxl-09": {
                "name": "Stable Diffusion XL 0.9",
                "dimensions": "Fixed dimensions only",
                "quality": "Very Good",
                "speed": "Medium",
                "best_for": "Good quality with medium speed"
            }
        }
        
        current = None
        for key, value in self.available_engines.items():
            if value == self.engine_id:
                current = key
                break
        
        if current and current in engine_info:
            info = engine_info[current]
            print(f"\n🔧 Current Engine: {info['name']}")
            print(f"📐 Dimensions: {info['dimensions']}")
            print(f"⭐ Quality: {info['quality']}")
            print(f"⚡ Speed: {info['speed']}")
            print(f"🎯 Best for: {info['best_for']}")
        
        return engine_info
    # inside your class, replace your generation method with this variant
    def generate_realism_training_dataset(
        self,
        style_prompt: str,
        input_dir: str = "input_images",
        save_dir: str = "generated_training_data",
        negative_prompt: str = (
            "blurry, low detail, soft focus, extra fingers, extra limbs, "
            "deformed hands, deformed face, distorted anatomy, artifacts, "
            "low quality, text, watermark, logo"
        ),
        strengths=(0.2, 0.25, 0.3),
        cfg_scale: float = 5.5,
        steps: int = 40,
        sampler: str = "K_EULER_ANCESTRAL",
        style_preset: str = "photographic",
        seed: int = 123456,
        timeout: int = 120,
    ):
        from pathlib import Path
        import base64, time
        from PIL import Image

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(input_dir).mkdir(parents=True, exist_ok=True)

        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        inputs = [p for p in Path(input_dir).iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not inputs:
            raise Exception(f"No image files found in {input_dir}")

        print(f"Found {len(inputs)} images to process...")

        all_out = []
        ok = 0

        for i, img_path in enumerate(inputs):
            print(f"Processing image {i+1}/{len(inputs)}: {img_path.name}")

            # sanity check: SDXL is happier with multiples of 64
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                if (w % 64) or (h % 64):
                    print(f"⚠️  {img_path.name} is {w}x{h}. Consider preprocessing to 1024x1024.")
            except Exception as e:
                print(f"❌ Failed to inspect {img_path.name}: {e}")
                continue

            for s in strengths:
                try:
                    with open(img_path, "rb") as f:
                        files = {"init_image": f}
                        data = {
                            # do NOT include width or height for image-to-image v1
                            "image_strength": s,
                            "init_image_mode": "IMAGE_STRENGTH",
                            "text_prompts[0][text]": style_prompt,
                            "text_prompts[0][weight]": 1.0,
                            "text_prompts[1][text]": negative_prompt,
                            "text_prompts[1][weight]": -1.0,
                            "cfg_scale": cfg_scale,
                            "samples": 1,
                            "steps": steps,
                            "seed": seed,
                            "style_preset": style_preset,
                            "sampler": sampler,
                        }

                        resp = requests.post(
                            f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image",
                            headers={
                                "Accept": "application/json",
                                "Authorization": f"Bearer {self.api_key}",
                            },
                            files=files,
                            data=data,
                            timeout=timeout,
                        )

                    if resp.status_code != 200:
                        print(f"❌ {img_path.name} (strength={s}) HTTP {resp.status_code}: {resp.text[:300]}")
                        continue

                    payload = resp.json()
                    arts = payload.get("artifacts", [])
                    if not arts:
                        print(f"⚠️  No artifacts for {img_path.name} at strength {s}. Payload keys: {list(payload.keys())}")
                        continue

                    for j, art in enumerate(arts):
                        fr = art.get("finishReason")
                        if fr and fr != "SUCCESS":
                            print(f"⚠️  finishReason={fr} on {img_path.name} at strength {s}")

                        b64 = art.get("base64")
                        if not b64:
                            print(f"⚠️  Missing base64 for {img_path.name} at strength {s}")
                            continue

                        out_name = f"{img_path.stem}_s{s:.2f}_styled_{j}.png"
                        out_path = Path(save_dir) / out_name
                        with open(out_path, "wb") as outf:
                            outf.write(base64.b64decode(b64))

                        all_out.append(str(out_path))
                        ok += 1

                    time.sleep(0.8)

                except requests.Timeout:
                    print(f"⏱️  Timeout on {img_path.name} at strength {s}")
                except Exception as e:
                    print(f"❌ Error on {img_path.name} at strength {s}: {e}")

        return {
            "status": "success",
            "total_input_images": len(inputs),
            "successful_generations": ok,
            "images_generated": len(all_out),
            "image_paths": all_out,
            "prompt": style_prompt,
            "engine": self.engine_id,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "sampler": sampler,
            "style_preset": style_preset,
        }
    

    def test_api_connection(self) -> bool:
        """Test if the API connection is working"""
        try:
            response = requests.get(
                f"{self.api_host}/v1/user/balance",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.status_code == 200
        except Exception:
            return False

def main():
    """Main menu for Stability AI integration"""
    print("Stability AI API Integration for Style Transfer")
    
    # Check for API key
    api_key = os.getenv("STABILITY_API_KEY")
    
    if not api_key:
        print("STABILITY_API_KEY environment variable not set")
        print("Please set your Stability AI API key:")
        print("export STABILITY_API_KEY='your-api-key-here'")
        return
    
    print("Stability AI API key found")
    
    # Initialize integration
    try:
        stability = StabilityAIIntegration(api_key)
    except Exception as e:
        print(f"Error initializing Stability AI: {e}")
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Generate realism dataset")
        print("2. Preprocess images for API")
        print("3. Switch engine/model")
        print("4. Test API connection")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            style_prompt = input("Enter style prompt: ").strip()
            if style_prompt:
                input_dir = input("Enter input images directory (default: input_images): ").strip()
                if not input_dir:
                    input_dir = "input_images"
                
                output_dir = input("Enter output directory (default: generated_training_data): ").strip()
                if not output_dir:
                    output_dir = "generated_training_data"
                
                # Check if images need preprocessing
                print("Checking image compatibility...")
                needs_preprocessing = False
                for img_file in Path(input_dir).glob("*"):
                    if img_file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}:
                        if not stability.preprocessor.validate_image(img_file):
                            needs_preprocessing = True
                            break
                
                if needs_preprocessing:
                    print("⚠️  Some images need preprocessing for optimal API compatibility")
                    preprocess_choice = input("Preprocess images first? (y/n, default: y): ").strip().lower()
                    if preprocess_choice != 'n':
                        print("Preprocessing images...")
                        try:
                            preprocessed_dir = f"{input_dir}_preprocessed"
                            processed = stability.preprocessor.preprocess_directory(input_dir, preprocessed_dir)
                            print(f"✅ Preprocessed {len(processed)} images to {preprocessed_dir}")
                            input_dir = preprocessed_dir
                        except Exception as e:
                            print(f"❌ Preprocessing failed: {e}")
                            continue
                
                try:
                    result = stability.generate_realism_training_dataset(style_prompt, input_dir, output_dir)
                    if result:
                        print(f"✅ Processed {result['total_input_images']} input images")
                        print(f"✅ Successfully generated {result['successful_generations']} styled images")
                        print(f"Images saved to: {', '.join(result['image_paths'])}")
                except Exception as e:
                    print(f"❌ Error generating dataset: {e}")
        
        elif choice == "2":
            input_dir = input("Enter input images directory: ").strip()
            if input_dir:
                try:
                    processed = stability.preprocessor.preprocess_directory(input_dir)
                    print(f"✅ Preprocessed {len(processed)} images")
                    for img_path in processed:
                        print(f"  - {img_path}")
                except Exception as e:
                    print(f"❌ Error preprocessing images: {e}")
        
        elif choice == "3":
            print("\n🔧 Available Engines:")
            stability.get_engine_info()
            print(f"\nCurrent engine: {stability.engine_id}")
            print("\nEngine options:")
            print("sdxl       - High quality, strict dimensions (needs preprocessing)")
            print("sdxl-turbo - Fast, good quality, strict dimensions")
            print("sdxl-09    - Good quality, medium speed, strict dimensions")
            
            engine_choice = input("\nEnter engine choice (sdxl/sdxl-turbo/sdxl-09): ").strip().lower()
            if engine_choice:
                stability.set_engine(engine_choice)
                print(f"✅ Switched to: {stability.engine_id}")
        
        elif choice == "4":
            print("Testing API connection...")
            if stability.test_api_connection():
                print("✅ API connection successful!")
            else:
                print("❌ API connection failed")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
