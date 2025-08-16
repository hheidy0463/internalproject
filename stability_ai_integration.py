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

class StabilityAIIntegration:
    """Integration with Stability AI API for dataset generation"""
    
    def __init__(self, api_key: str):
        self.engine_id = "stable-diffusion-xl-1024-v1-0"
        self.api_host = os.getenv('API_HOST', 'https://api.stability.ai')
        self.api_key = api_key

        if api_key is None:
            raise Exception("Missing Stability API key.")

    def generate_realism_training_dataset(self, style_prompt: str, 
                                       input_dir: str = "input_images",
                                       save_dir: str = "generated_training_data") -> Optional[Dict]:
        """Generate training dataset using Stability AI's image-to-image API for multiple images"""
        
        # Ensure directories exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all image files from input directory
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        input_images = []
        
        for file_path in Path(input_dir).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                input_images.append(file_path)
        
        if not input_images:
            raise Exception(f"No image files found in {input_dir}")
        
        print(f"Found {len(input_images)} images to process...")
        
        all_generated_images = []
        successful_generations = 0
        
        for i, image_path in enumerate(input_images):
            print(f"Processing image {i+1}/{len(input_images)}: {image_path.name}")
            
            try:
                response = requests.post(
                    f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image",
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    files={
                        "init_image": open(image_path, "rb")
                    },
                    data={
                        "image_strength": 0.35,
                        "init_image_mode": "IMAGE_STRENGTH",
                        "text_prompts[0][text]": style_prompt,
                        "cfg_scale": 7,
                        "samples": 1,
                        "steps": 30,
                    }
                )

                if response.status_code != 200:
                    print(f"❌ Failed to process {image_path.name}: {response.text}")
                    continue

                data = response.json()
                
                for j, image in enumerate(data["artifacts"]):
                    output_filename = f"{image_path.stem}_styled_{j}.png"
                    image_path_output = f"{save_dir}/{output_filename}"
                    
                    with open(image_path_output, "wb") as f:
                        f.write(base64.b64decode(image["base64"]))
                    
                    all_generated_images.append(image_path_output)
                    successful_generations += 1
                
                # Add small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                continue
        
        return {
            "status": "success",
            "total_input_images": len(input_images),
            "successful_generations": successful_generations,
            "images_generated": len(all_generated_images),
            "image_paths": all_generated_images,
            "prompt": style_prompt
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
        print("2. Test API connection")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            style_prompt = input("Enter style prompt: ").strip()
            if style_prompt:
                input_dir = input("Enter input images directory (default: input_images): ").strip()
                if not input_dir:
                    input_dir = "input_images"
                
                try:
                    result = stability.generate_realism_training_dataset(style_prompt, input_dir)
                    if result:
                        print(f"✅ Processed {result['total_input_images']} input images")
                        print(f"✅ Successfully generated {result['successful_generations']} styled images")
                        print(f"Images saved to: {', '.join(result['image_paths'])}")
                except Exception as e:
                    print(f"❌ Error generating dataset: {e}")
        
        elif choice == "2":
            print("Testing API connection...")
            if stability.test_api_connection():
                print("✅ API connection successful!")
            else:
                print("❌ API connection failed")
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()
