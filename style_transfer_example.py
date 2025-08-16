#!/usr/bin/env python3
"""
Practical Example: Img-to-Img Style Transfer with Real Images
This script shows how to set up training data and train a style transfer model
"""

import os
from pathlib import Path
from img2img_style_training import Img2ImgStyleTrainer

def setup_training_folders():
    """Create the folder structure for style transfer training"""
    print("Setting up training folder structure...")
    
    folders = [
        "style_training_data",
        "style_training_data/source",      # Original photos
        "style_training_data/target",      # Stylized versions  
        "style_training_data/output"       # Training results
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")
    
    print("\nNext steps:")
    print("1. Put your original photos in 'style_training_data/source/'")
    print("2. Put the stylized versions in 'style_training_data/target/'")
    print("3. Run the training script")

def train_realism_style():
    """Example: Train on realism style transfer"""
    print("\nTraining Realism Style Transfer Model")
    print("=" * 50)
    
    # Initialize trainer
    trainer = Img2ImgStyleTrainer()
    
    # Check if we have real training data
    source_folder = Path("style_training_data/source")
    target_folder = Path("style_training_data/target")
    
    if not source_folder.exists() or not target_folder.exists():
        print("Training folders not found. Run setup_training_folders() first.")
        return
    
    # Get available images
    source_images = list(source_folder.glob("*.jpg")) + list(source_folder.glob("*.png"))
    target_images = list(target_folder.glob("*.jpg")) + list(target_folder.glob("*.png"))
    
    if not source_images or not target_images:
        print("No images found in training folders.")
        print("Please add images to:")
        print(f"  Source: {source_folder}")
        print(f"  Target: {target_folder}")
        return
    
    print(f"Found {len(source_images)} source images and {len(target_images)} target images")
    
    # Ensure we have matching pairs
    if len(source_images) != len(target_images):
        print("Warning: Number of source and target images don't match")
        print("Training will use the minimum number of pairs")
    
    # Use the minimum number of pairs
    num_pairs = min(len(source_images), len(target_images))
    source_images = source_images[:num_pairs]
    target_images = target_images[:num_pairs]
    
    # Style prompt for realism
    style_prompt = "realistic photograph, high quality"
    style_prompts = [style_prompt] * num_pairs
    
    print(f"🎯 Training on {num_pairs} image pairs with style: '{style_prompt}'")
    
    # Prepare training data
    training_data = trainer.prepare_style_training_data(
        [str(img) for img in source_images],
        [str(img) for img in target_images],
        style_prompts,
        image_size=512  # Adjust based on your image quality
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train(
        training_data=training_data,
        num_epochs=20,           # Start with 20 epochs
        batch_size=1,            # Safe batch size
        learning_rate=5e-5,      # Good starting learning rate
        save_steps=50,           # Save every 50 steps
        output_dir="./style_training_data/output"
    )
    
    print("Training complete!")

def test_style_transfer():
    """Test the trained model on new images"""
    print("\n Testing Style Transfer Model")
    print("=" * 40)
    
    # Initialize trainer
    trainer = Img2ImgStyleTrainer()
    
    # Check if we have a trained model
    model_path = Path("style_training_data/output/final_model")
    if not model_path.exists():
        print("No trained model found. Train a model first.")
        return
    
    print(f"Found trained model at: {model_path}")
    
    # Test on a sample image
    test_image = "style_training_data/source/test_image.jpg"  # Adjust path as needed
    
    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        print("Please add a test image to test the model")
        return
    
    print(f"🎨 Testing style transfer on: {test_image}")
    
    # Generate style transfer
    try:
        result = trainer.generate_style_transfer(
            source_image_path=test_image,
            style_prompt="impressionist painting style with visible brushstrokes",
            output_path="style_transfer_result.png",
            num_inference_steps=20
        )
        print("Style transfer test complete!")
        print("Check 'style_transfer_result.png' for the result")
        
    except Exception as e:
        print(f"Style transfer failed: {e}")
        print("This might be due to model loading issues or memory constraints")

def batch_style_transfer():
    """Apply style transfer to multiple images"""
    print("\nBatch Style Transfer")
    print("=" * 30)
    
    # Initialize trainer
    trainer = Img2ImgStyleTrainer()
    
    # Check if we have a trained model
    model_path = Path("style_training_data/output/final_model")
    if not model_path.exists():
        print("No trained model found. Train a model first.")
        return
    
    # Input and output folders
    input_folder = Path("input_photos")
    output_folder = Path("styled_results")
    
    # Create folders if they don't exist
    input_folder.mkdir(exist_ok=True)
    output_folder.mkdir(exist_ok=True)
    
    # Get input images
    input_images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))
    
    if not input_images:
        print(f"No images found in {input_folder}")
        print("Please add photos to process")
        return
    
    print(f"Processing {len(input_images)} images...")
    
    # Style prompt
    style_prompt = "impressionist painting style with visible brushstrokes"
    
    # Process each image
    for i, image_file in enumerate(input_images, 1):
        print(f"Processing {i}/{len(input_images)}: {image_file.name}")
        
        output_path = output_folder / f"styled_{image_file.name}"
        
        try:
            trainer.generate_style_transfer(
                source_image_path=str(image_file),
                style_prompt=style_prompt,
                output_path=str(output_path),
                num_inference_steps=20
            )
            print(f"Saved: {output_path.name}")
            
        except Exception as e:
            print(f"Failed to process {image_file.name}: {e}")
    
    print(f"\nBatch processing complete! Check {output_folder} for results")

def main():
    """Main menu for style transfer operations"""
    print("Img-to-Img Style Transfer - Practical Examples")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Setup training folders")
        print("2. Train realism style model")
        print("3. Test style transfer")
        print("4. Batch style transfer")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            setup_training_folders()
        elif choice == "2":
            train_realism_style()
        elif choice == "3":
            test_style_transfer()
        elif choice == "4":
            batch_style_transfer()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
