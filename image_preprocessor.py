#!/usr/bin/env python3
"""
Image Preprocessing Utilities for Stability AI API
Handles image resizing, format conversion, and optimization
"""

import os
from pathlib import Path
from PIL import Image, ImageOps
import io
from typing import Tuple, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for Stability AI API compatibility"""
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        self.target_size = target_size
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
    def preprocess_single_image(self, image_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Preprocess a single image for Stability AI API
        
        Args:
            image_path: Path to input image
            output_path: Path for output image (optional, auto-generated if None)
            
        Returns:
            Path to preprocessed image
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    logger.info(f"Converted {image_path.name} to RGB")
                
                # Check current dimensions
                original_size = img.size
                logger.info(f"Original size: {original_size[0]}x{original_size[1]}")
                
                # Resize and pad to target size
                processed_img = self._resize_with_padding(img)
                
                # Generate output path if not provided
                if output_path is None:
                    output_dir = image_path.parent / "preprocessed"
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"{image_path.stem}_preprocessed.png"
                
                # Save processed image
                processed_img.save(output_path, format='PNG', quality=95)
                logger.info(f"Saved preprocessed image: {output_path}")
                
                return output_path
                
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            raise
    
    def preprocess_directory(self, input_dir: str, output_dir: Optional[str] = None) -> List[Path]:
        """
        Preprocess all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output images (optional)
            
        Returns:
            List of paths to preprocessed images
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_path = input_path / "preprocessed"
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        if not image_files:
            logger.warning(f"No supported image files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images to preprocess")
        
        processed_images = []
        for i, image_file in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                output_file = output_path / f"{image_file.stem}_preprocessed.png"
                processed_path = self.preprocess_single_image(image_file, output_file)
                processed_images.append(processed_path)
            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
                continue
        
        logger.info(f"Successfully preprocessed {len(processed_images)} images")
        return processed_images
    
    def _resize_with_padding(self, img: Image.Image) -> Image.Image:
        """
        Resize image to target size while maintaining aspect ratio and adding padding
        
        Args:
            img: PIL Image object
            
        Returns:
            Resized and padded image
        """
        # Calculate scaling factor to fit within target size
        img_ratio = img.size[0] / img.size[1]
        target_ratio = self.target_size[0] / self.target_size[1]
        
        if img_ratio > target_ratio:
            # Image is wider than target, scale by width
            new_width = self.target_size[0]
            new_height = int(new_width / img_ratio)
        else:
            # Image is taller than target, scale by height
            new_height = self.target_size[1]
            new_width = int(new_height * img_ratio)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', self.target_size, (255, 255, 255))  # White padding
        
        # Calculate position to center the image
        offset_x = (self.target_size[0] - new_width) // 2
        offset_y = (self.target_size[1] - new_height) // 2
        
        # Paste resized image onto padded canvas
        new_img.paste(resized_img, (offset_x, offset_y))
        
        return new_img
    
    def validate_image(self, image_path: Path) -> bool:
        """
        Validate if an image meets Stability AI requirements
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image meets requirements, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Check format
                if img.format not in ['PNG', 'JPEG', 'BMP', 'TIFF']:
                    logger.warning(f"{image_path.name}: Unsupported format {img.format}")
                    return False
                
                # Check size
                if img.size[0] < 512 or img.size[1] < 512:
                    logger.warning(f"{image_path.name}: Too small ({img.size[0]}x{img.size[1]})")
                    return False
                
                if img.size[0] > 2048 or img.size[1] > 2048:
                    logger.warning(f"{image_path.name}: Too large ({img.size[0]}x{img.size[1]})")
                    return False
                
                # Check mode
                if img.mode not in ['RGB', 'L']:
                    logger.warning(f"{image_path.name}: Unsupported color mode {img.mode}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error validating {image_path}: {e}")
            return False

def main():
    """Command line interface for image preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess images for Stability AI API")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output directory (optional)")
    parser.add_argument("-s", "--size", default="1024x1024", help="Target size (default: 1024x1024)")
    
    args = parser.parse_args()
    
    # Parse target size
    try:
        width, height = map(int, args.size.split('x'))
        target_size = (width, height)
    except ValueError:
        print("Invalid size format. Use WIDTHxHEIGHT (e.g., 1024x1024)")
        return
    
    preprocessor = ImagePreprocessor(target_size)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        try:
            output_path = preprocessor.preprocess_single_image(input_path)
            print(f"✅ Preprocessed: {output_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif input_path.is_dir():
        # Directory
        try:
            processed = preprocessor.preprocess_directory(str(input_path), args.output)
            print(f"✅ Preprocessed {len(processed)} images")
            for img_path in processed:
                print(f"  - {img_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        print(f"❌ Input not found: {input_path}")

if __name__ == "__main__":
    main()
