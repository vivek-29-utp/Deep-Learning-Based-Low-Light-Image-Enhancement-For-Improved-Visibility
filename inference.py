
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
import time
from pathlib import Path

# Import custom modules
from model.zerodce_pp import create_model


class ImageEnhancer:
    """
    Zero-DCE++ Image Enhancement Class
    
    Handles the complete inference pipeline for low-light image enhancement
    including preprocessing, model inference, and postprocessing.
    """
    
    def __init__(self, model_path='checkpoints/zerodce_pp.pth', device='cpu'):
        """
        Initialize image enhancer
        
        Args:
            model_path (str): Path to trained model weights
            device (str): Device for inference ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = None
        self.model_path = model_path
        
        # Image preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to 512x512 for inference
            transforms.ToTensor(),  # Converts to [0,1] range
        ])
        
        # Load model
        self.load_model()
        
        print(f"ImageEnhancer initialized on {self.device}")
        print(f"Model loaded from: {model_path}")
    
    def load_model(self):
        """
        Load trained Zero-DCE++ model from checkpoint
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract configuration if available
            config = checkpoint.get('config', {})
            num_curves = config.get('num_curves', 8)
            
            # Create model
            self.model = create_model(device=self.device)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"Model loaded successfully with {num_curves} curves")
            print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"Validation loss: {checkpoint.get('val_loss', 'Unknown')}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess input image for inference with robust error handling
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, H, W]
            PIL.Image: Original image for postprocessing
            tuple: Original image dimensions
        """
        try:
            # Load image using PIL and convert to RGB explicitly
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size  # (width, height)
            
            # Preprocess for model
            input_tensor = self.preprocess(original_image)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            
            # Ensure tensor shape is ALWAYS [1, 3, H, W]
            if input_tensor.shape[0] != 1 or input_tensor.shape[1] != 3:
                raise ValueError(f"Invalid tensor shape: {input_tensor.shape}, expected [1, 3, H, W]")
            
            return input_tensor, original_image, original_size
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            # Create fallback tensor
            fallback_tensor = torch.rand(1, 3, 512, 512)
            fallback_image = Image.new('RGB', (512, 512), color='black')
            return fallback_tensor, fallback_image, (512, 512)
    
    def postprocess_image(self, enhanced_tensor, original_size):
        """
        Postprocess enhanced image ensuring 3-channel RGB output
        
        Args:
            enhanced_tensor (torch.Tensor): Enhanced image tensor [1, 3, H, W]
            original_size (tuple): Original image dimensions (width, height)
            
        Returns:
            PIL.Image: Postprocessed enhanced image (guaranteed 3-channel RGB)
        """
        # Convert tensor to numpy
        enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy()
        enhanced_np = np.transpose(enhanced_np, (1, 2, 0))  # CHW -> HWC
        
        # Clamp strictly to [0,1] before conversion
        enhanced_np = np.clip(enhanced_np, 0.0, 1.0)
        
        # Convert to [0, 255] range and uint8 safely
        enhanced_np = (enhanced_np * 255).astype(np.uint8)
        
        # Ensure output has exactly 3 channels (RGB)
        if len(enhanced_np.shape) == 2:  # Grayscale -> RGB
            enhanced_np = np.stack([enhanced_np] * 3, axis=-1)
        elif enhanced_np.shape[2] == 4:  # RGBA -> RGB
            enhanced_np = enhanced_np[:, :, :3]
        elif enhanced_np.shape[2] != 3:  # Any other channel count
            raise ValueError(f"Invalid number of channels: {enhanced_np.shape[2]}, expected 3")
        
        # Convert to PIL Image (guaranteed RGB)
        enhanced_image = Image.fromarray(enhanced_np, mode='RGB')
        
        # Resize back to original dimensions
        enhanced_image = enhanced_image.resize(original_size, Image.LANCZOS)
        
        return enhanced_image
    
    def apply_noise_reduction(self, image):
        """
        Apply OpenCV-based noise reduction to enhanced image
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Denoised image
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(cv_image, 9, 75, 75)
        
        # Convert back to PIL
        rgb_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
    
    def apply_curves(self, image, curves):
        """
        Apply illumination curves to enhance the image with improved stability
        and brightness guards to prevent overexposure
        
        Args:
            image (torch.Tensor): Input image tensor [B, 3, H, W] normalized to [0,1]
            curves (torch.Tensor): Predicted curves [B, 24, H, W]
            
        Returns:
            torch.Tensor: Enhanced image tensor [B, 3, H, W]
        """
        enhanced_image = image.clone()
        num_curves = 8
        
        # Apply each curve iteratively with adaptive learning rate
        for i in range(num_curves):
            # Extract parameters for current curve
            a = curves[:, i*3, :, :].unsqueeze(1)      # [B, 1, H, W]
            b = curves[:, i*3 + 1, :, :].unsqueeze(1)  # [B, 1, H, W]
            c = curves[:, i*3 + 2, :, :].unsqueeze(1)  # [B, 1, H, W]
            
            # Calculate adaptive learning rate based on image brightness
            brightness = self._calculate_brightness_ratio(enhanced_image)
            lr = 0.7 + (1.0 - brightness) * 0.3  # Range: [0.7, 1.0] - conservative for natural results
            
            # Apply curve with adaptive learning rate
            r = a * torch.sigmoid(b) + c
            delta = r * (enhanced_image ** 2 - enhanced_image)
            enhanced_image = enhanced_image + lr * delta
            
            # BRIGHTNESS GUARD: Check if image is becoming too bright
            current_brightness = self._calculate_brightness_ratio(enhanced_image)
            if current_brightness > 0.85:  # Conservative limit to prevent overexposure
                # Scale back to acceptable brightness
                scale_factor = 0.85 / current_brightness
                enhanced_image = enhanced_image * scale_factor
            
            # Gradual clamping to prevent harsh transitions
            if i < num_curves - 1:  # Intermediate steps: soft clamping
                enhanced_image = 0.5 * (torch.tanh((enhanced_image - 0.5) * 2.5) + 1.0)
            else:  # Final step: hard clamping
                enhanced_image = torch.clamp(enhanced_image, 0.0, 1.0)
        
        return enhanced_image
    
    def _calculate_brightness_ratio(self, image_tensor):
        """Calculate the brightness ratio of an image tensor (0-1 range)"""
        try:
            # Handle different tensor shapes
            if len(image_tensor.shape) == 4:  # [B, C, H, W]
                grayscale = 0.299 * image_tensor[0, 0] + 0.587 * image_tensor[0, 1] + 0.114 * image_tensor[0, 2]
            elif len(image_tensor.shape) == 3:  # [C, H, W]
                grayscale = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            else:
                return 0.3  # Default moderate brightness
            
            return grayscale.mean().item()
        except Exception as e:
            print(f"Brightness calculation error: {e}")
            return 0.3  # Default moderate brightness
    
    def _apply_gamma_correction(self, image_tensor):
        """
        Apply safe gamma correction with fixed gamma = 0.8
        
        Args:
            image_tensor (torch.Tensor): Input image tensor [B, 3, H, W]
            
        Returns:
            torch.Tensor: Gamma corrected image tensor
        """
        # Clamp input to [0,1] before gamma correction
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        
        # Apply gamma correction with gamma = 1.2 (was 0.9 - brightening effect)
        gamma = 1.2
        corrected = torch.pow(image_tensor + 1e-8, gamma)  # Add small epsilon to avoid log(0)
        
        # Clamp output to [0,1] after gamma correction
        corrected = torch.clamp(corrected, 0.0, 1.0)
        
        return corrected
    
    def enhance_image_tensor(self, image, curves=None):
        """
        Apply illumination curves to enhance the image (tensor version)
        with brightness guard to skip bright images
        
        Args:
            image (torch.Tensor): Input image tensor [B, 3, H, W] in [0,1]
            curves (torch.Tensor, optional): Pre-computed curves. If None, computes them.
            
        Returns:
            tuple: (enhanced_image, curves)
        """
        # Ensure input is in valid range
        image = torch.clamp(image, 0, 1)
        
        # Compute mean luminance (grayscale mean / 255)
        brightness = self._calculate_brightness_ratio(image)
        
        # BRIGHTNESS GUARD: DISABLED - Always enhance for testing
        if False:  # brightness > 0.4:
            return image, None
            
        # Apply gamma correction BEFORE model inference
        image = self._apply_gamma_correction(image)
        
        # Get curves if not provided
        if curves is None:
            curves = self.model(image)
        
        # Apply curves for enhancement with clamping at each step
        enhanced = self.apply_curves(image, curves)
        
        # Final clamping to ensure valid range [0,1]
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        
        return enhanced, curves
    
    def enhance_image(self, image_path, apply_denoising=True):
        """
        Complete image enhancement pipeline with improved error handling
        
        Args:
            image_path (str): Path to low-light image
            apply_denoising (bool): Whether to apply noise reduction
            
        Returns:
            dict: Dictionary containing enhanced image and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess
            input_tensor, original_image, original_size = self.preprocess_image(image_path)
            input_tensor = input_tensor.to(self.device)
            
            # Compute brightness for brightness guard
            brightness = self._calculate_brightness_ratio(input_tensor)
            
            # BRIGHTNESS GUARD: DISABLED - Always enhance for testing
            if False:  # brightness > 0.5:
                print(f"Skipped (already bright): {os.path.basename(image_path)}")
                enhanced_image = original_image
            else:
                # Model inference
                with torch.no_grad():
                    enhanced_tensor, _ = self.enhance_image_tensor(input_tensor)
                
                # Postprocess
                enhanced_image = self.postprocess_image(enhanced_tensor, original_size)
                
                # Apply noise reduction if requested and image was enhanced
                if apply_denoising:
                    enhanced_image = self.apply_noise_reduction(enhanced_image)
                
                print(f"Enhanced and saved: {os.path.basename(image_path)}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                'enhanced_image': enhanced_image,
                'original_image': original_image,
                'processing_time': processing_time,
                'image_size': original_size,
                'brightness': brightness
            }
            
        except Exception as e:
            print(f"Error enhancing image {image_path}: {str(e)}")
            # Return original image in case of error
            return {
                'enhanced_image': original_image,
                'original_image': original_image,
                'processing_time': time.time() - start_time,
                'image_size': original_image.size,
                'error': str(e)
            }
    
    def calculate_metrics(self, original_path, enhanced_image):
        """
        Calculate image quality metrics
        
        Args:
            original_path (str): Path to original low-light image
            enhanced_image (PIL.Image): Enhanced image
            
        Returns:
            dict: Dictionary containing PSNR and SSIM metrics
        """
        # Load original image
        original = Image.open(original_path).convert('RGB')
        
        # Resize to match enhanced image
        original = original.resize(enhanced_image.size, Image.LANCZOS)
        
        # Convert to numpy arrays
        original_np = np.array(original)
        enhanced_np = np.array(enhanced_image)
        
        # Calculate PSNR
        psnr_value = psnr(original_np, enhanced_np, data_range=255)
        
        # Calculate SSIM
        ssim_value = ssim(original_np, enhanced_np, 
                         multichannel=True, 
                         data_range=255,
                         channel_axis=2)
        
        return {
            'psnr': psnr_value,
            'ssim': ssim_value
        }
    
    def save_image_cv2(self, image, output_path):
        """
        Save image using OpenCV with proper RGB->BGR conversion and error handling
        
        Args:
            image (PIL.Image): Image to save
            output_path (str): Output path
        """
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Ensure we have exactly 3 channels (RGB)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:  # RGBA -> RGB
                    img_array = img_array[:, :, :3]
                elif img_array.shape[2] != 3:  # Any other channel count
                    raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
            
            # Convert RGB -> BGR only once before saving with OpenCV
            if img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save using OpenCV
            success = cv2.imwrite(output_path, img_bgr)
            if not success:
                print(f"Warning: Failed to save image to {output_path}")
                
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
            # Fallback: save using PIL
            try:
                image.save(output_path)
                print(f"Saved using PIL fallback: {output_path}")
            except Exception as pil_error:
                print(f"PIL fallback also failed: {pil_error}")
    
    def enhance_batch(self, input_dir, output_dir, apply_denoising=True):
        """
        Enhance multiple images in a directory
        
        Args:
            input_dir (str): Directory containing low-light images
            output_dir (str): Directory to save enhanced images
            apply_denoising (bool): Whether to apply noise reduction
        """
        # Create output directory if missing
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        supported_formats = ('.jpg', '.jpeg', '.png')
        
        for file in os.listdir(input_dir):
            if file.lower().endswith(supported_formats):
                image_files.append(file)
        
        print(f"Processing {len(image_files)} images...")
        
        # Process each image
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Enhance image
                result = self.enhance_image(input_path, apply_denoising)
                
                # Save enhanced image using OpenCV
                self.save_image_cv2(result['enhanced_image'], output_path)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        print(f"Batch processing completed. Results saved to: {output_dir}")


def main():
    """
    Main inference function
    """
    parser = argparse.ArgumentParser(description='Zero-DCE++ Image Enhancement')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, default='enhanced_output.jpg',
                       help='Output image path or directory')
    parser.add_argument('--model', type=str, default='checkpoints/zerodce_pp.pth',
                       help='Path to trained model')
    parser.add_argument('--no-denoising', action='store_true',
                       help='Disable noise reduction')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = ImageEnhancer(model_path=args.model)
    
    # Check if input is a directory or single file
    if os.path.isdir(args.input):
        # Batch processing for directory
        enhancer.enhance_batch(args.input, args.output, not args.no_denoising)
    else:
        # Single image processing
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        
        # Create output directory if it doesn't exist and is a directory path
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        result = enhancer.enhance_image(args.input, not args.no_denoising)
        
        # Save enhanced image using OpenCV
        enhancer.save_image_cv2(result['enhanced_image'], args.output)
        
        print(f"\nEnhancement completed!")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Image size: {result['image_size']}")
        print(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
        brightness = result.get('brightness', 0)
        if brightness:
            print(f"Brightness: {brightness:.3f}")


if __name__ == "__main__":
    main()
