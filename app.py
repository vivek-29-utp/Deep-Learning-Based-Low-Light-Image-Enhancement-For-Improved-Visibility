

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import uuid
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.zerodce_pp import create_model
from inference import ImageEnhancer
import numpy as np
import cv2
import sys

# Set UTF-8 encoding for file operations
if sys.version_info[0] >= 3:
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass

app = Flask(__name__)
app.config['SECRET_KEY'] = 'adaptive_enhancement_system_implementation'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Load the trained enhancer
enhancer = None
device = torch.device('cpu')

def load_model():
    """Load the trained enhancement model with all improvements"""
    global enhancer
    try:
        enhancer = ImageEnhancer(model_path='checkpoints/epoch_12.pth', device='cpu')
        print(f"EXCELLENT model loaded successfully from checkpoints/epoch_12.pth (56.6/100 accuracy)")
    except Exception as e:
        print(f"Error loading model: {e}")
        enhancer = None

def enhance_image(image_path):
    """Enhance image using deep learning model with color preservation"""
    try:
        # Load model
        model = create_model()
        checkpoint = torch.load('checkpoints/epoch_12.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Store original color information - resize to match enhanced size
        original_resized = image.resize((256, 256), Image.LANCZOS)
        original_np = np.array(original_resized)
        original_hsv = cv2.cvtColor(original_np, cv2.COLOR_RGB2HSV)
        
        # Simple transform - NO aggressive preprocessing
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Run model
        with torch.no_grad():
            enhanced_tensor, curves = model(input_tensor)
        
        # Convert back to PIL - MINIMAL processing
        enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).numpy()
        enhanced_np = np.clip(enhanced_np, 0, 1)
        
        # VERY CONSERVATIVE brightness adjustment
        overall_brightness = np.mean(enhanced_np)
        
        # Only boost if truly dark
        if overall_brightness < 0.25:
            brightness_factor = 1.1  # Very conservative
        else:
            brightness_factor = 1.0  # No change
        
        # Apply brightness factor for enhancement
        enhanced_np = enhanced_np * brightness_factor
        
        # Convert back to uint8
        enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
        
        # Simple HSV value replacement for color preservation
        enhanced_hsv = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2HSV)
        
        # Replace the Value (brightness) channel of the original HSV with the enhanced Value
        # This preserves the original Hue and Saturation, only changing brightness
        original_hsv[:, :, 2] = enhanced_hsv[:, :, 2]
        
        # Convert back to RGB
        final_np = cv2.cvtColor(original_hsv, cv2.COLOR_HSV2RGB)
        final_np = np.clip(final_np, 0, 255).astype(np.uint8)
        
        enhanced_pil = Image.fromarray(final_np)
        
        # Resize back with high-quality interpolation
        enhanced_pil = enhanced_pil.resize(original_size, Image.LANCZOS)
        
        print(f"Original brightness: {overall_brightness:.3f}")
        print(f"Brightness factor: {brightness_factor:.2f}")
        print(f"Color preservation: HSV value replacement active")
        
        return enhanced_pil
        
    except Exception as e:
        raise Exception(f"Enhancement failed: {str(e)}")

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html', results=None, error=None)

@app.route('/enhance', methods=['POST'])
def enhance():
    """Handle image enhancement"""
    if 'image' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        return render_template('index.html', results=None, error='Please upload an image file (JPG, PNG, BMP)')
    
    try:
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        original_filename = f"{unique_id}_{file.filename}"
        enhanced_filename = f"enhanced_{unique_id}_{file.filename}"
        
        # Ensure directories exist
        os.makedirs('static/uploads', exist_ok=True)
        os.makedirs('static/results', exist_ok=True)
        
        # Save original image (ensure binary mode)
        original_path = os.path.join('static/uploads', original_filename)
        file.save(original_path)
        
        # Verify file was saved correctly
        if not os.path.exists(original_path):
            raise Exception("Failed to save uploaded file")
        
        # Enhance image
        enhanced_image = enhance_image(original_path)
        
        # Save enhanced image
        enhanced_path = os.path.join('static/results', enhanced_filename)
        enhanced_image.save(enhanced_path, quality=95)
        
        # Prepare results
        results = {
            'original_filename': original_filename,
            'enhanced_filename': enhanced_filename,
            'original_size': os.path.getsize(original_path),
            'enhanced_size': os.path.getsize(enhanced_path)
        }
        
        return render_template('index.html', results=results, error=None)
        
    except Exception as e:
        return render_template('index.html', results=None, error=str(e))

@app.route('/download/<filename>')
def download_file(filename):
    """Download enhanced image"""
    try:
        file_path = os.path.join('static/results', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Download failed: {str(e)}')
        return redirect(url_for('index'))

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join('static/uploads', filename))

@app.route('/static/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_file(os.path.join('static/results', filename))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return render_template('index.html', results=None, error='File too large. Maximum size is 10MB.')

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', results=None, error='Page not found.')

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return render_template('index.html', results=None, error='Internal server error.')

if __name__ == '__main__':
    print("Starting Adaptive Low-Light Enhancement System...")
    print("Loading model...")
    load_model()
    
    if enhancer is None:
        print("Warning: Enhanced model not loaded. Please run training first.")
        print("Run: python train.py")
    
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
