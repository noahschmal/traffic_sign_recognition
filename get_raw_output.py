import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from segmentation_model import UNet

# Configuration
MODEL_PATH = 'models/segmentation_model.pth'
IMAGE_DIR = 'pics'
IMG_SIZE = (512, 512) 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'raw_output_results'

def process_image(model, image_path, output_path):
    # Load Image
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open {image_path}: {e}")
        return

    original_w, original_h = pil_img.size
    
    # Preprocess for Model
    input_img = TF.resize(pil_img, IMG_SIZE)
    input_tensor = TF.to_tensor(input_img).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
        # Create binary mask (0 or 1)
        mask = (probs > 0.5).float()

    # Postprocess mask
    mask_cpu = mask.squeeze().cpu().numpy()
    # Convert to 0-255 uint8
    mask_img = Image.fromarray((mask_cpu * 255).astype(np.uint8))
    # Resize back to original size
    mask_img = mask_img.resize((original_w, original_h))
    
    mask_img.save(output_path)
    print(f"Raw mask saved to {output_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return
        
    model.eval()

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp"]
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    print(f"Found {len(test_images)} images. Processing...")
    
    for i, img_path in enumerate(test_images):
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
        process_image(model, img_path, save_path)

if __name__ == '__main__':
    main()