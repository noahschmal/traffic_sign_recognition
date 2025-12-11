import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from segmentation_model import UNet
from classification_model import create_model, class_names

# Configuration
SEG_MODEL_PATH = 'models/segmentation_model.pth'
CLS_MODEL_PATH = 'models/classification_model.pth'
IMAGE_DIR = 'pics'
OUTPUT_DIR = 'results'
IMG_SIZE_SEG = (512, 512) 
IMG_SIZE_CLS = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_classification_model():
    # Create model with structure from classification_model
    model = create_model(num_classes=len(class_names))
    # Load weights
    try:
        model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded classification model from {CLS_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def get_classification_transform():
    return T.Compose([
        T.Resize(IMG_SIZE_CLS),
        T.ToTensor(),
        # Standard ImageNet normalization is usually a safe bet if not specified otherwise
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def boxes_are_near(box1, box2, threshold=30):
    """Check if two boxes are within 'threshold' pixels of each other or overlapping."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Expand box1 by threshold
    b1_x1 = x1 - threshold
    b1_y1 = y1 - threshold
    b1_x2 = x1 + w1 + threshold
    b1_y2 = y1 + h1 + threshold
    
    # Box 2 coords
    b2_x1 = x2
    b2_y1 = y2
    b2_x2 = x2 + w2
    b2_y2 = y2 + h2
    
    # Check intersection (if NOT disjoint, then they are near/overlapping)
    # Disjoint conditions:
    if (b1_x1 > b2_x2) or (b1_x2 < b2_x1) or (b1_y1 > b2_y2) or (b1_y2 < b2_y1):
        return False
    return True

def merge_boxes(boxes, threshold=30):
    """Iteratively merge boxes that are near each other."""
    while True:
        merged_any = False
        new_boxes = []
        used_indices = set()
        
        for i in range(len(boxes)):
            if i in used_indices:
                continue
            
            current_box = boxes[i]
            
            # Try to merge with all subsequent unused boxes
            for j in range(i + 1, len(boxes)):
                if j in used_indices:
                    continue
                
                if boxes_are_near(current_box, boxes[j], threshold):
                    # Merge logic
                    b1 = current_box
                    b2 = boxes[j]
                    
                    x_min = min(b1[0], b2[0])
                    y_min = min(b1[1], b2[1])
                    x_max = max(b1[0] + b1[2], b2[0] + b2[2])
                    y_max = max(b1[1] + b1[3], b2[1] + b2[3])
                    
                    current_box = (x_min, y_min, x_max - x_min, y_max - y_min)
                    used_indices.add(j)
                    merged_any = True
            
            new_boxes.append(current_box)
        
        boxes = new_boxes
        if not merged_any:
            break
            
    return boxes

def process_image(seg_model, cls_model, image_path, output_path):
    # Load Image
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open {image_path}: {e}")
        return

    original_w, original_h = pil_img.size
    
    # --- Segmentation Step ---
    # Preprocess for Segmentation Model
    input_img_seg = TF.resize(pil_img, IMG_SIZE_SEG)
    input_tensor_seg = TF.to_tensor(input_img_seg).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output_seg = seg_model(input_tensor_seg)
        probs = torch.sigmoid(output_seg)
        mask = (probs > 0.5).float().cpu().numpy().squeeze()
    
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask_uint8, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare for drawing
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Prepare classification transform
    cls_transform = get_classification_transform()
    
    # 1. Collect initial boxes from contours
    initial_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100: 
            continue
        rect = cv2.boundingRect(cnt) # Returns (x, y, w, h)
        initial_boxes.append(rect)
        
    # 2. Merge overlapping or nearby boxes
    # Threshold is in pixels. Since we are on original image scale, 30-50px is reasonable.
    final_boxes = merge_boxes(initial_boxes, threshold=30)

    box_count = 0
    for (x, y, w, h) in final_boxes:
        # --- Classification Step ---
        # Crop the sign from the PIL image
        # PIL crop is (left, upper, right, lower)
        crop_img = pil_img.crop((x, y, x + w, y + h))
        
        # Transform and predict
        input_tensor_cls = cls_transform(crop_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output_cls = cls_model(input_tensor_cls)
            probs = torch.nn.functional.softmax(output_cls, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            predicted_label = class_names[predicted_idx.item()]
            conf_pct = confidence.item() * 100
        
        # Draw Bounding Box
        cv2.rectangle(opencv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw Label
        label_text = f"{predicted_label} {conf_pct:.1f}%"
        cv2.putText(opencv_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        box_count += 1
        
    # Save result
    cv2.imwrite(output_path, opencv_img)
    print(f"Saved {output_path} (Found {box_count} items)")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Segmentation Model
    print(f"Loading segmentation model from {SEG_MODEL_PATH}...")
    seg_model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    if os.path.exists(SEG_MODEL_PATH):
        try:
            seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
            seg_model.eval()
        except Exception as e:
            print(f"Error loading seg model: {e}")
            return
    else:
        print(f"Segmentation model not found at {SEG_MODEL_PATH}")
        return

    # Load Classification Model
    cls_model = load_classification_model()
    if cls_model is None:
        return

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp"]
    input_images = []
    for ext in image_extensions:
        input_images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    print(f"Found {len(input_images)} images. Processing...")
    
    for img_path in input_images:
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"result_annotated_{filename}")
        process_image(seg_model, cls_model, img_path, save_path)

if __name__ == '__main__':
    main()
