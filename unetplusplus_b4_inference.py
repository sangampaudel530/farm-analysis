import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
CONFIG = {
    "IMG_SIZE": 512,
    "NUM_CLASSES": 4,             # 0:Back, 1:Veg, 2:Farm, 3:Sand
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MODEL_PATH": "./models/unetplusplus_b4_best.pth",
    "ENCODER": "efficientnet-b4",
    "ENCODER_WEIGHTS": "imagenet"
}

# Define colors for visualization [R, G, B]
CLASS_COLORS = np.array([
    [0, 0, 0],       # 0: Background (Black)
    [0, 255, 0],     # 1: Vegetation (Green)
    [139, 69, 19],   # 2: Farm (Brown)
    [255, 255, 0],   # 3: Sand (Yellow)
])

CLASS_NAMES = ["Background", "Vegetation", "Farm", "Sand"]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def decode_segmap(mask):
    """Converts a 2D segmentation mask to RGB."""
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for label_idx in range(CONFIG['NUM_CLASSES']):
        idx = (mask == label_idx)
        r[idx] = CLASS_COLORS[label_idx, 0]
        g[idx] = CLASS_COLORS[label_idx, 1]
        b[idx] = CLASS_COLORS[label_idx, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def process_mask(mask_path):
    """Loads mask and applies class mapping."""
    mask = np.array(Image.open(mask_path).convert("L"))
    new_mask = np.zeros_like(mask)
    
    # Mapping logic
    new_mask[(mask == 1) | (mask == 2) | (mask == 5)] = 1 # Veg
    new_mask[mask == 3] = 2 # Farm
    new_mask[mask == 4] = 3 # Sand
    
    # Ignore/Background
    new_mask[mask == 6] = 0 
    new_mask[mask > 6] = 0

    return new_mask

def get_inference_transform():
    return A.Compose([
        A.Resize(height=CONFIG['IMG_SIZE'], width=CONFIG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ==========================================
# 3. SINGLE IMAGE INFERENCE
# ==========================================
def predict_single_image(image_path, mask_path=None):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"--- Processing: {os.path.basename(image_path)} ---")

    # 1. Load Model
    model = smp.UnetPlusPlus(
        encoder_name=CONFIG['ENCODER'],
        encoder_weights=None,
        in_channels=3,
        classes=CONFIG['NUM_CLASSES']
    ).to(CONFIG['DEVICE'])

    if os.path.exists(CONFIG['MODEL_PATH']):
        model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
    else:
        print(f"Error: Model weights not found at {CONFIG['MODEL_PATH']}")
        return

    model.eval()
    transform = get_inference_transform()

    # 2. Load and Preprocess Image
    original_img = np.array(Image.open(image_path).convert("RGB"))
    # Resize original for display consistency
    display_img = np.array(Image.fromarray(original_img).resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])))

    augmentations = transform(image=original_img)
    input_tensor = augmentations["image"].unsqueeze(0).to(CONFIG['DEVICE'])

    # 3. Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pred_rgb = decode_segmap(pred_mask)

    # 4. Handle Ground Truth Mask
    has_gt = False
    gt_rgb = None
    
    if mask_path is not None and os.path.exists(mask_path):
        has_gt = True
        gt_mask_raw = process_mask(mask_path)
        gt_mask_resized = np.array(Image.fromarray(gt_mask_raw).resize(
            (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']), resample=Image.NEAREST
        ))
        gt_rgb = decode_segmap(gt_mask_resized)
    elif mask_path is not None:
        print(f"Warning: Mask path provided but file not found: {mask_path}")

    # 5. Visualization
    # If GT exists: 3 columns. If not: 2 columns.
    cols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    
    # Ensure axes is iterable even if cols=1 (though unlikely here)
    if cols == 1: axes = [axes]

    # Plot Original
    axes[0].imshow(display_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    if has_gt:
        # Plot GT
        axes[1].imshow(gt_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        # Plot Pred
        axes[2].imshow(pred_rgb)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
    else:
        # Plot Pred only
        axes[1].imshow(pred_rgb)
        axes[1].set_title("Prediction")
        axes[1].axis("off")

    # --- Add Legend ---
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i]/255.0, label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    fig.legend(handles=patches, loc='upper center', ncol=len(CLASS_NAMES),
               bbox_to_anchor=(0.5, 1.05), fontsize=12, frameon=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    
    # Define paths for single inference here
    SINGLE_IMG_PATH = "./test_images/img_.jpg"  # <--- Change this
    SINGLE_MASK_PATH = "./test_images/mask_.png" # <--- Change this (set to None if no mask)


    # Case 1: With Mask
    predict_single_image(SINGLE_IMG_PATH, SINGLE_MASK_PATH)
    
    # Case 2: Without Mask (pass None or invalid path)
    # predict_single_image(SINGLE_IMG_PATH, None)