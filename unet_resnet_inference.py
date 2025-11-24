import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from matplotlib.patches import Patch

# ==========================================
# CONFIG
# ==========================================
CONFIG = {
    "MODEL_PATH": "./models/resnet34_unet_best_4classes.pth",
    "IMG_SIZE": 512,
    "NUM_CLASSES": 4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==========================================
# LABELS + COLORS
# ==========================================
CLASS_COLORS = {
    0: [0, 0, 0],          # Background
    1: [34, 139, 34],      # Vegetation
    2: [124, 252, 0],      # Farmland
    3: [255, 215, 0],      # Sand
}

ID2LABEL = {
    0: "Background",
    1: "Vegetation",
    2: "Farmland",
    3: "Sand"
}

# ==========================================
# MODEL
# ==========================================
def build_model(num_classes):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
        activation=None
    )

# ==========================================
# REMAP GT MASK (same as training)
# ==========================================
def remap_gt_mask(mask):
    new_mask = np.zeros_like(mask)
    new_mask[(mask == 1) | (mask == 2) | (mask == 5)] = 1  # vegetation merged
    new_mask[mask == 3] = 2                                # farmland
    new_mask[mask == 4] = 3                                # sand
    return new_mask

# ==========================================
# COLORIZE PRED / GT MASK
# ==========================================
def colorize_mask(mask_2d):
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        rgb[mask_2d == cls_id] = color
    return rgb

# ==========================================
# PREDICT SINGLE IMAGE
# ==========================================
def get_prediction(model, image_np):
    transform = A.Compose([
        A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    augmented = transform(image=image_np)
    tensor_img = augmented["image"].unsqueeze(0).to(CONFIG["DEVICE"])

    with torch.no_grad():
        logits = model(tensor_img)

        # Resize logits back to original image size
        logits = F.interpolate(
            logits,
            size=image_np.shape[:2],
            mode="bilinear",
            align_corners=False
        )

        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    return pred_mask

# ==========================================
# VISUALIZE SINGLE IMAGE
# ==========================================
def test_single(image_path, mask_path=None):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("Loading ResNet-34 UNet model...")

    # Load Model
    model = build_model(CONFIG["NUM_CLASSES"])
    if os.path.exists(CONFIG["MODEL_PATH"]):
        model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"]))
    else:
        print(f"Error: Weights not found at {CONFIG['MODEL_PATH']}")
        return
    
    model.to(CONFIG["DEVICE"])
    model.eval()
    print("✓ Model loaded!")

    # 1. Load Image
    image_np = np.array(Image.open(image_path).convert("RGB"))

    # 2. Check and Load GT Mask
    has_gt = False
    gt_color = None
    
    if mask_path and os.path.exists(mask_path):
        has_gt = True
        raw_mask = np.array(Image.open(mask_path).convert("L"))
        gt_mask = remap_gt_mask(raw_mask)
        
        # Resize GT to match image for visualization if dimensions differ (optional safety)
        if gt_mask.shape != image_np.shape[:2]:
             gt_mask = np.array(Image.fromarray(gt_mask).resize(
                 (image_np.shape[1], image_np.shape[0]), resample=Image.NEAREST))
                 
        gt_color = colorize_mask(gt_mask)
    elif mask_path:
        print(f"Warning: Mask path provided but not found: {mask_path}")

    # 3. Prediction
    pred_mask = get_prediction(model, image_np)
    pred_color = colorize_mask(pred_mask)

    # 4. Display
    # Determine columns: 3 if GT exists, else 2
    cols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))

    # Setup axes array (ensure it's iterable if cols=1, though unlikely)
    if cols == 1: axes = [axes]

    # Plot Original Image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    if has_gt:
        # Plot GT
        axes[1].imshow(gt_color)
        axes[1].set_title("Ground Truth (remapped)")
        axes[1].axis("off")
        
        # Plot Prediction
        axes[2].imshow(pred_color)
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")
    else:
        # Plot Prediction immediately
        axes[1].imshow(pred_color)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

    # Legend
    legend_items = [Patch(facecolor=np.array(CLASS_COLORS[i]) / 255, label=ID2LABEL[i])
                    for i in CLASS_COLORS.keys()]

    fig.legend(handles=legend_items,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.05),
               ncol=4,
               fontsize=12)

    plt.tight_layout()
    plt.show()


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    # Example 1: With Mask (Make sure paths exist)
    test_single(
        image_path="./test_images/img_.jpg",
        mask_path="./test_images/mask_.png"
    )
    
    # Example 2: Without Mask (pass None)
    # test_single(
    #     image_path="./test_images/img1.jpeg",
    #     mask_path=None
    # )