import torch

CONFIG = {
    "ROOT_DIR": "./data",
    "IMG_SIZE": 512,
    "BATCH_SIZE": 1,
    "LR": 2e-4,
    "EPOCHS": 50,
    "NUM_CLASSES": 4,             # 0:Back, 1:Veg, 2:Farm, 3:Sand
    "IGNORE_INDEX": 255,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MODEL_SAVE_PATH": "unetplusplus_b4_best.pth",   
    # "MODEL_SAVE_PATH": "unet_resnet34_best.pth",
    "ENCODER": "efficientnet-b4",
    # "ENCODER": "resnet34",
    "ENCODER_WEIGHTS": "imagenet",
    "EARLY_STOPPING_PATIENCE": 12  # Stop if no improvement for 12 epochs
}