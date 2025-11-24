import albumentations as A
from albumentations.pytorch import ToTensorV2


# Training transforms with data augmentation
def get_train_transform(config): 
    return A.Compose([
        A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20,
            val_shift_limit=10, p=0.3
        ),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.3
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Validation transforms (only resizing and normalization)
def get_val_transform(config):
    return A.Compose([
        A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
