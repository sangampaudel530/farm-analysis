import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class FarmDataset(Dataset):
    def __init__(self, root_dir, split="train", config=None, transform=None):
        """
        Dataset for farm segmentation images.

        root_dir: dataset root directory
        split: "train", "val", or "test"
        transform: Albumentations transform pipeline (optional)
        """

        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")

        # Load image list safely
        if os.path.exists(self.img_dir):
            self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".jpg")])
        else:
            self.images = []
            print(f"⚠ Warning: Image directory not found: {self.img_dir}")

        self.transform = transform   # <-- Pass transform directly here
        self.config = config if config is not None else {}
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".jpg", "_mask.png")

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # 4-class remapping
        new_mask = np.zeros_like(mask)
        new_mask[(mask == 1) | (mask == 2) | (mask == 5)] = 1  # Veg
        new_mask[mask == 3] = 2                                 # Farm
        new_mask[mask == 4] = 3                                 # Sand
        new_mask[mask == 6] = self.config["IGNORE_INDEX"]
        new_mask[mask > 6] = self.config["IGNORE_INDEX"]

        if self.transform:
            augmented = self.transform(image=image, mask=new_mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()


# if __name__ == "__main__":
#     # Simple test to verify dataset loading
#     dataset = FarmDataset(root_dir="./data/train", split="train", config={"IGNORE_INDEX": 0})
#     print(f"Dataset size: {len(dataset)}")

#     sample_img, sample_mask = dataset[0]
#     print(f"Sample image shape: {sample_img.shape}")
#     print(f"Sample mask shape: {sample_mask.shape}")