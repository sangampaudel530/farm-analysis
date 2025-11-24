import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from training import train
from datasets.dataset import FarmDataset
from configs.training_config import CONFIG
from models.unetplusplus_b4_model import get_model
from models.unet_resnet34 import build_model
from utils.dataset import get_data
from utils.file_organizer import organize_files
from datasets.transforms import get_train_transform, get_val_transform
from val import validate

if __name__ == "__main__":

    # Data download from roboflow
    print("--- Checking Dataset ---")
    get_data()
    print("--- Dataset Ready ---")

    # organize folder
    organize_files("./AGRIFARM-15/train", "./data/train/images", "./data/train/masks")
    organize_files("./AGRIFARM-15/valid", "./data/valid/images", "./data/valid/masks")
    organize_files("./AGRIFARM-15/test", "./data/test/images", "./data/test/masks")
    
    print(f"--- Training Unet++ with {CONFIG['ENCODER']} ---")

    # ----- Model -----
    model = get_model(CONFIG)
    # Alternatively, to use ResNet34 U-Net:
    # model = build_model(CONFIG)

    train_transform = get_train_transform(CONFIG)
    val_transform = get_val_transform(CONFIG)


    # ----- Dataset -----
    train_ds = FarmDataset(CONFIG["ROOT_DIR"], split="train",config=CONFIG, transform=train_transform)
    
    if os.path.exists(os.path.join(CONFIG["ROOT_DIR"], "valid")):
        val_ds = FarmDataset(CONFIG["ROOT_DIR"], split="valid", config=CONFIG, transform=val_transform)
    else:
        print("No validation folder found — splitting train dataset...")
        train_size = int(0.85 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])


    # ----- Dataloaders -----
    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)


    # # ----- Start Training -----
    trained_model = train(model, train_loader, val_loader, CONFIG)
