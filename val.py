import torch
import numpy as np
from torch.cuda.amp import autocast
from utils.metrics import compute_metrics

def validate(model, val_loader, criterion, config):
    """
    Runs validation loop and returns:
    - avg_val_loss
    - avg_mIoU
    - avg_mDice
    """

    model.eval()
    val_loss = 0
    ious = []
    dices = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(config["DEVICE"])
            masks = masks.to(config["DEVICE"])

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()
            mIoU, mDice = compute_metrics(
                outputs, masks,
                config["NUM_CLASSES"],
                config["IGNORE_INDEX"]
            )

            ious.append(mIoU)
            dices.append(mDice)

    avg_val_loss = val_loss / len(val_loader)
    avg_iou = np.mean(ious)
    avg_dice = np.mean(dices)

    return avg_val_loss, avg_iou, avg_dice



# if __name__ == "__main__":
#     from configs.training_config import CONFIG
#     from models.losses import CustomLoss
#     criterion = CustomLoss()
#     validate(model=None, val_loader=None, criterion=criterion, config=CONFIG)