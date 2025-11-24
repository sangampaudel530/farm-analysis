import gc
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from val import validate
from models.losses import CustomLoss
from utils.earlystopping import EarlyStopping
# from datasets.dataset import FarmDataset


def train(model, train_loader, val_loader, config):
    """
    Performs full training over all epochs.
    Returns trained model.
    """

    # clear gpu cache
    torch.cuda.empty_cache()
    gc.collect()

    criterion = CustomLoss()
    optimizer = AdamW(model.parameters(), lr=config["LR"], weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()

    early_stopping = EarlyStopping(
        patience=config["EARLY_STOPPING_PATIENCE"],
        verbose=True,
        path=config["MODEL_SAVE_PATH"]
    )

    # Training Loop
    for epoch in range(config["EPOCHS"]):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['EPOCHS']}")
        epoch_loss = 0

        for images, masks in loop:
            images = images.to(config["DEVICE"])
            masks = masks.to(config["DEVICE"])

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        # ---- VALIDATION CALL ----
        val_loss, avg_iou, avg_dice = validate(model, val_loader, criterion, config)

        print(f"Val Loss: {val_loss:.4f} | mIoU: {avg_iou:.4f} | mDice: {avg_dice:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        early_stopping(avg_iou, model)
        if early_stopping.early_stop:
            print("⚠️ Early stopping triggered!")
            break

    return model
