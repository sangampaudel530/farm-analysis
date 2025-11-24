import torch
import numpy as np

def compute_metrics(preds, targets, num_classes, ignore_index=255):
    """Computes Mean IoU and Mean Dice ignoring the ignore_index"""
    pred_mask = torch.argmax(preds, dim=1)
    iou_list = []
    dice_list = []

    for c in range(num_classes):
        p_c = (pred_mask == c)
        t_c = (targets == c)

        valid_mask = (targets != ignore_index)
        p_c = p_c & valid_mask
        t_c = t_c & valid_mask

        intersection = (p_c & t_c).sum().float()
        union = (p_c | t_c).sum().float()
        dice_union = p_c.sum().float() + t_c.sum().float()

        # IoU Calculation
        if union == 0:
            iou = 1.0 if t_c.sum() == 0 else 0.0
        else:
            # Convert tensor to float item here
            iou = (intersection / (union + 1e-6)).item()

        # We strictly append the float value now
        iou_list.append(iou)

        # Dice Calculation
        if dice_union == 0:
            dice = 1.0 if t_c.sum() == 0 else 0.0
        else:
            # Convert tensor to float item here
            dice = ((2.0 * intersection) / (dice_union + 1e-6)).item()

        # We strictly append the float value now
        dice_list.append(dice)

    return np.mean(iou_list), np.mean(dice_list)