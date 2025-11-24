import segmentation_models_pytorch as smp
from configs.training_config import CONFIG
import torch.nn as nn

class CustomLoss(nn.Module):

    """Combined Dice and Focal Loss for multiclass segmentation.
    Focal Loss handles class imbalance(give equal importance to all classes)
    Weights: 0.4 * Dice + 0.6 * Focal
    """

    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=CONFIG["IGNORE_INDEX"])
        self.focal = smp.losses.FocalLoss(mode="multiclass", ignore_index=CONFIG["IGNORE_INDEX"])

    def forward(self, y_pred, y_true):
        return 0.4 * self.dice(y_pred, y_true) + 0.6 * self.focal(y_pred, y_true)

# Example usage:
# if __name__ == "__main__":
#     loss_fn = CustomLoss()
