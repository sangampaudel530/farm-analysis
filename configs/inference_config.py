import torch

CONFIG = {
    "TILE_SIZE": 512,
    "NUM_CLASSES": 4, 
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "R_IDX": 0, "G_IDX": 1, "B_IDX": 2, "NIR_IDX": 3,
    "NDVI_LOSS_THRESHOLD": 0.15,
    "NDWI_GAIN_THRESHOLD": 0.1, 
    "DEFAULT_PIXEL_AREA_M2": 0.25 
}