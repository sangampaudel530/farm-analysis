import os

# ---------------- CONFIG ----------------
pre_path = "./tile_2048_37376.tif"
post_path = "./tile_2048_37376 (1).tif"

base_dir = "./tiles_output"
pre_tif_dir = os.path.join(base_dir, "preTileTif")
post_tif_dir = os.path.join(base_dir, "postTileTif")
pre_png_dir = os.path.join(base_dir, "preTilePng")
post_png_dir = os.path.join(base_dir, "postTilePng")

tile_size = 32
stride = 8

# Strict thresholds
MAX_BAD_PERCENT = 0.1
MIN_STD_DEV = 2.0

# Ensure folders exist
os.makedirs(pre_tif_dir, exist_ok=True)
os.makedirs(post_tif_dir, exist_ok=True)
os.makedirs(pre_png_dir, exist_ok=True)
os.makedirs(post_png_dir, exist_ok=True)
