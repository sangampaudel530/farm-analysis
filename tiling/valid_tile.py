import numpy as np

def is_tile_valid(data, max_bad_percent=0.1, min_std=2.0):
    """
    Strict quality check for tile:
    - Rejects tiles with too many black/white pixels.
    - Rejects low-variance (flat) tiles.
    """
    # Use RGB channels only
    rgb = data[:3, :, :] if data.shape[0] >= 3 else data

    c, h, w = rgb.shape
    total_pixels = h * w

    black_mask = np.all(rgb <= 5, axis=0)
    white_mask = np.all(rgb >= 250, axis=0)

    count_bad = np.sum(black_mask) + np.sum(white_mask)
    bad_percent = (count_bad / total_pixels) * 100

    if bad_percent > max_bad_percent:
        return False

    if np.std(rgb) < min_std:
        return False

    return True
