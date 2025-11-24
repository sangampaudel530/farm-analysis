from rasterio.windows import Window
import numpy as np

def window_intersection_safe(win1, win2):
    col_off = max(win1.col_off, win2.col_off)
    row_off = max(win1.row_off, win2.row_off)
    col_end = min(win1.col_off + win1.width, win2.col_off + win2.width)
    row_end = min(win1.row_off + win1.height, win2.row_off + win2.height)

    width = col_end - col_off
    height = row_end - row_off

    if width <= 0 or height <= 0:
        return None

    return Window(
        int(np.floor(col_off)),
        int(np.floor(row_off)),
        int(np.ceil(width)),
        int(np.ceil(height))
    )
