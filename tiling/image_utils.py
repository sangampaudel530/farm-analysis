import numpy as np
from PIL import Image

def array_to_image(data):
    img_array = np.moveaxis(data, 0, -1)

    if img_array.shape[-1] > 3:
        img_array = img_array[:, :, :3]

    if img_array.dtype in [np.float32, np.float64]:
        if np.max(img_array) <= 1.5:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    return Image.fromarray(img_array)
