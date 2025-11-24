import os
import shutil

def organize_files(src_directory, img_directory, mask_directory):
    """
    Organizes files from the source directory into image and mask directories.

    Parameters:
    - src_directory (str): Path to the source directory containing mixed files.
    - img_directory (str): Path to the destination directory for images.
    - mask_directory (str): Path to the destination directory for masks.
    """

    # Create destination folders if they don't exist
    os.makedirs(img_directory, exist_ok=True)
    os.makedirs(mask_directory, exist_ok=True)

    # Move files
    for f in os.listdir(src_directory):
        src_path = os.path.join(src_directory, f)
        if f.endswith(".jpg"):
            shutil.copy(src_path, os.path.join(img_directory, f))
        elif f.endswith("_mask.png"):
            shutil.copy(src_path, os.path.join(mask_directory, f))

    print(f"✅ All images moved to {img_directory}")
    print(f"✅ All masks moved to {mask_directory}")


# Example usage:
# if __name__ == "__main__":
#     organize_files(
#         src_directory="data/agrifarm-3zwm0-15/train",
#         img_directory="data/images/train",
#         mask_directory="data/masks/train"
#     )
