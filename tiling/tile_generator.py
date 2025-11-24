import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds
import os
import numpy as np
from tqdm import tqdm

from valid_tile import is_tile_valid
from image_utils import array_to_image
from window_utils import window_intersection_safe
import settings


def generate_tiles():
    print("--- Starting Strict Tile Generation ---")

    try:
        with rasterio.open(settings.pre_path) as src_pre, rasterio.open(settings.post_path) as src_post:
            width, height = src_pre.width, src_pre.height

            x_coords = range(0, width - settings.tile_size + 1, settings.stride)
            y_coords = range(0, height - settings.tile_size + 1, settings.stride)
            total_tiles_potential = len(x_coords) * len(y_coords)

            pre_profile = src_pre.profile.copy()
            saved_count = 0

            with tqdm(total=total_tiles_potential, desc="Filtering Tiles") as pbar:
                for y in y_coords:
                    for x in x_coords:
                        pbar.update(1)

                        window = Window(x, y, settings.tile_size, settings.tile_size)

                        # Read pre tile
                        pre_tile_data = src_pre.read(window=window)

                        # Validate pre tile
                        if not is_tile_valid(pre_tile_data, settings.MAX_BAD_PERCENT, settings.MIN_STD_DEV):
                            continue

                        # Compute post-tile window
                        tile_bounds = src_pre.window_bounds(window)
                        tile_bounds_post_crs = transform_bounds(src_pre.crs, src_post.crs, *tile_bounds)
                        post_window_raw = from_bounds(*tile_bounds_post_crs, transform=src_post.transform)
                        full_post_window = Window(0, 0, src_post.width, src_post.height)

                        post_window_intersect = window_intersection_safe(post_window_raw, full_post_window)
                        if post_window_intersect is None:
                            continue

                        # Reproject post tile
                        post_tile_aligned = np.zeros((src_pre.count, settings.tile_size, settings.tile_size), dtype=src_pre.dtypes[0])
                        dst_transform = src_pre.window_transform(window)

                        reproject(
                            source=rasterio.band(src_post, range(1, src_post.count + 1)),
                            destination=post_tile_aligned,
                            src_transform=src_post.transform,
                            src_crs=src_post.crs,
                            dst_transform=dst_transform,
                            dst_crs=src_pre.crs,
                            resampling=Resampling.bilinear,
                            src_window=post_window_intersect
                        )

                        # Validate post tile
                        if not is_tile_valid(post_tile_aligned, settings.MAX_BAD_PERCENT, settings.MIN_STD_DEV):
                            continue

                        # Names
                        tif_name = f"tile_{y}_{x}.tif"
                        png_name = f"tile_{y}_{x}.png"

                        # Prepare metadata
                        out_meta = pre_profile.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": settings.tile_size,
                            "width": settings.tile_size,
                            "transform": dst_transform,
                            "count": src_pre.count,
                            "compress": "lzw"
                        })

                        # Save TIF files
                        with rasterio.open(os.path.join(settings.pre_tif_dir, tif_name), "w", **out_meta) as dst:
                            dst.write(pre_tile_data)

                        with rasterio.open(os.path.join(settings.post_tif_dir, tif_name), "w", **out_meta) as dst:
                            dst.write(post_tile_aligned)

                        # Save PNG
                        array_to_image(pre_tile_data).save(os.path.join(settings.pre_png_dir, png_name))
                        array_to_image(post_tile_aligned).save(os.path.join(settings.post_png_dir, png_name))

                        saved_count += 1

    except Exception as e:
        print(f"❌ Error: {e}")

    print(f"\nDone. Saved {saved_count} clean pre–post tile pairs.")
