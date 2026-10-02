# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:31:18 2026

@author: Admin
"""



from main.data import Tile

"""
Tiling of the cropped plate image.

Produces a grid of overlapping tiles as main.data.Tile objects, keyed by
(row, col). Tile geometry is stored as bbox = (x0, y0, x1, y1) in
cropped-plate coordinates; this offset is what the detector later uses to
lift tile-local ROIs into plate coordinates.
"""




def make_tiles(plate, grid=(2, 2), overlap=0.2):
    """
    Split an image into a grid of equally sized, overlapping tiles.

    Parameters
    ----------
    img     : uint8 BGR image (typically plate.cropped).
    grid    : (rows, cols).
    overlap : Fractional overlap relative to tile width/height.
              0.2 means 20% overlap. Must be in [0, 1).

    """

    # -- working image -----------------------------------------------------
    img = plate.cropped if plate.cropped is not None else plate.image
    if img is None:
        raise ValueError(f"Plate '{plate.sample_id}' has no image to tile.")

    # -- offset from cropped coords back to original coords ----------------
    ox, oy = (plate.crop_bbox[0], plate.crop_bbox[1]) if plate.crop_bbox else (0, 0)

    H, W = img.shape[:2]
    rows, cols = grid

    p = float(overlap)
    if not (0 <= p < 1):
        raise ValueError("overlap must be in [0, 1).")
    if rows < 1 or cols < 1:
        raise ValueError("grid must have at least 1 row and 1 col.")

    # -- size of a single tile such that the overlapping grid spans the image
    tile_w = int(W / (cols - p * (cols - 1)))
    tile_h = int(H / (rows - p * (rows - 1)))

    step_w = int(tile_w * (1 - p))
    step_h = int(tile_h * (1 - p))

    plate.tiles.clear()

    for r in range(rows):
        for c in range(cols):
            # local slice bounds, into img
            x0 = c * step_w
            y0 = r * step_h
            x1 = min(x0 + tile_w, W)
            y1 = min(y0 + tile_h, H)

            plate.tiles[(r, c)] = Tile(
                row=r,
                col=c,
                bbox=(x0 + ox, y0 + oy, x1 + ox, y1 + oy),  # original coords
                image=img[y0:y1, x0:x1].copy(),             # local slice
            )





    
    



