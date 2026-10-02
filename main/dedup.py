# -*- coding: utf-8 -*-
"""
Cross-tile deduplication.

make_tiles produces overlapping tiles, so a colony sitting in an overlap
band is detected twice. For each pair of neighbouring tiles this module
decides, per detection, whether it is a duplicate to drop, a fragment to
merge with its counterpart in the neighbour, or a clean detection to keep.

Operates in-place on the {(row, col): Tile} dict held by Plate.tiles.
"""

from dataclasses import replace


# ----------------------------------------------------------------------
# BOX GEOMETRY   (all boxes are (y1, x1, y2, x2))
# ----------------------------------------------------------------------

def iou(a, b):
    """Intersection over union of two boxes."""
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b

    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ay2 - ay1) * max(0, ax2 - ax1)
    area_b = max(0, by2 - by1) * max(0, bx2 - bx1)
    union = area_a + area_b - inter

    return 0.0 if union == 0 else inter / union


def union_box(a, b):
    """Smallest box containing both inputs."""
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    return (min(ay1, by1), min(ax1, bx1), max(ay2, by2), max(ax2, bx2))


# ----------------------------------------------------------------------
# PAIRWISE RESOLUTION
# ----------------------------------------------------------------------

def resolve_pair(primary, secondary, axis="x", tol=5):
    """
    Resolve duplicates between two neighbouring tiles, in-place.

    primary is the tile that comes first along the axis; secondary is its
    neighbour. Detections touching the shared edge are classified as:

      drop   - the box lies wholly inside the overlap band, so the
               neighbour holds the better copy
      merge  - the box is clipped by the edge, so it is a fragment and is
               unioned with the closest fragment in the neighbour
      keep   - everything else

    Parameters
    ----------
    primary, secondary : Tile
    axis               : "x" for left/right neighbours, "y" for top/bottom
    tol                : px tolerance when testing "touches the edge"

    Returns
    -------
    (dropped, merged) : counts, for the caller to aggregate
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'.")

    px0, py0, px1, py1 = primary.bbox
    sx0, sy0, sx1, sy1 = secondary.bbox

    def near(a, b):
        return abs(a - b) <= tol

    dropped = 0

    # -- primary: detections running into the trailing edge ---------------
    p_keep, p_merge = [], []
    for col in primary.colonies:
        y1, x1, y2, x2 = col.roi_global

        if axis == "x":
            at_edge, beyond = near(x2, px1), x1 > sx0
        else:
            at_edge, beyond = near(y2, py1), y1 > sy0

        if not at_edge:
            p_keep.append(col)
        elif beyond:
            dropped += 1
        else:
            p_merge.append(col)

    # -- secondary: detections behind the edge, or on the leading edge -----
    s_keep, s_merge = [], []
    for col in secondary.colonies:
        y1, x1, y2, x2 = col.roi_global

        if axis == "x":
            behind, at_edge = x2 < px1, near(x1, sx0)
        else:
            behind, at_edge = y2 < py1, near(y1, sy0)

        if behind:
            dropped += 1
        elif at_edge:
            s_merge.append(col)
        else:
            s_keep.append(col)

    # -- pair up fragments by best IoU -------------------------------------
    merged = 0
    for col in p_merge:
        if not s_merge:
            p_keep.append(col)
            continue

        best = max(range(len(s_merge)),
                   key=lambda i: iou(col.roi_global, s_merge[i].roi_global))
        partner = s_merge.pop(best)

        p_keep.append(replace(
            col,
            roi_global=union_box(col.roi_global, partner.roi_global),
            score=max(col.score, partner.score),
        ))
        merged += 1

    primary.colonies = p_keep
    secondary.colonies = s_keep + s_merge

    return dropped, merged


# ----------------------------------------------------------------------
# GRID SWEEP
# ----------------------------------------------------------------------

def resolve_duplicates(tiles, tol=5, verbose=True):
    """
    Sweep every neighbouring tile pair: horizontal first, then vertical.

    The grid shape is inferred from the (row, col) keys. Updates tiles
    in-place and prints one aggregate summary.

    Parameters
    ----------
    tiles   : dict {(row, col): Tile}
    tol     : px tolerance passed to resolve_pair
    verbose : print the summary

    Returns
    -------
    summary : list of per-pair records, e.g. for pd.DataFrame(summary)
    """
    if not tiles:
        return []

    rows = max(r for (r, _) in tiles) + 1
    cols = max(c for (_, c) in tiles) + 1

    # (axis, primary key, secondary key) for every adjacent pair
    pairs = [("x", (r, c), (r, c + 1))
             for r in range(rows) for c in range(cols - 1)]
    pairs += [("y", (r, c), (r + 1, c))
              for r in range(rows - 1) for c in range(cols)]

    summary = []
    for axis, a, b in pairs:
        if a not in tiles or b not in tiles:
            continue

        dropped, merged = resolve_pair(tiles[a], tiles[b], axis=axis, tol=tol)
        summary.append({"pair": (a, b), "axis": axis,
                        "dropped": dropped, "merged": merged})

    if verbose:
        print(f"Tile pairs processed: {len(summary)}")
        print(f"dropped={sum(s['dropped'] for s in summary)}, "
              f"merged={sum(s['merged'] for s in summary)}")

    return summary