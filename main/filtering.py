# -*- coding: utf-8 -*-
"""
Score filtering.

The detector is run with a permissive confidence threshold so that faint
colonies survive into the tile dicts; the final cut is made here.

Optionally the threshold is adapted to colony density: sparse plates
tolerate a looser cut, crowded plates need a stricter one, because false
positives scale with the number of candidate objects. The adaptation is a
straight line through two anchor points, expressed as deltas on the base
threshold.

Operates in-place on the {(row, col): Tile} dict held by Plate.tiles.
"""

MIN_COUNT = 10
MAX_COUNT = 300


# ----------------------------------------------------------------------
# REGRESSION
# ----------------------------------------------------------------------

def _line(base_score, low_delta, high_delta,
          min_count=MIN_COUNT, max_count=MAX_COUNT):
    """
    Slope and intercept of the line satisfying:
        y(min_count) = base_score + low_delta
        y(max_count) = base_score + high_delta
    """
    y_low = base_score + low_delta
    y_high = base_score + high_delta

    slope = (y_high - y_low) / (max_count - min_count)
    intercept = y_low - slope * min_count

    return slope, intercept


def regression_from_score(base_score, low_delta=None, high_delta=None,
                          min_count=MIN_COUNT, max_count=MAX_COUNT):
    """
    Human-readable form of the density correction, for logging at startup.

    Returns None when either delta is unset, i.e. when no correction is
    applied.
    """
    if low_delta is None or high_delta is None:
        return None

    slope, intercept = _line(base_score, low_delta, high_delta,
                             min_count, max_count)
    return f"corrected_score = ({slope:.6f} * count) + {intercept:.4f}"


# ----------------------------------------------------------------------
# FILTERING
# ----------------------------------------------------------------------

def filter_by_score(tiles, threshold, score_regression=None,
                    min_count=MIN_COUNT, max_count=MAX_COUNT,
                    verbose=False):
    """
    Drop colonies scoring below the threshold, in-place.

    With score_regression set, the cut is made twice: a first pass counts
    the survivors of the base threshold, that count fixes the corrected
    threshold, and the corrected threshold is then applied to the original
    detections — not to the first pass's output.

    Parameters
    ----------
    tiles            : dict {(row, col): Tile}
    threshold        : float, base confidence cut
    score_regression : (low_delta, high_delta), or None to disable
    min_count        : count anchoring low_delta
    max_count        : count anchoring high_delta
    verbose          : print the derived line and count

    Returns
    -------
    threshold : float, the cut actually applied
    """
    thr = float(threshold)

    if not score_regression or None in score_regression:
        for t in tiles.values():
            t.colonies = [c for c in t.colonies if c.score >= thr]
        return thr

    # -- first pass: count only, leave tiles untouched ---------------------
    count = sum(sum(1 for c in t.colonies if c.score >= thr)
                for t in tiles.values())

    # -- derive the corrected threshold from that count --------------------
    low_delta, high_delta = score_regression
    slope, intercept = _line(thr, low_delta, high_delta,
                             min_count, max_count)
    corrected = min(1.0, max(0.0, slope * count + intercept))

    if verbose:
        print(f"[score_regression] base={thr:.4f}, "
              f"low_delta={low_delta:+.4f}, high_delta={high_delta:+.4f}\n"
              f"                   slope={slope:.6f}, intercept={intercept:.6f}\n"
              f"                   count={count}, corrected={corrected:.4f}")

    # -- second pass: apply the corrected threshold to the originals -------
    for t in tiles.values():
        t.colonies = [c for c in t.colonies if c.score >= corrected]

    return corrected