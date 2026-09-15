"""
Stage 0 — Geometry loader.

Input:  a UIUC-format .dat airfoil coordinate file.
Output: np.ndarray of shape (N, 2), normalized to unit chord, in Selig
        ordering (TE -> upper surface -> LE -> lower surface -> TE), with
        the trailing edge closed (first point == last point).

Handles two classes of malformed input:
  - inconsistent point ordering (e.g. lower surface listed first, or the
    whole file traversed in the opposite direction)
  - an open trailing edge (first point != last point)

Also defensively rejects/repairs two adversarial cases found probing
Stage 0/1 for troubleshooter-agent scoping (2026-09-14/15), before they
ever reach the far more expensive Stage 1 gmsh attempt:
  - consecutive duplicate points (a digitization artifact) are deduped --
    a repeated coordinate is a zero-length polygon edge, which crashes
    Stage 1's arc-length CubicSpline resampling outright (a raw scipy
    "x must be strictly increasing" ValueError), not just a cosmetic
    defect
  - a genuinely self-intersecting contour is rejected with a clear
    ValueError -- it is unmeshable garbage, not a case for repair, and
    letting it through wastes a full Stage 1 gmsh retry ladder before
    failing anyway with a much less legible gmsh error
"""

import os
import numpy as np

from data.airfoil_downloader import load_dat_file


def _dedupe_consecutive(raw: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate points (keeps the first occurrence)."""
    if len(raw) < 2:
        return raw
    keep = np.ones(len(raw), dtype=bool)
    keep[1:] = np.any(np.diff(raw, axis=0) != 0.0, axis=1)
    return raw[keep]


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if closed segment p1->p2 properly crosses closed segment p3->p4."""

    def orientation(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12
        )

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, p3):
        return True
    if o2 == 0 and on_segment(p1, p2, p4):
        return True
    if o3 == 0 and on_segment(p3, p4, p1):
        return True
    if o4 == 0 and on_segment(p3, p4, p2):
        return True
    return False


def _polygon_has_self_intersections(coords: np.ndarray) -> bool:
    """
    Check every pair of non-adjacent edges of a closed polygon for crossings.

    `coords` must already be closed (coords[0] == coords[-1]) -- edges are
    just consecutive pairs, no extra wraparound edge is added back to
    index 0 (that would be a zero-length duplicate of the TE point and
    produce false positives).
    """
    edges = [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _segments_intersect(*edges[i], *edges[j]):
                return True
    return False


def _reorder_selig(raw: np.ndarray) -> np.ndarray:
    """
    Re-sequence arbitrary-ordered airfoil points into canonical Selig order.

    Assumes `raw` traces a single loop starting at one trailing-edge point
    (upper or lower) and ending at the other, passing through the leading
    edge — true for every UIUC file regardless of which surface is listed
    first or which direction it's traversed in. The leading edge is the
    point with minimum x, which splits the loop into an upper arc and a
    lower arc; the arc with the greater mean y is the upper surface.
    """
    le_idx = int(np.argmin(raw[:, 0]))
    arc_a = raw[: le_idx + 1]  # start -> LE
    arc_b = raw[le_idx:]       # LE -> end

    if arc_a[:, 1].mean() >= arc_b[:, 1].mean():
        upper, lower = arc_a, arc_b
    else:
        upper, lower = arc_b, arc_a

    # Canonical directions: upper goes TE->LE (x decreasing), lower goes LE->TE (x increasing).
    if upper[0, 0] < upper[-1, 0]:
        upper = upper[::-1]
    if lower[0, 0] > lower[-1, 0]:
        lower = lower[::-1]

    if np.array_equal(upper[-1], lower[0]):
        lower = lower[1:]  # drop duplicate LE point at the junction

    return np.vstack([upper, lower])


def _normalize_chord(coords: np.ndarray) -> np.ndarray:
    """Shift so the leading edge sits at x=0 and scale so the chord length is 1.0."""
    x = coords[:, 0]
    x_min, x_max = x.min(), x.max()
    chord = x_max - x_min
    if chord <= 0:
        raise ValueError(f"Degenerate airfoil geometry: chord length {chord} <= 0")

    normalized = coords.copy()
    normalized[:, 0] = (coords[:, 0] - x_min) / chord
    normalized[:, 1] = coords[:, 1] / chord
    return normalized


def _close_trailing_edge(coords: np.ndarray) -> np.ndarray:
    """Snap the first and last points to their shared midpoint, closing any TE gap."""
    if np.array_equal(coords[0], coords[-1]):
        return coords

    closed = coords.copy()
    te_mid = (closed[0] + closed[-1]) / 2.0
    closed[0] = te_mid
    closed[-1] = te_mid
    return closed


def load_airfoil(dat_path: str) -> np.ndarray:
    """
    Run Stage 0 end-to-end: parse -> reorder -> normalize -> close TE.

    Args:
        dat_path: Path to a UIUC-format .dat airfoil coordinate file.

    Returns:
        (N, 2) float64 ndarray, unit-chord, Selig-ordered, TE closed.

    Raises:
        ValueError: if the file is empty, has fewer than 3 points,
                    resolves to a degenerate (zero-chord) geometry, or
                    the contour is genuinely self-intersecting.
    """
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"No such .dat file: {dat_path}")

    _, raw_points = load_dat_file(dat_path)
    if len(raw_points) < 3:
        raise ValueError(f"Need at least 3 coordinate points, got {len(raw_points)}: {dat_path}")

    raw = np.array(raw_points, dtype=np.float64)
    raw = _dedupe_consecutive(raw)
    if len(raw) < 3:
        raise ValueError(
            f"Fewer than 3 distinct coordinate points after deduping "
            f"consecutive duplicates, got {len(raw)}: {dat_path}"
        )
    coords = _reorder_selig(raw)
    # Close the TE gap BEFORE normalizing. Some real UIUC files record
    # slightly different x for the upper vs. lower TE point (digitization
    # noise, e.g. 1.00003 vs 0.99997 in naca23012.dat) -- normalizing
    # first would lock the chord to whichever one happens to be larger,
    # then closing would average it away from exactly x=1.0. Closing
    # first establishes a single authoritative TE point, so the chord
    # (and x=1.0 at the TE) is defined by the actual closed geometry.
    coords = _close_trailing_edge(coords)
    coords = _normalize_chord(coords)

    if _polygon_has_self_intersections(coords):
        raise ValueError(
            f"Geometry is self-intersecting after reordering/closing, "
            f"cannot produce a valid airfoil contour: {dat_path}"
        )

    return coords


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 0: load and normalize a UIUC .dat airfoil.")
    parser.add_argument("--input", required=True, help="Path to the .dat file.")
    args = parser.parse_args()

    result = load_airfoil(args.input)
    print(f"Loaded {len(result)} points, x range [{result[:, 0].min():.4f}, {result[:, 0].max():.4f}]")
