"""
Generates deterministic .dat fixtures for Stage 0 geometry loader tests.

Analytic NACA 4-digit coordinates (not downloaded) so tests run offline and
never flake on network access. Covers the shape diversity called for in the
build spec: thin symmetric, thick symmetric, moderate camber, high camber.
Also writes two synthetic edge-case files: one with a deliberately scrambled
point order, one with an open (unclosed) trailing edge.

Run once to (re)generate fixtures: python tests/fixtures/generate_fixtures.py
"""

import os
import numpy as np

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# (name, max_camber/100, camber_position/10, max_thickness/100)
NACA_DEFS = [
    ("naca0006", 0.00, 0.0, 0.06),  # very thin symmetric
    ("naca0012", 0.00, 0.0, 0.12),  # thin symmetric (baseline)
    ("naca0021", 0.00, 0.0, 0.21),  # thick symmetric
    ("naca2412", 0.02, 0.4, 0.12),  # moderate camber
    ("naca4412", 0.04, 0.4, 0.12),  # higher camber
    ("naca6412", 0.06, 0.4, 0.12),  # high camber
]

N_PER_SURFACE = 60


def naca4_coords(m, p, t, n=N_PER_SURFACE):
    """Return Selig-ordered (TE->upper->LE->lower->TE) coords for a NACA 4-digit airfoil."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))  # cosine spacing, LE and TE dense

    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p ** 2 * (2 * p * x - x ** 2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2),
        )

    # Standard tabulated NACA 4-digit form: thickness applied vertically
    # (not perpendicular to the camber line). This is what UIUC's published
    # .dat coordinates use, and it keeps both surfaces at the same x for a
    # given station -- including x=1.0 exactly at the TE on both surfaces.
    xu = x
    yu = yc + yt
    xl = x
    yl = yc - yt

    upper = np.column_stack([xu, yu])[::-1]  # LE->TE flipped to TE->LE
    lower = np.column_stack([xl, yl])[1:]    # LE->TE, drop duplicate LE point

    return np.vstack([upper, lower])


def write_dat(path, name, coords):
    with open(path, "w") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"  {x:.6f}  {y:.6f}\n")


def main():
    for name, m, p, t in NACA_DEFS:
        coords = naca4_coords(m, p, t)
        write_dat(os.path.join(FIXTURE_DIR, f"{name}.dat"), name, coords)

    # --- Edge case: scrambled ordering ---
    # Take naca4412 and rebuild it lower-surface-first (reverse of Selig),
    # i.e. TE(lower) -> LE -> TE(upper). A conformant loader must still
    # recover canonical Selig ordering from this.
    base = naca4_coords(0.04, 0.4, 0.12)
    le_idx = int(np.argmin(base[:, 0]))
    upper_arc = base[: le_idx + 1]      # TE(upper) -> LE
    lower_arc = base[le_idx:]           # LE -> TE(lower)
    scrambled = np.vstack([lower_arc[::-1], upper_arc[::-1][1:]])
    write_dat(os.path.join(FIXTURE_DIR, "scrambled_order.dat"), "scrambled naca4412", scrambled)

    # --- Edge case: unclosed trailing edge (blunt TE gap) ---
    base012 = naca4_coords(0.0, 0.0, 0.12)
    open_te = base012.copy()
    open_te[0, 1] += 0.01   # pull upper TE point up
    open_te[-1, 1] -= 0.01  # pull lower TE point down -> visible gap
    write_dat(os.path.join(FIXTURE_DIR, "open_te.dat"), "open TE naca0012", open_te)

    # --- Adversarial: consecutive duplicate points (a digitization
    # artifact -- a scanned/manually-entered UIUC file can repeat a
    # coordinate) -- crashes Stage 1's CubicSpline resampling with a raw
    # scipy ValueError ("`x` must be strictly increasing") if Stage 0
    # doesn't dedupe first. Found via adversarial probing 2026-09-14/15.
    duplicate_points = naca4_coords(0.0, 0.0, 0.12).copy()
    duplicate_points[10] = duplicate_points[9]
    duplicate_points[11] = duplicate_points[9]
    duplicate_points[30] = duplicate_points[29]
    write_dat(
        os.path.join(FIXTURE_DIR, "duplicate_points.dat"),
        "duplicate points naca0012", duplicate_points,
    )

    # --- Adversarial: genuinely self-intersecting contour (not just
    # scrambled ordering -- a handful of interior points flipped/amplified
    # across the chord line so the polygon actually crosses itself).
    # Correct behavior is a fast, clear Stage 0 rejection, not burning a
    # Stage 1 gmsh attempt on unmeshable garbage. Found via adversarial
    # probing 2026-09-14/15.
    rng = np.random.default_rng(42)
    self_intersecting = naca4_coords(0.02, 0.4, 0.12).copy()
    idx = rng.choice(len(self_intersecting) - 2, size=6, replace=False) + 1
    self_intersecting[idx, 1] *= -3.0
    write_dat(
        os.path.join(FIXTURE_DIR, "self_intersecting.dat"),
        "self intersecting naca2412", self_intersecting,
    )

    print(f"Wrote fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
