"""
UIUC Airfoil Coordinate Downloader & Loader

Downloads .dat airfoil coordinate files from the UIUC Airfoil Data Site
(https://m-selig.ae.illinois.edu/ads/coord_database.html), maintained by
Prof. Michael Selig at the University of Illinois.

Two format sources are available:
  - "database"  : the main coord_database.html page (mixed formats)
  - "selig"     : the Selig-format directory (standardised format)

Original scraping approach by JoshTheEngineer (with permission from
Dr. Michael Selig, 01/16/19). Refactored into a reusable module for
the multi-agent aero-structural optimisation system.

Prerequisites:
    pip install beautifulsoup4 lxml requests
"""

import os
import re
import requests
from bs4 import BeautifulSoup

# ─── URL Constants ──────────────────────────────────────────────────────────
# The UIUC database offers two directories of .dat coordinate files:
#   1) The main database page — links point into sub-paths like "coord/..."
#   2) The Selig-format directory — a flat listing of .dat files already in
#      a standardised (x going 1→0 upper, then 0→1 lower) format.
SOURCES = {
    "database": {
        "index_url": "https://m-selig.ae.illinois.edu/ads/coord_database.html",
        "base_url":  "https://m-selig.ae.illinois.edu/ads/",
    },
    "selig": {
        "index_url": "https://m-selig.ae.illinois.edu/ads/coord_seligFmt/",
        "base_url":  "https://m-selig.ae.illinois.edu/ads/coord_seligFmt/",
    },
}


def scrape_dat_links(source: str = "selig") -> list[str]:
    """
    Scrape the UIUC webpage for all .dat file download URLs.

    We prefer the 'selig' source because those files are already in a
    consistent format (Selig format: upper surface from TE→LE, then lower
    surface from LE→TE). The 'database' source has mixed formats which
    require extra parsing logic.

    Args:
        source: "database" or "selig" — which UIUC directory to scrape.

    Returns:
        List of full URLs to .dat files.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Choose 'database' or 'selig'.")

    index_url = SOURCES[source]["index_url"]
    base_url  = SOURCES[source]["base_url"]

    # Fetch the HTML index page that lists all the airfoil .dat file links.
    response = requests.get(index_url, timeout=30)
    response.raise_for_status()  # blow up early if the server is down

    # Parse the HTML with lxml (fast C-based parser) and find every <a> tag
    # whose href ends in ".dat" — that's how the original scripts identified
    # the coordinate files among all the other links on the page.
    soup = BeautifulSoup(response.text, "lxml")
    dat_links = soup.find_all("a", attrs={"href": re.compile(r"\.dat$", re.IGNORECASE)})

    # Build full URLs. Some hrefs are relative paths like "coord/naca0012.dat",
    # so we prepend the base URL to make them absolute.
    urls = []
    for tag in dat_links:
        href = tag.get("href")
        if href.startswith("http"):
            urls.append(href)                 # already absolute
        else:
            urls.append(base_url + href)      # relative → absolute
    return urls


def download_airfoils(
    source: str = "selig",
    save_dir: str = "data/raw/airfoils",
) -> list[str]:
    """
    Download every .dat airfoil coordinate file from the UIUC database.

    The original scripts used urllib.request.urlretrieve in a tight loop.
    We swap to requests.get because it gives us better error handling,
    timeout control, and doesn't silently swallow HTTP errors.

    Args:
        source:   "database" or "selig" — which UIUC index to scrape.
        save_dir: Local folder to save the .dat files into.

    Returns:
        List of local file paths that were saved.
    """
    # Create the output directory if it doesn't exist yet.
    os.makedirs(save_dir, exist_ok=True)

    # Step 1 — get the list of .dat URLs from the HTML page.
    urls = scrape_dat_links(source=source)
    print(f"Found {len(urls)} airfoil .dat files from '{source}' source.")

    saved_files = []
    for i, url in enumerate(urls, start=1):
        # Extract the filename from the URL (e.g. "naca0012.dat").
        filename = url.rsplit("/", 1)[-1]
        filepath = os.path.join(save_dir, filename)

        try:
            # Stream the download to avoid loading huge files into memory
            # (these .dat files are tiny, but good practice regardless).
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(resp.content)

            saved_files.append(filepath)
            print(f"  [{i}/{len(urls)}] Saved {filename}")

        except requests.RequestException as e:
            # Don't crash the whole batch if one file is unavailable —
            # just warn and keep going with the rest.
            print(f"  [{i}/{len(urls)}] FAILED {filename}: {e}")

    print(f"Download complete. {len(saved_files)}/{len(urls)} files saved to '{save_dir}'.")
    return saved_files


def load_dat_file(filepath: str) -> tuple[str, list[tuple[float, float]]]:
    """
    Parse a single UIUC .dat airfoil coordinate file and return (name, coords).

    The Selig format looks like this:
        NACA 0012
        1.000000  0.001260
        0.999416  0.001476
        ...
    Line 1 is the airfoil name. Every subsequent line that contains exactly
    two floats is an (x, y) coordinate point. We skip any blank lines or
    lines with non-numeric content (some files have extra header rows with
    the number of points).

    Coordinates are normalised to a chord of 1.0 in the .dat files. To get
    Meters for the CFD tool, multiply by your desired chord length.

    Args:
        filepath: Path to the .dat file on disk.

    Returns:
        (name, coords): airfoil name string and list of (x, y) tuples.
    """
    coords = []
    name = ""

    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty .dat file: {filepath}")

    # First non-blank line is always the airfoil name / identifier.
    name = lines[0].strip()

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                # Each coordinate pair is (x/c, y/c) — normalised by chord.
                x, y = float(parts[0]), float(parts[1])
                coords.append((x, y))
            except ValueError:
                # Skip lines that look like two columns but aren't numbers
                # (e.g. some files have a "33  33" point-count header).
                continue

    return name, coords


# ─── CLI entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and/or load UIUC airfoil .dat files."
    )
    parser.add_argument(
        "--source",
        choices=["database", "selig"],
        default="selig",
        # We default to "selig" because those files are already in a
        # standardised format that doesn't need extra parsing heuristics.
        help="Which UIUC directory to scrape (default: selig).",
    )
    parser.add_argument(
        "--save-dir",
        default="data/raw/airfoils",
        help="Folder to save .dat files into (default: data/raw/airfoils).",
    )
    parser.add_argument(
        "--test-load",
        action="store_true",
        # Quick sanity check: after downloading, load the first .dat file
        # and print its name + first 5 coordinate pairs to verify parsing.
        help="After downloading, test-load the first file and print its coords.",
    )
    args = parser.parse_args()

    # Download all .dat files from the chosen source.
    saved = download_airfoils(source=args.source, save_dir=args.save_dir)

    # Optionally verify the parser works on a real file.
    if args.test_load and saved:
        name, coords = load_dat_file(saved[0])
        print(f"\nTest load of '{saved[0]}':")
        print(f"  Airfoil name : {name}")
        print(f"  Total points : {len(coords)}")
        print(f"  First 5 pts  : {coords[:5]}")
