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

