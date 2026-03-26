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
