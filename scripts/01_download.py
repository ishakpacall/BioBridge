"""Download raw public data with hash enforcement.

  python scripts/01_download.py
  python scripts/01_download.py --import-legacy /path/to/old/full_data_sources.json   # exact pilot files
  python scripts/01_download.py --only metabric_patient metabric_sample metabric_expression
"""
import argparse
import _bootstrap  # noqa: F401
from biobridge.download import download_all, import_legacy

ap = argparse.ArgumentParser()
ap.add_argument('--import-legacy', default=None); ap.add_argument('--only', nargs='*')
a = ap.parse_args()
download_all(a.only, import_legacy(a.import_legacy) if a.import_legacy else None)
print('COMPLETE: raw data verified/locked in data/raw/DATA_LOCK.json')
