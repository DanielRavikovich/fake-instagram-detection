"""
download_data.py
----------------
Downloads the InstaFake dataset from GitHub (fcakyon/instafake-dataset).
The Kaggle dataset must be downloaded manually (see instructions below).

Usage:
    python download_data.py
"""

import os
import json
import requests

# ── InstaFake (GitHub) ──────────────────────────────────────────────────────

INSTAFAKE_BASE = (
    "https://raw.githubusercontent.com/fcakyon/instafake-dataset/"
    "master/data/fake-v1.0/"
)
FILES = {
    "fakeAccountData.json": INSTAFAKE_BASE + "fakeAccountData.json",
    "realAccountData.json": INSTAFAKE_BASE + "realAccountData.json",
}


def download_instafake(dest_dir: str = ".") -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for filename, url in FILES.items():
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            print(f"  [skip] {filename} already exists.")
            continue
        print(f"  Downloading {filename} ...", end=" ", flush=True)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(response.json(), f)
        print("done.")


# ── Kaggle (manual) ─────────────────────────────────────────────────────────

KAGGLE_INSTRUCTIONS = """
Kaggle dataset: "Instagram Fake Spammer Genuine Accounts"
URL: https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts

To download:
  Option A - Kaggle CLI (recommended):
    pip install kaggle
    kaggle datasets download -d free4ever1/instagram-fake-spammer-genuine-accounts
    unzip instagram-fake-spammer-genuine-accounts.zip

  Option B - Browser:
    1. Log in to https://www.kaggle.com
    2. Open the dataset URL above
    3. Click "Download" and extract the ZIP
    4. Place train.csv and test.csv in the project root (same folder as this script)

Expected files after setup:
    fakeAccountData.json   <- downloaded automatically by this script
    realAccountData.json   <- downloaded automatically by this script
    train.csv              <- from Kaggle (manual)
    test.csv               <- from Kaggle (manual)
"""


if __name__ == "__main__":
    print("=== Downloading InstaFake dataset from GitHub ===")
    download_instafake(dest_dir=".")
    print("\n=== Kaggle dataset (manual step required) ===")
    print(KAGGLE_INSTRUCTIONS)
    print("Done. Place all four data files in the project root before running the notebook.")
