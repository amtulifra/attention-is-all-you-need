import os
import tarfile
import urllib.request
from pathlib import Path

# Create data directory
DATA_DIR = Path("data/multi30k")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download URLs
TRAIN_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/train.lc.norm.tok.de"
VAL_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/val.lc.norm.tok.de"
TEST_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/test_2016_flickr.lc.norm.tok.de"

TRAIN_EN_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/train.lc.norm.tok.en"
VAL_EN_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/val.lc.norm.tok.en"
TEST_EN_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/tok/test_2016_flickr.lc.norm.tok.en"

def download_file(url, filename):
    """Download a file from URL to filename."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists, skipping download")

def main():
    # Download German files
    print("Downloading German files...")
    download_file(TRAIN_URL, DATA_DIR / "train.de")
    download_file(VAL_URL, DATA_DIR / "val.de")
    download_file(TEST_URL, DATA_DIR / "test.de")
    
    # Download English files
    print("\nDownloading English files...")
    download_file(TRAIN_EN_URL, DATA_DIR / "train.en")
    download_file(VAL_EN_URL, DATA_DIR / "val.en")
    download_file(TEST_EN_URL, DATA_DIR / "test.en")
    
    print("\nDataset downloaded successfully!")
    print(f"Files are saved in: {os.path.abspath(DATA_DIR)}")

if __name__ == "__main__":
    main()
