import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DATASET = {
    "name": "fake_or_real_news.csv",
    "url": "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
    "source_page": "https://github.com/lutzhamel/fake-news/blob/master/data/fake_or_real_news.csv",
    "description": "Labelled fake/real news CSV derived from the commonly used fake-or-real-news dataset.",
}


def download(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / DATASET["name"]
    source_file = DATA_DIR / "dataset_source.json"

    if target.exists() and not force:
        print(f"Dataset already exists: {target}")
    else:
        print(f"Downloading {DATASET['name']}...")
        urlretrieve(DATASET["url"], target)
        print(f"Saved dataset to {target}")

    source_file.write_text(json.dumps(DATASET, indent=2))
    print(f"Wrote source metadata to {source_file}")
    return target


def main():
    parser = argparse.ArgumentParser(description="Download a labelled fake/real news dataset.")
    parser.add_argument("--force", action="store_true", help="Replace the existing downloaded CSV.")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
