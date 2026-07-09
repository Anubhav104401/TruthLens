import argparse
import json
import os
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DATASETS = [
    {
        "id": "fake_or_real_news",
        "name": "fake_or_real_news.csv",
        "url": "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
        "source": "https://github.com/lutzhamel/fake-news/blob/master/data/fake_or_real_news.csv",
        "description": "Labelled fake/real news CSV derived from the commonly used fake-or-real-news dataset.",
        "type": "direct"
    },
    {
        "id": "isot",
        "name": "ISOT Fake and Real News",
        "kaggle_path": "clmentbisaillon/fake-and-real-news-dataset",
        "source": "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
        "description": "ISOT Fake and Real News Dataset.",
        "type": "kaggle"
    },
    {
        "id": "liar",
        "name": "LIAR Dataset",
        "url": "https://cs.ucsb.edu/~william/data/liar_dataset.zip",
        "source": "https://cs.ucsb.edu/~william/data/liar_dataset.zip",
        "description": "LIAR short political statements dataset.",
        "type": "zip"
    }
]

def load_source_metadata():
    source_file = DATA_DIR / "dataset_source.json"
    if source_file.exists():
        try:
            data = json.loads(source_file.read_text())
            return data if isinstance(data, list) else [data]
        except Exception:
            return []
    return []

def save_source_metadata(metadata_list):
    source_file = DATA_DIR / "dataset_source.json"
    source_file.write_text(json.dumps(metadata_list, indent=2))

def download_dataset(dataset, force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    record = {
        "id": dataset["id"],
        "name": dataset.get("name", ""),
        "source": dataset.get("source", ""),
        "description": dataset.get("description", ""),
        "download_timestamp": timestamp
    }
    
    if dataset["type"] == "direct":
        target = DATA_DIR / dataset["name"]
        if target.exists() and not force:
            print(f"Dataset already exists: {target}")
            return record
            
        print(f"Downloading {dataset['name']}...")
        urlretrieve(dataset["url"], target)
        print(f"Saved dataset to {target}")
        
    elif dataset["type"] == "kaggle":
        try:
            import kagglehub
            target_fake = DATA_DIR / "Fake.csv"
            target_true = DATA_DIR / "True.csv"
            
            if target_fake.exists() and target_true.exists() and not force:
                print(f"Dataset already exists: ISOT (Fake.csv and True.csv)")
                return record
                
            print(f"Downloading {dataset['name']} via kagglehub...")
            path = kagglehub.dataset_download(dataset["kaggle_path"])
            
            # Copy Fake.csv and True.csv to data dir
            shutil.copy(Path(path) / "Fake.csv", target_fake)
            shutil.copy(Path(path) / "True.csv", target_true)
            print(f"Saved ISOT to {DATA_DIR}")
        except Exception as e:
            print(f"Failed to download {dataset['name']}. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set. Error: {e}")
            return None

    elif dataset["type"] == "zip":
        target_dir = DATA_DIR / dataset["id"]
        if target_dir.exists() and not force:
            print(f"Dataset already exists: {target_dir}")
            return record
            
        print(f"Downloading {dataset['name']}...")
        zip_path = DATA_DIR / f"{dataset['id']}.zip"
        urlretrieve(dataset["url"], zip_path)
        
        print(f"Extracting {dataset['name']}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Clean up zip
        zip_path.unlink()
        print(f"Saved {dataset['name']} to {target_dir}")

    return record

def main():
    parser = argparse.ArgumentParser(description="Download fake/real news datasets.")
    parser.add_argument("--force", action="store_true", help="Replace existing downloaded CSVs.")
    args = parser.parse_args()
    
    metadata_list = load_source_metadata()
    existing_ids = {m.get("id") for m in metadata_list}
    
    new_metadata = []
    
    for dataset in DATASETS:
        record = download_dataset(dataset, force=args.force)
        if record:
            # Update or append
            if record["id"] in existing_ids:
                # keep old record if not forced, but we just override for simplicity
                new_metadata.append(record)
            else:
                new_metadata.append(record)
        else:
            # Retain old record if download failed but it existed
            old = next((m for m in metadata_list if m.get("id") == dataset["id"]), None)
            if old:
                new_metadata.append(old)
                
    save_source_metadata(new_metadata)
    print("Download process complete.")

if __name__ == "__main__":
    main()
