"""Clean unused datasets from cache"""

import shutil
from pathlib import Path

# Datasets we actually use (based on dataset_loaders.py)
USED_DATASETS = {
    "Anthropic___hh-rlhf",
    "declare-lab___harmful_qa",
    "toxigen___toxigen-data",
    "ucberkeley-dlab___measuring-hate-speech",
    "tatsu-lab___alpaca",
    "Amod___mental_health_counseling_conversations",
    "arianaazarbal___self-harm-synthetic-eval",
    "jquiros___suicide",
    "lewtun___drug-reviews",
    "thu-coai___safety_bench",
}

CACHE_DIR = Path("datasets_cache")


def clean_unused_datasets():
    """Remove unused datasets from cache"""
    print("=" * 80)
    print("Cleaning Unused Datasets")
    print("=" * 80)
    
    if not CACHE_DIR.exists():
        print("Cache directory doesn't exist")
        return
    
    # Get all dataset directories
    all_dirs = [d for d in CACHE_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    
    removed = []
    kept = []
    
    for dataset_dir in all_dirs:
        dataset_name = dataset_dir.name
        if dataset_name not in USED_DATASETS:
            print(f"Removing: {dataset_name}")
            try:
                shutil.rmtree(dataset_dir)
                removed.append(dataset_name)
            except Exception as e:
                print(f"  Error removing {dataset_name}: {e}")
        else:
            kept.append(dataset_name)
    
    # Also remove lock files for unused datasets
    lock_files = list(CACHE_DIR.glob("_*lock"))
    for lock_file in lock_files:
        # Extract dataset name from lock file
        lock_name = lock_file.name
        is_unused = True
        for used in USED_DATASETS:
            if used.replace("___", "/").replace("_", "-") in lock_name or used in lock_name:
                is_unused = False
                break
        
        if is_unused:
            try:
                lock_file.unlink()
                removed.append(f"lock: {lock_file.name[:50]}")
            except Exception as e:
                print(f"  Error removing lock {lock_file.name}: {e}")
    
    print(f"\n✓ Kept {len(kept)} datasets")
    print(f"✓ Removed {len(removed)} unused datasets/dirs")
    print(f"\nKept datasets:")
    for name in sorted(kept):
        print(f"  - {name}")


if __name__ == "__main__":
    clean_unused_datasets()

