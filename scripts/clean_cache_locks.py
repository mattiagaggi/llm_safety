"""Clean up empty lock files from datasets_cache root"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loaders import DATA_CACHE_DIR


def clean_lock_files():
    """Remove empty lock files from cache root"""
    cache_dir = Path(DATA_CACHE_DIR)
    
    if not cache_dir.exists():
        print(f"Cache directory doesn't exist: {cache_dir}")
        return
    
    print(f"Cleaning lock files in: {cache_dir}")
    print("=" * 80)
    
    # Find all .lock files in root
    lock_files = list(cache_dir.glob("*.lock"))
    
    if not lock_files:
        print("No lock files found in root")
        return
    
    print(f"Found {len(lock_files)} lock files:")
    
    total_size = 0
    removed = 0
    
    for lock_file in lock_files:
        size = lock_file.stat().st_size
        total_size += size
        
        # Only remove empty lock files (they should all be empty)
        if size == 0:
            try:
                lock_file.unlink()
                removed += 1
                print(f"  ✓ Removed: {lock_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to remove {lock_file.name}: {e}")
        else:
            print(f"  ⚠️  Skipping non-empty file: {lock_file.name} ({size} bytes)")
    
    print("=" * 80)
    print(f"Removed {removed} lock files")
    print(f"Total size: {total_size} bytes")
    
    # Note: Lock files in subdirectories (.locks/ and dataset dirs) are kept
    # as they're part of the normal cache structure


if __name__ == "__main__":
    clean_lock_files()
