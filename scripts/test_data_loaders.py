"""Test that data loaders can access cached datasets"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_beavertails,
    load_harmfulqa,
    load_combined_dataset,
    SafetyCategory,
    SafetyDataset
)
from src.data.dataset_loaders import DATA_CACHE_DIR


def test_beavertails_loader():
    """Test BeaverTails loader"""
    print("=" * 80)
    print("Testing BeaverTails Loader")
    print("=" * 80)
    print(f"Cache directory: {DATA_CACHE_DIR}")
    
    try:
        # Load a small sample
        print("\n1. Loading small sample (100 examples)...")
        dataset = load_beavertails(split="train", max_examples=100, include_safe=True)
        
        print(f"✓ Successfully loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        print(f"  Safe: {(~dataset.get_labels().astype(bool)).sum()}")
        
        # Test accessing data
        print("\n2. Testing data access...")
        texts = dataset.get_texts()
        labels = dataset.get_labels()
        
        print(f"✓ Got {len(texts)} texts")
        print(f"✓ Got {len(labels)} labels")
        print(f"  Sample text: {texts[0][:80]}...")
        print(f"  Sample label: {labels[0]} ({'unsafe' if labels[0] else 'safe'})")
        
        # Test train/test split
        print("\n3. Testing train/test split...")
        train, test = dataset.split(train_ratio=0.8)
        print(f"✓ Train: {len(train)}, Test: {len(test)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_harmfulqa_loader():
    """Test HarmfulQA loader"""
    print("\n" + "=" * 80)
    print("Testing HarmfulQA Loader")
    print("=" * 80)
    print(f"Cache directory: {DATA_CACHE_DIR}")
    
    try:
        # Load a small sample
        print("\n1. Loading small sample (50 examples)...")
        dataset = load_harmfulqa(split="train", max_examples=50)
        
        print(f"✓ Successfully loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        print(f"  Safe: {(~dataset.get_labels().astype(bool)).sum()}")
        
        # Test accessing data
        print("\n2. Testing data access...")
        texts = dataset.get_texts()
        labels = dataset.get_labels()
        categories = dataset.get_category_labels()
        
        print(f"✓ Got {len(texts)} texts")
        print(f"✓ Got {len(labels)} labels")
        print(f"✓ Got {len(categories)} category labels")
        
        # Show category distribution
        from collections import Counter
        cat_counts = Counter([cat.value for cat in categories])
        print(f"  Category distribution: {dict(cat_counts)}")
        
        print(f"  Sample text: {texts[0][:80]}...")
        print(f"  Sample category: {categories[0].value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_loader():
    """Test combined dataset loader"""
    print("\n" + "=" * 80)
    print("Testing Combined Dataset Loader")
    print("=" * 80)
    
    try:
        print("\n1. Loading combined dataset...")
        dataset = load_combined_dataset(
            beavertails_max=200,
            harmfulqa_max=50,
            include_safe=True
        )
        
        print(f"✓ Successfully loaded {len(dataset)} total examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        print(f"  Safe: {(~dataset.get_labels().astype(bool)).sum()}")
        
        # Test all access methods
        print("\n2. Testing all data access methods...")
        texts = dataset.get_texts()
        labels = dataset.get_labels()
        categories = dataset.get_category_labels()
        
        print(f"✓ get_texts(): {len(texts)} items")
        print(f"✓ get_labels(): {len(labels)} items")
        print(f"✓ get_category_labels(): {len(categories)} items")
        
        # Test indexing
        print("\n3. Testing dataset indexing...")
        example = dataset[0]
        print(f"✓ dataset[0]: {type(example)}")
        print(f"  Text: {example.text[:60]}...")
        print(f"  Category: {example.category.value}")
        print(f"  Unsafe: {example.is_unsafe}")
        
        # Test iteration
        print("\n4. Testing dataset iteration...")
        count = 0
        for example in dataset:
            count += 1
            if count >= 3:
                break
        print(f"✓ Can iterate over dataset (checked {count} examples)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_category_filtering():
    """Test category filtering"""
    print("\n" + "=" * 80)
    print("Testing Category Filtering")
    print("=" * 80)
    
    try:
        # Load dataset
        dataset = load_beavertails(split="train", max_examples=200, include_safe=True)
        
        print("\n1. Testing category filtering...")
        for category in [SafetyCategory.SAFE, SafetyCategory.VIOLENCE]:
            filtered = SafetyDataset(
                [e for e in dataset.examples if e.category == category]
            )
            print(f"  {category.value}: {len(filtered)} examples")
        
        print("✓ Category filtering works!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("Data Loader Access Test")
    print("=" * 80)
    print("\nThis script verifies that your data loaders can:")
    print("1. Access cached datasets")
    print("2. Load data correctly")
    print("3. Provide all expected methods")
    print("=" * 80)
    
    results = {}
    
    # Test each loader
    results["beavertails"] = test_beavertails_loader()
    results["harmfulqa"] = test_harmfulqa_loader()
    results["combined"] = test_combined_loader()
    results["filtering"] = test_category_filtering()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}: {'PASS' if passed else 'FAIL'}")
    
    if all_passed:
        print("\n✓ All tests passed! Your data loaders are working correctly.")
        print("\nYou can now use them in your experiments:")
        print("  from src.data import load_beavertails")
        print("  dataset = load_beavertails(split='train', max_examples=10000)")
    else:
        print("\n⚠ Some tests failed. Check errors above.")
    
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

