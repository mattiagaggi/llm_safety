"""Download datasets and verify loaders can access them"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from datasets import load_dataset
from src.data import (
    load_beavertails,
    load_harmfulqa,
    load_combined_dataset,
    load_red_team_attempts,
    load_toxigen,
    load_measuring_hate_speech,
    load_alpaca,
    load_mental_health_counseling,
    load_self_harm_synthetic,
    load_suicide_dataset,
    load_drug_reviews,
    load_safetybench,
    SafetyCategory,
    SafetyDataset
)

# Import cache directory from dataset_loaders
from src.data.dataset_loaders import DATA_CACHE_DIR


def download_and_verify_beavertails():
    """Download and verify BeaverTails dataset"""
    print("\n" + "=" * 80)
    print("1. BeaverTails Dataset (Anthropic/hh-rlhf)")
    print("=" * 80)
    
    try:
        print("Step 1: Downloading from HuggingFace...")
        start_time = time.time()
        
        # Trigger download by loading the dataset
        hf_dataset = load_dataset(
            "Anthropic/hh-rlhf",
            split="train[:100]",
            cache_dir=str(DATA_CACHE_DIR)
        )
        download_time = time.time() - start_time
        
        print(f"✓ Download successful! (took {download_time:.2f}s for sample)")
        print(f"  Sample size: {len(hf_dataset)} examples")
        print(f"  Columns: {hf_dataset.column_names}")
        
        print("\nStep 2: Testing custom loader...")
        start_time = time.time()
        
        # Test our custom loader
        dataset = load_beavertails(split="train", max_examples=1000, include_safe=True)
        load_time = time.time() - start_time
        
        print(f"✓ Loader works! (took {load_time:.2f}s)")
        print(f"  Total examples loaded: {len(dataset)}")
        print(f"  Unsafe examples: {dataset.get_labels().sum()}")
        print(f"  Safe examples: {(~dataset.get_labels().astype(bool)).sum()}")
        
        # Show category distribution
        categories = {}
        for example in dataset.examples:
            cat = example.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"  Category distribution:")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")
        
        # Show sample examples
        print(f"\n  Sample examples:")
        unsafe_examples = [e for e in dataset.examples if e.is_unsafe][:2]
        safe_examples = [e for e in dataset.examples if not e.is_unsafe][:2]
        
        for i, example in enumerate(unsafe_examples, 1):
            print(f"    Unsafe {i}: {example.text[:80]}...")
        for i, example in enumerate(safe_examples, 1):
            print(f"    Safe {i}: {example.text[:80]}...")
        
        return True, dataset
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def download_and_verify_harmfulqa():
    """Download and verify HarmfulQA dataset"""
    print("\n" + "=" * 80)
    print("2. HarmfulQA Dataset (declare-lab/HarmfulQA)")
    print("=" * 80)
    
    try:
        print("Step 1: Downloading from HuggingFace...")
        start_time = time.time()
        
        # Trigger download
        hf_dataset = load_dataset(
            "declare-lab/HarmfulQA",
            split="train[:100]",
            cache_dir=str(DATA_CACHE_DIR)
        )
        download_time = time.time() - start_time
        
        print(f"✓ Download successful! (took {download_time:.2f}s for sample)")
        print(f"  Cache location: {DATA_CACHE_DIR}")
        print(f"  Sample size: {len(hf_dataset)} examples")
        print(f"  Columns: {hf_dataset.column_names}")
        
        print("\nStep 2: Testing custom loader...")
        start_time = time.time()
        
        # Test our custom loader
        dataset = load_harmfulqa(split="train", max_examples=500)
        load_time = time.time() - start_time
        
        print(f"✓ Loader works! (took {load_time:.2f}s)")
        print(f"  Total examples loaded: {len(dataset)}")
        print(f"  Unsafe examples: {dataset.get_labels().sum()}")
        print(f"  Safe examples: {(~dataset.get_labels().astype(bool)).sum()}")
        
        # Show category distribution
        categories = {}
        for example in dataset.examples:
            cat = example.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"  Category distribution:")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")
        
        # Show sample examples
        print(f"\n  Sample examples:")
        for i, example in enumerate(dataset.examples[:3], 1):
            print(f"    Example {i}: {example.text[:80]}...")
            print(f"      Category: {example.category.value}, Unsafe: {example.is_unsafe}")
        
        return True, dataset
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_combined_loader():
    """Test the combined dataset loader"""
    print("\n" + "=" * 80)
    print("3. Combined Dataset Loader")
    print("=" * 80)
    
    try:
        print("Loading combined dataset from multiple sources...")
        start_time = time.time()
        
        dataset = load_combined_dataset(
            beavertails_max=1000,
            harmfulqa_max=500,
            include_safe=True
        )
        load_time = time.time() - start_time
        
        print(f"✓ Combined loader works! (took {load_time:.2f}s)")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        print(f"  Safe: {(~dataset.get_labels().astype(bool)).sum()}")
        
        # Show category distribution
        categories = {}
        for example in dataset.examples:
            cat = example.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"  Category distribution:")
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")
        
        # Test train/test split
        print(f"\n  Testing train/test split...")
        train, test = dataset.split(train_ratio=0.8)
        print(f"    Train: {len(train)}, Test: {len(test)}")
        
        return True, dataset
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_category_filtering():
    """Test filtering by category"""
    print("\n" + "=" * 80)
    print("4. Category Filtering Test")
    print("=" * 80)
    
    try:
        # Load a small dataset
        dataset = load_beavertails(split="train", max_examples=500, include_safe=True)
        
        print("Testing category filtering...")
        
        for category in [SafetyCategory.SELF_HARM, SafetyCategory.VIOLENCE, SafetyCategory.SAFE]:
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
    """Main function"""
    print("=" * 80)
    print("Dataset Download and Verification Script")
    print("=" * 80)
    print(f"\nData will be cached in: {DATA_CACHE_DIR}")
    print("(This directory is gitignored)")
    print("\nThis script will:")
    print("1. Download datasets from HuggingFace (cached for future use)")
    print("2. Verify custom loaders work correctly")
    print("3. Show dataset statistics")
    print("4. Test data access and filtering")
    print("\nNote: First-time downloads may take a few minutes")
    print("=" * 80)
    
    results = {}
    
    # Test BeaverTails
    success, dataset = download_and_verify_beavertails()
    results["beavertails"] = {"success": success, "dataset": dataset}
    
    # Test HarmfulQA
    success, dataset = download_and_verify_harmfulqa()
    results["harmfulqa"] = {"success": success, "dataset": dataset}
    
    # Test combined loader
    success, dataset = test_combined_loader()
    results["combined"] = {"success": success, "dataset": dataset}
    
    # Test red team attempts
    print("\n\n5. Red Team Attempts Dataset")
    print("-" * 80)
    try:
        dataset = load_red_team_attempts(split="train", max_examples=100)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        results["red_team"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["red_team"] = {"success": False}
    
    # Test ToxiGen
    print("\n\n6. ToxiGen Dataset")
    print("-" * 80)
    try:
        dataset = load_toxigen(split="train", max_examples=100, min_toxicity=0.5)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        print(f"  Safe: {(~dataset.get_labels().astype(bool)).sum()}")
        results["toxigen"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["toxigen"] = {"success": False}
    
    # Test Measuring Hate Speech
    print("\n\n7. Measuring Hate Speech Dataset")
    print("-" * 80)
    try:
        dataset = load_measuring_hate_speech(split="train", max_examples=100, min_hate_score=0.0)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        print(f"  Safe: {(~dataset.get_labels().astype(bool)).sum()}")
        results["measuring_hate"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["measuring_hate"] = {"success": False}
    
    # Test Alpaca
    print("\n\n8. Alpaca Dataset")
    print("-" * 80)
    try:
        dataset = load_alpaca(split="train", max_examples=100)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        print(f"  Safe: {(~dataset.get_labels().astype(bool)).sum()}")
        results["alpaca"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["alpaca"] = {"success": False}
    
    # Test Mental Health Counseling
    print("\n\n9. Mental Health Counseling Dataset")
    print("-" * 80)
    try:
        dataset = load_mental_health_counseling(split="train", max_examples=50, filter_self_harm=True)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        results["mental_health"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["mental_health"] = {"success": False}
    
    # Test Self-Harm Synthetic
    print("\n\n10. Self-Harm Synthetic Dataset")
    print("-" * 80)
    try:
        dataset = load_self_harm_synthetic(split="train", max_examples=50)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        results["self_harm_synthetic"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["self_harm_synthetic"] = {"success": False}
    
    # Test Suicide Dataset
    print("\n\n11. Suicide Dataset")
    print("-" * 80)
    try:
        dataset = load_suicide_dataset(split="train", max_examples=50)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        results["suicide"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["suicide"] = {"success": False}
    
    # Test Drug Reviews
    print("\n\n12. Drug Reviews Dataset")
    print("-" * 80)
    try:
        dataset = load_drug_reviews(split="train", max_examples=50, filter_illegal=True)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        results["drug_reviews"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["drug_reviews"] = {"success": False}
    
    # Test SafetyBench
    print("\n\n13. SafetyBench Dataset")
    print("-" * 80)
    try:
        dataset = load_safetybench(config="test", split="en", max_examples=50)
        print(f"✓ Loaded {len(dataset)} examples")
        print(f"  Unsafe: {dataset.get_labels().sum()}")
        from collections import Counter
        cats = Counter([e.category.value for e in dataset.examples])
        print(f"  Categories: {dict(cats)}")
        results["safetybench"] = {"success": True, "dataset": dataset}
    except Exception as e:
        print(f"✗ Error: {e}")
        results["safetybench"] = {"success": False}
    
    # Test category filtering
    success = test_category_filtering()
    results["filtering"] = {"success": success}
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_success = all(r["success"] for r in results.values())
    
    for name, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"{status} {name}: {'PASS' if result['success'] else 'FAIL'}")
    
    if all_success:
        print("\n✓ All tests passed! Your loaders are ready to use.")
        print("\nExample usage:")
        print("  from src.data import load_beavertails")
        print("  dataset = load_beavertails(split='train', max_examples=10000)")
    else:
        print("\n⚠ Some tests failed. Check errors above.")
    
    print("=" * 80)
    
    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

