"""Test balancing utilities with new datasets (AIR-Bench 2024, TrustLLM)"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_air_bench_2024,
    load_trustllm,
    load_beavertails,
    load_alpaca,
    SafetyCategory
)
from src.data.balance_utils import balance_dataset, combine_and_balance
from src.data.category_balance import (
    balance_by_category,
    get_category_distribution,
    create_balanced_category_dataset
)


def show_balance_info(dataset, name):
    """Show balance information for a dataset"""
    print(f"\n{name}:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Unsafe: {dataset.get_labels().sum()}")
    print(f"  Safe: {(~dataset.get_labels().astype(bool)).sum()}")
    if len(dataset) > 0:
        dist = get_category_distribution(dataset)
        print(f"  Category distribution: {dist}")


def test_balance_with_air_bench():
    """Test balancing with AIR-Bench 2024"""
    print("=" * 80)
    print("Test 1: Balancing AIR-Bench 2024 with Safe Examples")
    print("=" * 80)
    
    # Load unsafe examples from AIR-Bench
    air_bench = load_air_bench_2024(split="test", max_examples=500)
    show_balance_info(air_bench, "AIR-Bench 2024 (original)")
    
    # Load safe examples from Alpaca
    alpaca = load_alpaca(split="train", max_examples=500)
    show_balance_info(alpaca, "Alpaca (safe examples)")
    
    # Combine and balance to 50/50
    print("\n--- Combining and balancing to 50/50 ratio ---")
    balanced = combine_and_balance(
        [air_bench, alpaca],
        target_ratio=0.5,  # 50% unsafe, 50% safe
        max_examples_per_dataset=500
    )
    show_balance_info(balanced, "Balanced dataset (50/50)")
    
    return balanced


def test_category_balance_with_new_datasets():
    """Test category balancing including new datasets"""
    print("\n" + "=" * 80)
    print("Test 2: Category Balancing with New Datasets")
    print("=" * 80)
    
    # Load datasets
    datasets = []
    
    # AIR-Bench 2024 (unsafe: violence, cybercrime, drugs)
    air_bench = load_air_bench_2024(split="test", max_examples=200)
    datasets.append(air_bench)
    show_balance_info(air_bench, "AIR-Bench 2024")
    
    # BeaverTails (mixed safe/unsafe)
    beavertails = load_beavertails(split="train", max_examples=200, include_safe=True)
    datasets.append(beavertails)
    show_balance_info(beavertails, "BeaverTails")
    
    # Alpaca (all safe)
    alpaca = load_alpaca(split="train", max_examples=200)
    datasets.append(alpaca)
    show_balance_info(alpaca, "Alpaca")
    
    # Create balanced dataset with equal representation per category
    print("\n--- Creating balanced category dataset (200 per category) ---")
    balanced = create_balanced_category_dataset(
        datasets=datasets,
        target_per_category=200,
        random_seed=42
    )
    show_balance_info(balanced, "Balanced by category")
    
    return balanced


def test_custom_category_balance():
    """Test custom category balancing"""
    print("\n" + "=" * 80)
    print("Test 3: Custom Category Balance")
    print("=" * 80)
    
    # Load AIR-Bench (has violence, cybercrime, drugs)
    air_bench = load_air_bench_2024(split="test", max_examples=1000)
    
    # Show original distribution
    print("\nOriginal AIR-Bench distribution:")
    dist = get_category_distribution(air_bench)
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")
    
    # Balance to specific target counts
    print("\n--- Balancing to custom targets ---")
    target_counts = {
        SafetyCategory.VIOLENCE: 300,
        SafetyCategory.CYBERCRIME: 200,
        SafetyCategory.DRUGS: 100,
        SafetyCategory.SELF_HARM: 0,  # Not in AIR-Bench
        SafetyCategory.SAFE: 0,  # AIR-Bench has no safe examples
    }
    
    balanced = balance_by_category(
        air_bench,
        target_counts=target_counts,
        random_seed=42
    )
    
    print("\nBalanced distribution:")
    dist = get_category_distribution(balanced)
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")
    
    return balanced


def main():
    """Run all balance tests"""
    print("Testing Balance Utilities with New Datasets")
    print("=" * 80)
    
    # Test 1: Safe/Unsafe balance
    balanced1 = test_balance_with_air_bench()
    
    # Test 2: Category balance
    balanced2 = test_category_balance_with_new_datasets()
    
    # Test 3: Custom category balance
    balanced3 = test_custom_category_balance()
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\n✓ All balance tests completed successfully!")
    print("\nKey points:")
    print("  1. AIR-Bench 2024 is 100% unsafe (all malicious prompts)")
    print("  2. Use combine_and_balance() to balance with safe datasets")
    print("  3. Use create_balanced_category_dataset() for equal category representation")
    print("  4. Use balance_by_category() for custom category targets")
    print("\nNote: TrustLLM may require HuggingFace access (gated dataset)")


if __name__ == "__main__":
    main()
