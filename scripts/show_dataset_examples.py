"""Show example prompts from each dataset to demonstrate differences"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loaders import (
    load_beavertails,
    load_harmfulqa,
    load_toxigen,
    load_measuring_hate_speech,
    load_alpaca,
    load_red_team_attempts,
    load_mental_health_counseling,
    load_self_harm_synthetic,
    load_suicide_dataset,
    load_drug_reviews,
    load_safetybench,
)


def show_examples(dataset_name, dataset, max_examples=3):
    """Display examples from a dataset"""
    print("\n" + "=" * 80)
    print(f"{dataset_name}")
    print("=" * 80)
    
    safe_examples = [e for e in dataset.examples if not e.is_unsafe]
    unsafe_examples = [e for e in dataset.examples if e.is_unsafe]
    
    print(f"\nTotal examples: {len(dataset)}")
    print(f"  Safe: {len(safe_examples)}")
    print(f"  Unsafe: {len(unsafe_examples)}")
    
    # Show safe examples
    if safe_examples:
        print(f"\n📗 Safe Examples ({min(max_examples, len(safe_examples))}):")
        for i, example in enumerate(safe_examples[:max_examples], 1):
            text = example.text
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"\n  Example {i}:")
            print(f"    Category: {example.category.value}")
            print(f"    Text: {text}")
            if example.metadata:
                print(f"    Metadata: {example.metadata}")
    
    # Show unsafe examples
    if unsafe_examples:
        print(f"\n📕 Unsafe Examples ({min(max_examples, len(unsafe_examples))}):")
        for i, example in enumerate(unsafe_examples[:max_examples], 1):
            text = example.text
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"\n  Example {i}:")
            print(f"    Category: {example.category.value}")
            print(f"    Text: {text}")
            if example.metadata:
                print(f"    Metadata: {example.metadata}")


def main():
    """Show examples from all datasets"""
    print("=" * 80)
    print("Dataset Prompt Examples - Showing Differences")
    print("=" * 80)
    
    datasets_to_load = [
        ("BeaverTails", lambda: load_beavertails(split="train", max_examples=100, include_safe=True)),
        ("HarmfulQA", lambda: load_harmfulqa(split="train", max_examples=50)),
        ("ToxiGen", lambda: load_toxigen(split="train", max_examples=50, min_toxicity=0.5)),
        ("Measuring Hate Speech", lambda: load_measuring_hate_speech(split="train", max_examples=50, min_hate_score=0.5)),
        ("Alpaca", lambda: load_alpaca(split="train", max_examples=50)),
        ("Red Team Attempts", lambda: load_red_team_attempts(split="train", max_examples=50)),
        ("Mental Health Counseling", lambda: load_mental_health_counseling(split="train", max_examples=50, filter_self_harm=True)),
        ("Self-Harm Synthetic", lambda: load_self_harm_synthetic(split="train", max_examples=50)),
        ("Suicide Dataset", lambda: load_suicide_dataset(split="train", max_examples=50)),
        ("Drug Reviews", lambda: load_drug_reviews(split="train", max_examples=50, filter_illegal=True)),
        ("SafetyBench", lambda: load_safetybench(config="test", split="en", max_examples=50)),
    ]
    
    for name, loader_func in datasets_to_load:
        try:
            dataset = loader_func()
            show_examples(name, dataset, max_examples=2)
        except Exception as e:
            print(f"\n❌ Error loading {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Summary of Differences")
    print("=" * 80)
    print("""
1. **BeaverTails**: Conversation responses (chosen vs rejected pairs)
2. **HarmfulQA**: Harmful questions with topics
3. **ToxiGen**: Toxic statements about minority groups
4. **Measuring Hate Speech**: Social media posts/comments
5. **Alpaca**: Instruction-following examples (instruction + output)
6. **Red Team Attempts**: Full conversation transcripts
7. **Mental Health Counseling**: Counseling conversations (context + response)
8. **Self-Harm Synthetic**: Evaluation prompts/instructions
9. **Suicide Dataset**: Short text snippets
10. **Drug Reviews**: Product reviews
11. **SafetyBench**: Multiple-choice questions with options
    """)


if __name__ == "__main__":
    main()

