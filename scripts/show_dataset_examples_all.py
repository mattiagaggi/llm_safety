"""Show 2-3 examples from each dataset"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_beavertails,
    load_harmfulqa,
    load_red_team_attempts,
    load_toxigen,
    load_measuring_hate_speech,
    load_alpaca,
    load_mental_health_counseling,
    load_self_harm_synthetic,
    load_suicide_dataset,
    load_drug_reviews,
    load_safetybench,
    load_air_bench_2024,
    load_trustllm,
)


def show_examples(name, loader_func, *args, **kwargs):
    """Show examples from a dataset"""
    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)
    
    try:
        dataset = loader_func(*args, **kwargs)
        
        if len(dataset) == 0:
            print("  No examples loaded")
            return
        
        print(f"\nTotal examples: {len(dataset)}")
        print(f"Unsafe: {dataset.get_labels().sum()}, Safe: {(~dataset.get_labels().astype(bool)).sum()}")
        
        # Show 2-3 examples
        num_examples = min(3, len(dataset))
        for i, example in enumerate(dataset.examples[:num_examples], 1):
            print(f"\n--- Example {i} ---")
            print(f"Category: {example.category.value}")
            print(f"Unsafe: {example.is_unsafe}")
            text_preview = example.text[:300] + "..." if len(example.text) > 300 else example.text
            print(f"Text:\n{text_preview}")
            if example.metadata:
                # Show a few key metadata fields
                metadata_keys = list(example.metadata.keys())[:3]
                if metadata_keys:
                    print(f"Metadata: {', '.join(metadata_keys)}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Show examples from all datasets"""
    print("=" * 80)
    print("Dataset Examples - 2-3 examples from each dataset")
    print("=" * 80)
    
    # Main datasets
    show_examples("BeaverTails", load_beavertails, 
                  split="train", max_examples=100, include_safe=True)
    
    show_examples("HarmfulQA", load_harmfulqa, 
                  split="train", max_examples=50)
    
    show_examples("Red Team Attempts", load_red_team_attempts, 
                  split="train", max_examples=50)
    
    show_examples("ToxiGen", load_toxigen, 
                  split="train", max_examples=50, min_toxicity=0.5)
    
    show_examples("SafetyBench", load_safetybench, 
                  config="test", split="en", max_examples=50)
    
    show_examples("AIR-Bench 2024", load_air_bench_2024, 
                  split="test", max_examples=50)
    
    show_examples("TrustLLM", load_trustllm, 
                  category="safety", max_examples=50)
    
    # Safe datasets
    show_examples("Alpaca", load_alpaca, 
                  split="train", max_examples=50)
    
    show_examples("Measuring Hate Speech", load_measuring_hate_speech, 
                  split="train", max_examples=50, min_hate_score=0.0)
    
    # Category-specific
    show_examples("Mental Health Counseling", load_mental_health_counseling, 
                  split="train", max_examples=20, filter_self_harm=True)
    
    show_examples("Self-Harm Synthetic", load_self_harm_synthetic, 
                  split="train", max_examples=20)
    
    show_examples("Suicide Dataset", load_suicide_dataset, 
                  split="train", max_examples=20)
    
    show_examples("Drug Reviews", load_drug_reviews, 
                  split="train", max_examples=20, filter_illegal=True)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
