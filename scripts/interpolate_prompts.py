"""Script to interpolate in latent space to generate new unsafe prompts"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_harmfulqa, load_safetybench, SafetyCategory
from src.geometry.latent_interpolation import LatentInterpolator
from src.models import ModelLoader  # Assuming this exists


def prepare_prompts_for_interpolation(dataset, category, min_length=5, max_length=50):
    """
    Filter prompts to similar format/length for better interpolation.
    """
    # Get prompts of same category
    prompts = [e.text for e in dataset.examples 
               if e.is_unsafe and e.category == category]
    
    # Filter by word length
    filtered = []
    for p in prompts:
        words = p.split()
        if min_length <= len(words) <= max_length:
            filtered.append(p)
    
    return filtered


def main():
    """Main interpolation pipeline"""
    print("=" * 80)
    print("Latent Space Interpolation for Unsafe Prompts")
    print("=" * 80)
    
    # 1. Load datasets with similar formats
    print("\n1. Loading datasets...")
    
    # HarmfulQA - questions, good format for interpolation
    harmfulqa = load_harmfulqa(split="train", max_examples=1000)
    harmfulqa_prompts = prepare_prompts_for_interpolation(
        harmfulqa, SafetyCategory.VIOLENCE, min_length=5, max_length=30
    )
    
    print(f"  HarmfulQA: {len(harmfulqa_prompts)} prompts")
    
    # SafetyBench - questions, consistent format
    safetybench = load_safetybench(config="test", split="en", max_examples=1000)
    safetybench_prompts = prepare_prompts_for_interpolation(
        safetybench, SafetyCategory.VIOLENCE, min_length=10, max_length=50
    )
    
    print(f"  SafetyBench: {len(safetybench_prompts)} prompts")
    
    # Combine prompts
    all_prompts = harmfulqa_prompts[:500] + safetybench_prompts[:500]
    print(f"  Total: {len(all_prompts)} prompts")
    
    # 2. Load model and extract activations
    print("\n2. Extracting activations...")
    model = ModelLoader("gpt2")  # Or your target model
    
    # Extract activations from target layer (e.g., layer 8)
    target_layer = 8
    activations_list = []
    valid_prompts = []
    
    for prompt in all_prompts:
        try:
            # Extract activation for this prompt
            act = model.extract_layer_activations([prompt], layer=target_layer)
            if act is not None and len(act) > 0:
                activations_list.append(act[0])  # [seq_len, hidden_dim]
                valid_prompts.append(prompt)
        except Exception as e:
            print(f"  Warning: Failed to extract activation for prompt: {e}")
            continue
    
    print(f"  Extracted activations for {len(activations_list)} prompts")
    
    # 3. Align representations using Procrustes
    print("\n3. Aligning representations with Procrustes...")
    interpolator = LatentInterpolator(
        alignment_method='procrustes',
        interpolation_method='spherical'
    )
    
    aligned_activations = interpolator.align(activations_list)
    print(f"  Aligned shape: {aligned_activations.shape}")
    
    # 4. Find interpolation pairs
    print("\n4. Finding interpolation pairs...")
    from src.geometry.latent_interpolation import find_interpolation_pairs
    
    pairs = find_interpolation_pairs(
        aligned_activations, valid_prompts, num_pairs=50
    )
    print(f"  Found {len(pairs)} pairs")
    
    # Show some example pairs
    print("\n  Example pairs:")
    for i, (idx1, idx2, sim) in enumerate(pairs[:3]):
        print(f"    Pair {i+1} (similarity: {sim:.3f}):")
        print(f"      Prompt 1: {valid_prompts[idx1][:60]}...")
        print(f"      Prompt 2: {valid_prompts[idx2][:60]}...")
    
    # 5. Generate interpolated prompts
    print("\n5. Generating interpolated prompts...")
    interpolated_latents, decoded_texts = interpolator.generate(
        aligned_activations, valid_prompts, num_pairs=50, num_steps=5
    )
    
    print(f"  Generated {len(decoded_texts)} interpolated prompts")
    
    # 6. Show results
    print("\n6. Sample interpolated prompts:")
    for i, text in enumerate(decoded_texts[:10]):
        print(f"  {i+1}. {text[:80]}...")
    
    # 7. Save results
    output_file = Path("results/interpolated_prompts.txt")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("Interpolated Unsafe Prompts\n")
        f.write("=" * 80 + "\n\n")
        for i, text in enumerate(decoded_texts, 1):
            f.write(f"{i}. {text}\n")
    
    print(f"\n✓ Saved {len(decoded_texts)} prompts to {output_file}")
    
    print("\n" + "=" * 80)
    print("Note: These are decoded using nearest neighbor.")
    print("For better results, consider using LLM-based generation with activation steering.")
    print("=" * 80)


if __name__ == "__main__":
    main()
