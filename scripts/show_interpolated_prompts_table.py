"""Show interpolated prompts in a table format: prompt1, prompt2, layer, interpolated"""

import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_harmfulqa,
    load_safetybench,
    load_self_harm_synthetic,
    load_drug_reviews,
    SafetyCategory
)
from src.geometry.latent_interpolation import (
    LatentInterpolator,
    align_representations_procrustes,
    interpolate_latent
)


def prepare_category_prompts(dataset, category, min_length=5, max_length=50, max_prompts=50):
    """Extract prompts of a specific category."""
    prompts = []
    for e in dataset.examples:
        if e.is_unsafe and e.category == category:
            words = e.text.split()
            if min_length <= len(words) <= max_length:
                prompts.append(e.text)
                if len(prompts) >= max_prompts:
                    break
    return prompts


def extract_activations(model, prompts, layer):
    """
    Extract activations from model.
    
    NOTE: Replace this with your actual model extraction code:
    return model.extract_layer_activations(prompts, layer=layer)
    """
    # Placeholder - returns random activations for demonstration
    hidden_dim = 768
    activations = []
    for prompt in prompts:
        seq_len = len(prompt.split())
        act = torch.randn(seq_len, hidden_dim)
        activations.append(act)
    return activations


def create_interpolation_table(
    category1: SafetyCategory,
    category2: SafetyCategory,
    dataset1,
    dataset2,
    model,
    layer: int,
    num_pairs: int = 5,
    num_steps: int = 3
):
    """
    Create a table showing interpolated prompts between two categories.
    
    Returns:
        pandas DataFrame with columns: prompt1, prompt2, layer, interpolated_prompt
    """
    # Prepare prompts
    prompts1 = prepare_category_prompts(dataset1, category1, max_prompts=30)
    prompts2 = prepare_category_prompts(dataset2, category2, max_prompts=30)
    
    if len(prompts1) == 0 or len(prompts2) == 0:
        print(f"⚠️  Not enough prompts: {category1.value}={len(prompts1)}, {category2.value}={len(prompts2)}")
        return pd.DataFrame()
    
    # Extract activations
    activations1_list = extract_activations(model, prompts1, layer)
    activations2_list = extract_activations(model, prompts2, layer)
    
    # Align
    all_activations = activations1_list + activations2_list
    aligned = align_representations_procrustes(all_activations, reference_idx=0)
    
    aligned1 = aligned[:len(prompts1)]
    aligned2 = aligned[len(prompts1):]
    
    # Find pairs
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(aligned1, aligned2)
    
    pairs = []
    for i in range(len(prompts1)):
        for j in range(len(prompts2)):
            sim = similarities[i, j]
            if 0.2 <= sim <= 0.8:
                pairs.append((i, j, sim))
    
    pairs.sort(key=lambda x: abs(x[2] - 0.5))
    pairs = pairs[:num_pairs]
    
    # Interpolate
    interpolator = LatentInterpolator(interpolation_method='spherical')
    all_aligned = np.vstack([aligned1, aligned2])
    all_prompts = prompts1 + prompts2
    
    results = []
    
    for i, j, sim in pairs:
        prompt1 = prompts1[i]
        prompt2 = prompts2[j]
        z1 = aligned1[i]
        z2 = aligned2[j]
        
        # Interpolate (get middle point)
        z_interp = interpolator.interpolate(z1, z2, num_steps=num_steps + 2)
        z_mid = z_interp[len(z_interp) // 2]  # Middle interpolation
        
        # Decode (nearest neighbor)
        distances = np.linalg.norm(all_aligned - z_mid, axis=1)
        nearest_idx = np.argmin(distances)
        interpolated_prompt = all_prompts[nearest_idx]
        
        results.append({
            'prompt1': prompt1,
            'prompt2': prompt2,
            'category1': category1.value,
            'category2': category2.value,
            'layer': layer,
            'interpolated_prompt': interpolated_prompt,
            'similarity': f"{sim:.3f}"
        })
    
    return pd.DataFrame(results)


def main():
    """Generate interpolation tables for different category pairs."""
    
    print("=" * 80)
    print("Cross-Category Prompt Interpolation Table")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets...")
    harmfulqa = load_harmfulqa(split="train", max_examples=500)
    safetybench = load_safetybench(config="test", split="en", max_examples=500)
    self_harm = load_self_harm_synthetic(split="train", max_examples=500)
    drug_reviews = load_drug_reviews(split="train", max_examples=500, filter_illegal=True)
    
    # Placeholder model
    class DummyModel:
        pass
    model = DummyModel()
    
    # Test different layers
    layers = [4, 8, 12]
    
    all_results = []
    
    # Category pairs to test
    category_pairs = [
        (SafetyCategory.VIOLENCE, SafetyCategory.SELF_HARM, harmfulqa, self_harm),
        (SafetyCategory.VIOLENCE, SafetyCategory.CYBERCRIME, harmfulqa, safetybench),
        (SafetyCategory.SELF_HARM, SafetyCategory.DRUGS, self_harm, drug_reviews),
    ]
    
    for cat1, cat2, ds1, ds2 in category_pairs:
        print(f"\n{'='*80}")
        print(f"Processing: {cat1.value} ↔ {cat2.value}")
        print(f"{'='*80}")
        
        for layer in layers:
            print(f"  Layer {layer}...")
            df = create_interpolation_table(
                cat1, cat2, ds1, ds2, model, layer, num_pairs=3, num_steps=3
            )
            
            if not df.empty:
                all_results.append(df)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Display table
        print("\n" + "=" * 80)
        print("INTERPOLATION RESULTS TABLE")
        print("=" * 80)
        
        # Show in a readable format
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 60)
        
        print("\n" + combined_df.to_string(index=False))
        
        # Save to CSV
        output_file = Path("results/interpolated_prompts_table.csv")
        output_file.parent.mkdir(exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to {output_file}")
        
        # Save to markdown table
        md_file = Path("results/interpolated_prompts_table.md")
        with open(md_file, 'w') as f:
            f.write("# Interpolated Prompts Table\n\n")
            f.write(combined_df.to_markdown(index=False))
        print(f"✓ Saved to {md_file}")
        
        # Show summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total interpolations: {len(combined_df)}")
        print(f"Category pairs: {combined_df.groupby(['category1', 'category2']).size().to_dict()}")
        print(f"Layers tested: {sorted(combined_df['layer'].unique())}")
    else:
        print("\n⚠️  No results generated. Check that datasets have enough prompts.")


if __name__ == "__main__":
    main()
