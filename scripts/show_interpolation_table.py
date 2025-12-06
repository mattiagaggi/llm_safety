"""Show interpolation table: prompt1, prompt2, layer, interpolated_prompt"""

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
    align_representations_procrustes
)


def prepare_category_prompts(dataset, category, min_length=5, max_length=50, max_prompts=30):
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
    
    NOTE: Replace extract_activations_simple() with your actual model code:
    return model.extract_layer_activations(prompts, layer=layer)
    """
    # Placeholder - replace with actual model extraction
    hidden_dim = 768
    activations = []
    for prompt in prompts:
        seq_len = max(len(prompt.split()), 1)
        act = torch.randn(seq_len, hidden_dim)
        activations.append(act)
    return activations


def main():
    """Generate interpolation table showing prompt1, prompt2, layer, interpolated_prompt."""
    
    print("=" * 80)
    print("Cross-Category Prompt Interpolation")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets...")
    harmfulqa = load_harmfulqa(split="train", max_examples=500)
    safetybench = load_safetybench(config="test", split="en", max_examples=500)
    self_harm = load_self_harm_synthetic(split="train", max_examples=500)
    drug_reviews = load_drug_reviews(split="train", max_examples=500, filter_illegal=True)
    
    # Placeholder model (replace with actual ModelLoader)
    class DummyModel:
        pass
    model = DummyModel()
    
    # Category pairs to interpolate
    category_pairs = [
        (SafetyCategory.VIOLENCE, SafetyCategory.SELF_HARM, harmfulqa, self_harm, "Violence ↔ Self-Harm"),
        (SafetyCategory.VIOLENCE, SafetyCategory.CYBERCRIME, harmfulqa, safetybench, "Violence ↔ Cybercrime"),
        (SafetyCategory.SELF_HARM, SafetyCategory.DRUGS, self_harm, drug_reviews, "Self-Harm ↔ Drugs"),
    ]
    
    # Layers to test
    layers = [4, 8, 12]
    
    all_results = []
    
    for cat1, cat2, ds1, ds2, pair_name in category_pairs:
        print(f"\n{'='*80}")
        print(f"Processing: {pair_name}")
        print(f"{'='*80}")
        
        # Prepare prompts
        prompts1 = prepare_category_prompts(ds1, cat1, max_prompts=20)
        prompts2 = prepare_category_prompts(ds2, cat2, max_prompts=20)
        
        if len(prompts1) == 0 or len(prompts2) == 0:
            print(f"  ⚠️  Skipping: {cat1.value}={len(prompts1)}, {cat2.value}={len(prompts2)}")
            continue
        
        print(f"  {cat1.value}: {len(prompts1)} prompts")
        print(f"  {cat2.value}: {len(prompts2)} prompts")
        
        # Extract and align activations for each layer
        for layer in layers:
            print(f"  Layer {layer}...", end=" ")
            
            # Extract activations
            activations1_list = extract_activations(model, prompts1, layer)
            activations2_list = extract_activations(model, prompts2, layer)
            
            # Align
            all_activations = activations1_list + activations2_list
            aligned = align_representations_procrustes(all_activations, reference_idx=0)
            
            aligned1 = aligned[:len(prompts1)]
            aligned2 = aligned[len(prompts1):]
            
            # Find good pairs
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(aligned1, aligned2)
            
            pairs = []
            for i in range(len(prompts1)):
                for j in range(len(prompts2)):
                    sim = similarities[i, j]
                    if 0.2 <= sim <= 0.8:  # Moderate similarity
                        pairs.append((i, j, sim))
            
            pairs.sort(key=lambda x: abs(x[2] - 0.5))
            pairs = pairs[:3]  # Top 3 pairs per layer
            
            if not pairs:
                print("No suitable pairs found")
                continue
            
            # Interpolate
            interpolator = LatentInterpolator(interpolation_method='spherical')
            all_aligned = np.vstack([aligned1, aligned2])
            all_prompts = prompts1 + prompts2
            
            for i, j, sim in pairs:
                prompt1 = prompts1[i]
                prompt2 = prompts2[j]
                z1 = aligned1[i]
                z2 = aligned2[j]
                
                # Interpolate (get middle point)
                z_interp = interpolator.interpolate(z1, z2, num_steps=5)
                z_mid = z_interp[2]  # Middle interpolation point
                
                # Decode (nearest neighbor)
                distances = np.linalg.norm(all_aligned - z_mid, axis=1)
                nearest_idx = np.argmin(distances)
                interpolated_prompt = all_prompts[nearest_idx]
                
                all_results.append({
                    'prompt1': prompt1,
                    'prompt2': prompt2,
                    'layer': layer,
                    'interpolated_prompt': interpolated_prompt
                })
            
            print(f"✓ {len(pairs)} pairs")
    
    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Display table
        print("\n" + "=" * 80)
        print("INTERPOLATION TABLE")
        print("=" * 80)
        print("\nColumns: prompt1 | prompt2 | layer | interpolated_prompt")
        print("-" * 80)
        
        # Show table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        # Display in groups by layer for readability
        for layer in sorted(df['layer'].unique()):
            layer_df = df[df['layer'] == layer]
            print(f"\n--- Layer {layer} ---")
            for idx, row in layer_df.iterrows():
                print(f"\nPrompt 1: {row['prompt1'][:60]}...")
                print(f"Prompt 2: {row['prompt2'][:60]}...")
                print(f"Layer: {row['layer']}")
                print(f"Interpolated: {row['interpolated_prompt'][:60]}...")
                print("-" * 80)
        
        # Save to CSV
        output_file = Path("results/interpolation_table.csv")
        output_file.parent.mkdir(exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to {output_file}")
        
        # Save formatted table
        md_file = Path("results/interpolation_table.md")
        with open(md_file, 'w') as f:
            f.write("# Interpolation Table\n\n")
            f.write("| Prompt 1 | Prompt 2 | Layer | Interpolated Prompt |\n")
            f.write("|---------|----------|-------|---------------------|\n")
            for _, row in df.iterrows():
                p1 = row['prompt1'].replace('|', '\\|')[:50]
                p2 = row['prompt2'].replace('|', '\\|')[:50]
                interp = row['interpolated_prompt'].replace('|', '\\|')[:50]
                f.write(f"| {p1}... | {p2}... | {row['layer']} | {interp}... |\n")
        print(f"✓ Saved to {md_file}")
        
        print(f"\nTotal interpolations: {len(df)}")
    else:
        print("\n⚠️  No results generated. Check that:")
        print("   1. Datasets have enough prompts")
        print("   2. Model extraction is implemented")
        print("   3. Categories have overlapping examples")


if __name__ == "__main__":
    main()
