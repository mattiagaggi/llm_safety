# Quick Start Guide

## Setup

1. **Create and activate virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -e .
   ```

## Using Real Datasets

### Option 1: BeaverTails (Recommended)

```python
from src.data import load_beavertails, SafetyCategory

# Load BeaverTails dataset
dataset = load_beavertails(split="train", max_examples=10000)

# Filter by category if needed
self_harm = SafetyDataset(
    [e for e in dataset.examples if e.category == SafetyCategory.SELF_HARM]
)
```

### Option 2: HarmfulQA

```python
from src.data import load_harmfulqa

# Load HarmfulQA dataset
dataset = load_harmfulqa(split="train", max_examples=1000)
```

### Option 3: Combined Dataset

```python
from src.data import load_combined_dataset

# Load combined dataset from multiple sources
dataset = load_combined_dataset(
    beavertails_max=10000,
    harmfulqa_max=1000,
    include_safe=True
)
```

## Run Experiments

```bash
# Run layer-wise probing
python experiments/run_layer_probing.py --model gpt2 --categories all
```

## Download and Verify Datasets

**First time setup - download and verify all datasets:**

```bash
# Download datasets and verify loaders work
python scripts/download_and_verify.py
```

This script will:
- Download datasets from HuggingFace (cached for future use)
- Verify custom loaders work correctly
- Show dataset statistics
- Test data access and filtering

**Alternative - just test access:**

```bash
# Test if datasets are accessible
python scripts/test_datasets.py
```

## Notes

- **First download**: Datasets are automatically downloaded and cached by HuggingFace on first use
- **Download location**: `~/.cache/huggingface/datasets/`
- **Internet required**: First-time downloads require internet connection
- **Size**: BeaverTails is ~160k examples, may take a few minutes to download

