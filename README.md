# Discovering Unsafe-Intent Layers in Large Language Models

This repository contains the code for the paper "Discovering Unsafe-Intent Layers in Large Language Models".

## Overview

This project uses linear probes to identify which transformer layers encode unsafe intent most strongly. We create the first "Unsafe Intent Layer Atlas" for LLMs by:

1. **Layer-wise Linear Probing**: Training probes at each layer to measure separability of unsafe vs safe intents
2. **Multi-Category Analysis**: Comparing different safety categories (self-harm, violence, drugs, cybercrime)

## Project Structure

```
llm_safety/
├── src/
│   ├── models/          # Model loading and feature extraction
│   ├── probing/         # Linear probing implementation
│   ├── data/            # Data loading and preprocessing
│   └── visualization/   # Plotting utilities
├── experiments/         # Main experiment scripts
├── scripts/             # Utility scripts
└── datasets_cache/      # Downloaded datasets (gitignored)

```

## Installation

Using `uv` (recommended):
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -e .
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download and Verify Datasets

```bash
# Download datasets and verify loaders work
python scripts/download_and_verify.py
```

This will:
- Download datasets to `datasets_cache/` (gitignored)
- Verify all loaders work correctly
- Show dataset statistics

### 2. Run Experiments

```bash
# Run layer-wise probing for all safety categories
python experiments/run_layer_probing.py --model gpt2 --categories all
```

## Datasets

See [DATASETS.md](DATASETS.md) for recommended datasets. Key datasets:
- **BeaverTails** (Anthropic): `Anthropic/hh-rlhf`
- **HarmfulQA**: `declare-lab/HarmfulQA`

Datasets are automatically downloaded to `datasets_cache/` on first use.

## Usage

### Loading Datasets

```python
from src.data import load_beavertails, load_harmfulqa

# Load BeaverTails
dataset = load_beavertails(split="train", max_examples=10000)

# Load HarmfulQA
dataset = load_harmfulqa(split="train", max_examples=1000)
```

### Running Layer Probing

```python
from src.models import ModelLoader
from src.data import load_beavertails
from src.probing import train_layer_probes

# Load model and data
model = ModelLoader("gpt2")
dataset = load_beavertails(max_examples=1000)

# Extract activations
activations = model.extract_all_layer_activations(dataset.get_texts())

# Train probes
probes = train_layer_probes(activations, dataset.get_labels())
```

## Data Location

- **Downloaded datasets**: `datasets_cache/` (gitignored, local to project)
- **Results**: `results/` (gitignored)
- **Models**: Downloaded to HuggingFace cache (typically `~/.cache/huggingface/`)

## Citation

If you use this code, please cite:

```bibtex
@article{unsafe_intent_layers,
  title={Discovering Unsafe-Intent Layers in Large Language Models},
  author={...},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
