# Recommended Datasets for Unsafe Intent Layer Discovery

This document outlines recommended datasets for each safety category in your research.

## Primary Recommendations

### 1. **BeaverTails** (Anthropic) ✅ VERIFIED
- **URL**: https://huggingface.co/datasets/Anthropic/hh-rlhf
- **Description**: High-quality human-annotated dataset of helpful and harmful prompts
- **Categories**: Covers multiple safety categories
- **Size**: 160,800 examples (train) + 8,552 (test)
- **Structure**: `chosen` (helpful) and `rejected` (harmful) responses
- **Usage**: 
  ```python
  from datasets import load_dataset
  dataset = load_dataset("Anthropic/hh-rlhf", split="train")
  # Use 'chosen' for safe examples, 'rejected' for unsafe
  ```

### 2. **HarmfulQA** ✅ VERIFIED
- **URL**: https://huggingface.co/datasets/declare-lab/HarmfulQA
- **Description**: Harmful question-answering dataset with multiple categories
- **Categories**: Self-harm, violence, drugs, cybercrime, etc.
- **Size**: 1,960 examples
- **Structure**: Contains conversations (blue=helpful, red=harmful)
- **Usage**:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("declare-lab/HarmfulQA", split="train")
  # Note: Structure is different - contains conversations, not simple text
  ```

### 3. **ToxiGen** ✅ VERIFIED
- **URL**: https://huggingface.co/datasets/toxigen/toxigen-data
- **Description**: Machine-generated toxic and benign statements about minority groups
- **Categories**: Violence, hate speech, toxicity (primarily)
- **Size**: 8,960 train + 940 test examples
- **Structure**: Contains text, toxicity scores (AI and human), target groups, annotations
- **Usage**:
  ```python
  from src.data import load_toxigen
  dataset = load_toxigen(split="train", max_examples=1000, min_toxicity=0.5)
  ```

### 4. **SafetyBench**
- **URL**: https://github.com/thu-coai/SafetyBench
- **Description**: Comprehensive safety evaluation benchmark
- **Categories**: Multiple safety categories
- **Usage**: Download from GitHub repository

### 5. **Anthropic Red Team Dataset** ✅ VERIFIED
- **URL**: https://huggingface.co/datasets/Anthropic/hh-rlhf (use `data_dir="red-team-attempts"`)
- **Description**: Red team attempts to elicit harmful outputs
- **Categories**: Various harmful categories (self-harm, violence, drugs, cybercrime)
- **Size**: ~38k examples
- **Structure**: Contains transcripts, ratings, task descriptions, and tags
- **Usage**:
  ```python
  from src.data import load_red_team_attempts
  dataset = load_red_team_attempts(split="train", max_examples=1000)
  ```

## Category-Specific Datasets

### Self-Harm
- **Crisis Text Line** (if available with proper permissions)
- **Reddit r/SuicideWatch** (with proper anonymization)
- **BeaverTails** filtered for self-harm category

### Violence
- **ToxiGen** (violence subset)
- **Hate Speech Datasets**:
  - https://huggingface.co/datasets/hate_speech18
  - https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech

### Drugs
- **BeaverTails** filtered for drug-related content
- **HarmfulQA** drug category

### Cybercrime
- **HarmfulQA** cybercrime category
- **Anthropic Red Team** filtered for hacking/cybercrime

## Safe/Control Datasets

For contrast, you'll also need safe examples:
- **Alpaca Dataset**: https://huggingface.co/datasets/tatsu-lab/alpaca
- **OpenWebText**: General web text
- **BeaverTails** helpful (non-harmful) subset

## Implementation Notes

1. **Data Balance**: Ensure balanced safe/unsafe examples per category
2. **Quality**: Prefer human-annotated datasets over synthetic
3. **Ethics**: Use datasets with proper licenses and ethical guidelines
4. **Size**: Aim for at least 1k examples per category for robust results

## Quick Start

See `src/data/data_loader.py` for implementation examples. The loader supports:
- Loading from HuggingFace datasets
- Custom CSV/JSON files
- Filtering by category
- Train/test splitting

