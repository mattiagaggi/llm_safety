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

### 2. **HarmfulQA** ✅ VERIFIED (BEST FOR REFUSAL TESTING)
- **URL**: https://huggingface.co/datasets/declare-lab/HarmfulQA
- **Description**: Harmful question-answering dataset with multiple categories
- **Categories**: Self-harm, violence, drugs, cybercrime, etc.
- **Size**: 1,960 examples
- **Structure**: Contains conversations (blue=helpful, red=harmful)
- **Note**: ✅ Contains DIRECT harmful requests (e.g., "How can I make a bomb?") that models SHOULD refuse
- **Usage**:
  ```python
  from src.data import load_harmfulqa
  dataset = load_harmfulqa(split="train", max_examples=1000)
  # All examples are harmful - models should refuse these
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

### 4. **SafetyBench** ✅ VERIFIED
- **URL**: https://huggingface.co/datasets/thu-coai/SafetyBench
- **Description**: Comprehensive safety evaluation benchmark with 11,435 questions
- **Categories**: 7 safety categories (self-harm, violence, drugs, cybercrime, etc.)
- **Size**: 11,435 examples (English) + 11,435 (Chinese) + 2,100 (Chinese subset)
- **Structure**: Multiple-choice questions with categories
- **Note**: ⚠️ SafetyBench asks "Is this text harmful?" - this is an EVALUATION task, not a harmful request. Models should answer, not refuse.
- **Usage**:
  ```python
  from src.data import load_safetybench
  dataset = load_safetybench(config="test", split="en", max_examples=1000)
  ```

### 5. **AIR-Bench 2024**
- **URL**: https://huggingface.co/datasets/stanford-crfm/air-bench-2024
- **Description**: AI safety benchmark aligned with emerging government regulations and company policies
- **Categories**: 314 risk categories including cybersecurity, CSAM, surveillance, and more
- **Size**: 5,694 test examples across 314 risk categories
- **Structure**: Measures refused requests; reports mean % appropriate refusals
- **Note**: Focuses on measuring appropriate refusal behavior across diverse malicious prompts
- **Usage**:
  ```python
  from src.data import load_air_bench_2024
  dataset = load_air_bench_2024(split="test", max_examples=1000)
  # Optional: filter by region or categories
  dataset = load_air_bench_2024(split="test", region="us", max_examples=1000)
  ```

### 6. **TrustLLM**
- **URL**: https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset
- **Description**: Comprehensive trustworthiness evaluation across 6 dimensions over 30+ datasets
- **Categories**: Truthfulness, safety, fairness, robustness, privacy, machine ethics
- **Size**: 30+ integrated datasets
- **Structure**: Multi-dimensional evaluation framework including jailbreak testing
- **Note**: ⚠️ **Gated Dataset** - Requires HuggingFace access approval. Visit the dataset page to request access, then authenticate with `huggingface-cli login`
- **Usage**:
  ```python
  from src.data import load_trustllm
  # Load safety category (most relevant for safety evaluation)
  dataset = load_trustllm(category="safety", max_examples=1000)
  # Other categories: "truthfulness", "fairness", "robustness", "privacy", "ethics"
  ```

### 7. **Anthropic Red Team Dataset** ✅ VERIFIED
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

### Self-Harm ✅ VERIFIED
- **Mental Health Counseling** ✅: `Amod/mental_health_counseling_conversations`
  - Usage: `load_mental_health_counseling(filter_self_harm=True)`
- **Self-Harm Synthetic Eval** ✅: `arianaazarbal/self-harm-synthetic-eval`
  - Usage: `load_self_harm_synthetic()`
- **Suicide Dataset** ✅: `jquiros/suicide`
  - Usage: `load_suicide_dataset()`
- **BeaverTails** filtered for self-harm category

### Violence
- **ToxiGen** (violence subset)
- **Hate Speech Datasets**:
  - https://huggingface.co/datasets/hate_speech18
  - https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech

### Drugs ✅ VERIFIED
- **Drug Reviews** ✅: `lewtun/drug-reviews`
  - Usage: `load_drug_reviews(filter_illegal=True)`
- **BeaverTails** filtered for drug-related content
- **HarmfulQA** drug category

### Cybercrime
- **HarmfulQA** cybercrime category
- **Anthropic Red Team** filtered for hacking/cybercrime

## Safe/Control Datasets ✅ VERIFIED

For contrast, you'll also need safe examples:
- **Alpaca Dataset** ✅: `tatsu-lab/alpaca`
  - Usage: `load_alpaca()`
- **Measuring Hate Speech** ✅: `ucberkeley-dlab/measuring-hate-speech`
  - Usage: `load_measuring_hate_speech(min_hate_score=0.5)` (lower scores = safe)
- **BeaverTails** helpful (non-harmful) subset

## Category Balance Status (with SafetyBench)

**Current distribution** (all datasets combined):
- **Self-harm**: 1,996 examples (12.7% of unsafe) ✅ **Improved significantly**
- **Violence**: 12,258 examples (78.1% of unsafe) ⚠️ **Still dominant**
- **Drugs**: 96 examples (0.6% of unsafe) ⚠️ **Still very low**
- **Cybercrime**: 1,342 examples (8.6% of unsafe) ✅ **Improved significantly**
- **Safe**: 2,500 examples (13.7% of total)

**SafetyBench contribution:**
- Self-harm: +1,566 examples
- Violence: +8,461 examples
- Drugs: +70 examples
- Cybercrime: +1,338 examples

**Balance ratio**: 0.78% (min/max) - **Still highly imbalanced**

**Recommendations:**
- Use `create_balanced_category_dataset()` to balance categories
- Consider keyword filtering on existing datasets for drugs
- Drugs category needs more data sources



## Implementation Notes

1. **Data Balance**: Use balancing utilities (`balance_utils.py`, `category_balance.py`) to create balanced datasets
2. **Quality**: Prefer human-annotated datasets over synthetic
3. **Ethics**: Use datasets with proper licenses and ethical guidelines
4. **Size**: Aim for at least 1k examples per category for robust results
5. **Splits**: Some datasets have native train/test splits, others need manual splitting (see `DATASET_SPLITS.md`)

## All Available Loaders

```python
from src.data import (
    # Main datasets
    load_beavertails,
    load_harmfulqa,
    load_red_team_attempts,
    load_toxigen,
    
    # Category-specific
    load_mental_health_counseling,  # Self-harm
    load_self_harm_synthetic,        # Self-harm
    load_suicide_dataset,            # Self-harm
    load_drug_reviews,               # Drugs
    load_safetybench,                # Comprehensive benchmark
    
    # Evaluation benchmarks
    load_air_bench_2024,            # AIR-Bench 2024
    load_trustllm,                  # TrustLLM
    
    # Safe/control
    load_measuring_hate_speech,
    load_alpaca,
)
```

## Quick Start

See `src/data/data_loader.py` for implementation examples. The loader supports:
- Loading from HuggingFace datasets
- Custom CSV/JSON files
- Filtering by category
- Train/test splitting
- Balancing utilities (see `src/data/balance_utils.py` and `category_balance.py`)

