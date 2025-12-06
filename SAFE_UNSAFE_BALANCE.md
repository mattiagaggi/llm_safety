# Safe vs Unsafe Content Balance Analysis

## Current Balance Status

**⚠️ NOT WELL BALANCED**

### Current Distribution (from DATASETS.md):
- **Safe examples**: 2,500 (13.7% of total)
- **Unsafe examples**: 15,692 (86.3% of total)
  - Self-harm: 1,996
  - Violence: 12,258
  - Drugs: 96
  - Cybercrime: 1,342

**Ratio**: ~1:6.3 (safe:unsafe) - **Highly imbalanced**

---

## Datasets Providing Safe Content

### 1. **BeaverTails** (Anthropic/hh-rlhf)
- **Safe examples**: `chosen` responses (helpful/safe)
- **Unsafe examples**: `rejected` responses (harmful)
- **Balance**: 1:1 ratio (each prompt has both chosen and rejected)
- **Total size**: 160,800 train examples = ~80,400 safe + ~80,400 unsafe
- **Note**: Only included if `include_safe=True` (default: True)

### 2. **Alpaca** (tatsu-lab/alpaca)
- **Safe examples**: ALL examples (instruction-following, all safe)
- **Unsafe examples**: 0
- **Total size**: 52,000+ examples (all safe)
- **Usage**: `load_alpaca()`

### 3. **Measuring Hate Speech** (ucberkeley-dlab/measuring-hate-speech)
- **Safe examples**: Examples with `hate_speech_score < min_hate_score` (default 0.5)
- **Unsafe examples**: Examples with `hate_speech_score >= min_hate_score`
- **Total size**: 135,000+ examples (mixed safe/unsafe)
- **Note**: Balance depends on `min_hate_score` threshold

### 4. **Mental Health Counseling** (Amod/mental_health_counseling_conversations)
- **Safe examples**: Conversations without self-harm keywords
- **Unsafe examples**: Conversations with self-harm keywords
- **Note**: Only if `filter_self_harm=False` (default: True, so mostly unsafe)

### 5. **Drug Reviews** (lewtun/drug-reviews)
- **Safe examples**: Reviews without illegal drug mentions
- **Unsafe examples**: Reviews with illegal drug mentions
- **Note**: Only if `filter_illegal=False` (default: True, so mostly unsafe)

---

## Datasets with NO Safe Content

All examples are unsafe:
- **HarmfulQA** - All 1,960 examples are harmful questions
- **ToxiGen** - All examples are toxic/benign statements (mostly toxic when filtered)
- **Red Team Attempts** - All ~38k examples are red teaming attempts
- **Self-Harm Synthetic** - All examples are self-harm related
- **Suicide Dataset** - All examples are suicide-related
- **SafetyBench** - All 11,435 examples are safety concerns (all unsafe)

---

## Why It's Imbalanced

1. **Most datasets focus on unsafe content** - 6 out of 10 datasets are 100% unsafe
2. **Limited safe datasets** - Only 3-4 datasets provide significant safe content:
   - BeaverTails (if `include_safe=True`)
   - Alpaca (all safe, but may not be used in all experiments)
   - Measuring Hate Speech (mixed, depends on threshold)
3. **Default filtering** - Some datasets filter out safe examples by default:
   - Mental Health Counseling: `filter_self_harm=True` (default)
   - Drug Reviews: `filter_illegal=True` (default)

---

## Recommendations to Improve Balance

### Option 1: Increase Safe Examples from Existing Datasets

```python
# Load more safe examples from BeaverTails
beavertails = load_beavertails(
    split="train",
    max_examples=50000,  # Increase this
    include_safe=True    # Ensure this is True
)

# Load more from Alpaca (all safe)
alpaca = load_alpaca(
    split="train",
    max_examples=50000  # Increase this
)

# Adjust Measuring Hate Speech threshold to get more safe examples
measuring_hate = load_measuring_hate_speech(
    split="train",
    max_examples=50000,
    min_hate_score=0.7  # Higher threshold = more safe examples
)
```

### Option 2: Include Safe Examples from Filtered Datasets

```python
# Include safe examples from Mental Health Counseling
mental_health = load_mental_health_counseling(
    split="train",
    max_examples=10000,
    filter_self_harm=False  # Include both safe and unsafe
)

# Include safe examples from Drug Reviews
drug_reviews = load_drug_reviews(
    split="train",
    max_examples=10000,
    filter_illegal=False  # Include both safe and unsafe
)
```

### Option 3: Use Balancing Utilities

```python
from src.data.category_balance import create_balanced_category_dataset

# Create balanced dataset (50/50 safe/unsafe)
balanced = create_balanced_category_dataset(
    target_per_category=5000,  # Equal number per category
    include_safe=True
)
```

### Option 4: Add More Safe Datasets

Consider adding:
- **Common Crawl** filtered for safe content
- **Wikipedia** articles
- **Stack Overflow** Q&A (technical, safe)
- **News articles** (filtered for neutral/safe content)

---

## Target Balance

For safety classification tasks, ideal ratios:
- **Minimum**: 30% safe, 70% unsafe (1:2.3)
- **Good**: 40% safe, 60% unsafe (1:1.5)
- **Ideal**: 50% safe, 50% unsafe (1:1)

**Current**: 13.7% safe, 86.3% unsafe (1:6.3) ⚠️ **Needs improvement**

---

## Quick Fix: Load More Safe Examples

To quickly improve balance, increase safe examples:

```python
from src.data import (
    load_beavertails,
    load_alpaca,
    load_measuring_hate_speech
)

# Load large amounts of safe examples
safe_datasets = [
    load_beavertails(split="train", max_examples=50000, include_safe=True),
    load_alpaca(split="train", max_examples=50000),
    load_measuring_hate_speech(split="train", max_examples=50000, min_hate_score=0.7)
]

# Combine safe examples
all_safe = []
for ds in safe_datasets:
    safe_examples = [e for e in ds.examples if not e.is_unsafe]
    all_safe.extend(safe_examples)

print(f"Total safe examples: {len(all_safe)}")
```

This should give you ~150k+ safe examples to balance against ~15k unsafe examples.
