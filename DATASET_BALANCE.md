# Dataset Balance Analysis

## Summary

Most datasets are **NOT balanced** between toxic/non-toxic examples. Here's the breakdown:

## Individual Dataset Balance

| Dataset | Total | Unsafe/Toxic | Safe/Non-toxic | Balance Status |
|---------|-------|--------------|----------------|----------------|
| **BeaverTails** | 10,000 | 5,000 (50%) | 5,000 (50%) | ✅ **Balanced** |
| **HarmfulQA** | 1,000 | 1,000 (100%) | 0 (0%) | ❌ **All toxic** |
| **Red Team Attempts** | ~38k | ~38k (100%) | 0 (0%) | ❌ **All toxic** |
| **ToxiGen** | 8,960 | Varies by threshold | Varies | ⚠️ **Configurable** |
| **Measuring Hate Speech** | 135k | ~35% | ~65% | ✅ **Somewhat balanced** |
| **Alpaca** | 52k | 0 (0%) | 52k (100%) | ❌ **All safe** |

## Recommendations

### For Balanced Training:

1. **Use BeaverTails** - Already balanced 50/50
2. **Combine datasets** - Use `combine_and_balance()` utility:
   ```python
   from src.data import (
       load_harmfulqa,
       load_toxigen,
       load_alpaca,
       combine_and_balance
   )
   
   # Load toxic datasets
   toxic = [
       load_harmfulqa(max_examples=1000),
       load_toxigen(max_examples=1000, min_toxicity=0.5),
   ]
   
   # Load safe datasets
   safe = [
       load_alpaca(max_examples=2000),
   ]
   
   # Combine and balance to 50/50
   balanced = combine_and_balance(
       toxic + safe,
       target_ratio=0.5,
       max_examples_per_dataset=1000
   )
   ```

3. **Use Measuring Hate Speech** - Already has good balance (~35% toxic)

### For Category-Specific Analysis:

- **Self-harm**: Need to filter/combine from multiple sources
- **Violence**: Available in most toxic datasets
- **Drugs**: Need keyword filtering
- **Cybercrime**: Need keyword filtering

## Balance Utility Functions

See `src/data/balance_utils.py` for:
- `balance_dataset()` - Balance a single dataset
- `combine_and_balance()` - Combine multiple datasets and balance them

