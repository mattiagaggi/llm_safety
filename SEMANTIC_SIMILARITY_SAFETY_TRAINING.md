# Semantic Similarity-Guided Safety Training for LLMs

## Overview

A novel fine-tuning approach that uses semantic embeddings (Sentence-BERT) to enforce consistent safety responses across semantically similar harmful prompts. The core idea: **if two prompts are semantically similar, the model should produce the same safe response**, enabling generalization beyond training examples.

## Core Method

### Step 1: Safety Probe Training
- Extract hidden states from LLM for safe/unsafe prompts
- Train linear probe: `f: hidden_state → {safe, unsafe}`
- Use this probe to identify unsafe intent during training

### Step 2: Semantic Similarity-Based Loss
For each harmful prompt `x_i`:
1. Get Sentence-BERT embedding: `e_i = SentenceBERT(x_i)`so t
2. Find semantically similar harmful prompts: `S_i = {x_j : cosine(e_i, e_j) > τ}`
3. Enforce consistency loss:
   ```
   L_consistency = Σ_{x_j ∈ S_i} ||f_LLM(x_i) - f_LLM(x_j)||²
   ```
   Where `f_LLM(x)` is the model's hidden state or output representation

### Step 3: Combined Training Objective
```
L_total = L_safety + λ * L_consistency + L_capability
```
- `L_safety`: Standard safety loss (refusal on harmful prompts)
- `L_consistency`: Semantic similarity consistency (proposed)
- `L_capability`: Preserve helpfulness on safe prompts

### Step 4: LoRA Adapter Training (Parameter-Efficient)

Instead of full fine-tuning, use **LoRA (Low-Rank Adaptation)** adapters:
- Add low-rank matrices to attention layers: `W' = W + BA` where `B ∈ R^{d×r}`, `A ∈ R^{r×d}`, `r << d`
- Only train adapter parameters (~0.1-1% of model parameters)
- Optimize embeddings to be similar for semantically similar prompts via consistency loss

## Key Innovation

**Semantic Regularization**: Instead of training on individual examples, enforce that **semantically similar harmful prompts produce the same safe response**. This:
- Generalizes to unseen harmful prompts (if semantically similar to training data)
- Reduces overfitting to specific phrasings
- Creates a more robust safety boundary in semantic space

**LoRA-Based Optimization**: Use parameter-efficient LoRA adapters to optimize embeddings/hidden states for similarity, making the approach:
- Computationally efficient (train only ~0.1-1% of parameters)
- Preserves base model capabilities (minimal interference)
- Enables multiple safety adapters (modular approach)

## Publication Assessment

### ✅ **Strengths**

1. **Novel Approach**:
   - Using semantic similarity to regularize safety training is under-explored
   - Different from standard fine-tuning (which treats each example independently)
   - Connects representation learning to safety alignment

2. **Clear Theoretical Motivation**:
   - If two prompts mean the same thing, model should respond the same way
   - Semantic similarity provides natural grouping for regularization
   - Addresses generalization gap in safety training

3. **Practical Value**:
   - Could improve robustness to paraphrased jailbreaks
   - Generalizes beyond training set
   - Works with any embedding model (Sentence-BERT, etc.)

4. **Natural Extension**:
   - Builds on existing work (safety fine-tuning, semantic embeddings)
   - Clear incremental contribution
   - Easy to understand and implement

### ⚠️ **Challenges**

1. **Novelty Concerns** (⚠️ **CRITICAL**):
   - **Similar existing work**:
     - "Semantic Loss Guided Data Efficient Supervised Fine Tuning for Safe Responses in LLMs" (2024) - Uses semantic similarity for safety
     - Contrastive learning for safety alignment (multiple papers)
     - Data augmentation via paraphrasing (standard practice)
     - Regularization techniques (well-established)
   - **What's been done**:
     - Semantic similarity for safety training ✅ (exists)
     - Contrastive learning for safety ✅ (exists)
     - Consistency regularization ✅ (exists)
   - **What might be novel**:
     - Using Sentence-BERT embeddings to define similarity groups (vs. paraphrasing or contrastive pairs)
     - Consistency loss on refusal probabilities (vs. hidden states or text)
     - LoRA + embedding MLP hybrid for safety
   - **Verdict**: **Incremental, not revolutionary** - Need strong empirical results or novel angle

2. **Implementation Details**:
   - How exactly to enforce "same output"? 
     - Hidden state similarity? ✅ Differentiable
     - Response text similarity? ❌ Not differentiable
     - Refusal probability similarity? ✅ Differentiable (see solutions below)
   - Need clear definition of what "same" means

3. **Empirical Validation**:
   - Must show improvement over:
     - Standard safety fine-tuning
     - Data augmentation (paraphrasing)
     - Contrastive learning baselines
   - Need large-scale evaluation

4. **Theoretical Foundation**:
   - Why should semantic similarity guarantee same response?
   - What if semantically similar prompts should have different responses?
   - Need analysis of when this assumption holds

### 📊 **Publication Potential** (REVISED - More Realistic)

**Top-Tier Conference**: ⚠️ **Unlikely without major novelty** (NeurIPS, ICLR, ICML)
- Similar work already exists (semantic similarity for safety)
- Need either:
  - **Strong empirical breakthrough** (e.g., 40%+ improvement)
  - **Novel theoretical contribution** (e.g., generalization bounds)
  - **New angle** (e.g., specific to a new problem setting)
- Current approach is **incremental**, not revolutionary

**Mid-Tier Conference**: ✅ **Possible** (ACL, EMNLP)
- If results are solid (20-30% improvement)
- Clear empirical contribution
- Good fit for safety/applications track
- **Best realistic target**

**Workshop**: ✅ **Very Likely** (NeurIPS/ICLR Safety Workshop)
- Incremental contributions are welcome
- Focus on practical value
- Good venue for safety work

**Best Path**: **Aim for mid-tier conference or workshop**. Top-tier requires stronger novelty.

## Feasibility Assessment

### ✅ **Likely to Work**

**Why it should work:**

1. **Semantic Similarity is Meaningful**:
   - Sentence-BERT embeddings capture semantic meaning well
   - Similar harmful prompts should trigger similar safety mechanisms
   - This is a reasonable inductive bias

2. **Regularization Effect**:
   - Enforcing consistency acts as regularization
   - Prevents overfitting to specific phrasings
   - Should improve generalization

3. **Empirical Precedent**:
   - Contrastive learning works (enforcing similarity)
   - Data augmentation via paraphrasing works
   - This combines both ideas for safety

### ⚠️ **Potential Issues**

1. **What Does "Same Output" Mean?**
   - **Option A**: Same hidden state representation
     - Pro: Direct, differentiable
     - Con: Hidden states might vary even for same semantic response
   - **Option B**: Same refusal probability
     - Pro: More interpretable
     - Con: Might be too restrictive
   - **Option C**: Same response text (via embedding similarity)
     - Pro: Most interpretable
     - Con: Harder to optimize, discrete outputs

2. **Semantic Similarity Threshold**:
   - Too high (τ): Few pairs, limited regularization
   - Too low (τ): Enforces similarity on dissimilar prompts (bad)
   - Need careful tuning or adaptive threshold

3. **Edge Cases**:
   - What if semantically similar prompts should have different responses?
     - E.g., "How to make a bomb?" vs "How to make a bomb safely?" (both harmful, but second might need different response)
   - Need to handle these cases

4. **Computational Cost**:
   - Finding similar prompts for each example: O(n²) similarity computations
   - Need efficient approximate nearest neighbor search (FAISS, etc.)

### 🎯 **Recommendation**

**Likelihood of Working: 75-85%**

- **High confidence** that semantic regularization will help
- **Medium-high confidence** that it improves over standard fine-tuning
- **Lower confidence** on exact implementation details (need experimentation)

**Best Approach**:

1. **Start with Refusal Classifier on Token Embeddings (Recommended)**: 
   - Train a small MLP on token sequence embeddings to predict refusal
   - Use token probabilities to compute expected embeddings (fully differentiable)
   - Enforce consistency: `L_consistency = (refusal_prob_i - refusal_prob_j)²`
   - ✅ Fully differentiable, more accurate (considers full sequence), interpretable

2. **Alternative: Soft Token Matching**:
   - Use soft matching of refusal patterns via token probabilities
   - No training needed, just pattern matching
   - ✅ Fully differentiable, interpretable, efficient

2. **Alternative: Refusal Probe**:
   - Train linear probe: `refusal_prob = sigmoid(probe(hidden_state))`
   - Use probe output for consistency loss
   - ✅ More flexible, can use any layer

3. **Progressive Refinement**:
   - Try different similarity thresholds (τ = 0.7-0.9)
   - Experiment with different layers (early vs. late)
   - Try multiple refusal tokens vs. single token
   - Add weighting (more weight to very similar pairs)

4. **Ablation Studies**:
   - Without consistency loss (baseline)
   - With consistency loss on refusal probs (proposed)
   - With hidden state consistency (alternative)
   - With data augmentation instead (alternative)
   - Combined (consistency + augmentation)

## Implementation Details

### Architecture

#### **Option 1: Refusal Classifier on Token Embeddings (Best - Fully Differentiable)**

```python
# 1. Train a refusal classifier on token sequences
# Pre-train this on (token_sequence, refusal_label) pairs
refusal_classifier = nn.Sequential(
    nn.Linear(vocab_size * embed_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# 2. Get token probabilities (not just first token, but full sequence)
outputs = model(input_ids, output_hidden_states=True)
logits = outputs.logits  # [batch, seq_len, vocab_size]

# 3. Get token embeddings from model
token_embeddings = model.get_input_embeddings()  # Embedding layer
# Or use: token_embeddings = model.embed_tokens

# 4. Compute weighted token embeddings using probabilities
probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
# For each position, get expected embedding
expected_embeddings = torch.einsum('bsv,ve->bse', probs, token_embeddings.weight)
# Shape: [batch, seq_len, embed_dim]

# 5. Pool sequence (mean or use CLS token)
pooled_embedding = expected_embeddings.mean(dim=1)  # [batch, embed_dim]
# Or: pooled_embedding = expected_embeddings[:, 0]  # First token

# 6. Predict refusal probability (DIFFERENTIABLE!)
refusal_probs = refusal_classifier(pooled_embedding.flatten(1))  # [batch, 1]

# 7. Consistency loss
consistency_loss = compute_consistency_loss(refusal_probs, similarity_matrix)
```

**Advantages:**
- ✅ Fully differentiable (uses token probabilities + embeddings)
- ✅ More accurate (considers full sequence, not just first token)
- ✅ Can learn complex refusal patterns
- ✅ Interpretable (refusal probability)

#### **Option 1b: Simpler - Soft Token Matching (Also Differentiable)**

```python
# 1. Define refusal token patterns (as embeddings)
refusal_patterns = [
    tokenizer.encode("I can't", return_tensors='pt')[0],
    tokenizer.encode("I cannot", return_tensors='pt')[0],
    tokenizer.encode("Sorry", return_tensors='pt')[0],
    # ... more patterns
]

# 2. Get token probabilities
outputs = model(input_ids)
logits = outputs.logits  # [batch, seq_len, vocab_size]
probs = torch.softmax(logits, dim=-1)

# 3. Compute soft matching to refusal patterns (differentiable!)
refusal_scores = []
for pattern in refusal_patterns:
    # For each position in sequence, compute probability of pattern starting there
    pattern_probs = []
    for pos in range(seq_len - len(pattern) + 1):
        # Probability that pattern appears at this position
        pos_probs = [probs[:, pos+i, pattern[i]] for i in range(len(pattern))]
        pattern_prob = torch.stack(pos_probs).prod(dim=0)  # [batch]
        pattern_probs.append(pattern_prob)
    refusal_scores.append(torch.stack(pattern_probs).max(dim=0)[0])  # Max over positions

# 4. Refusal probability = max over all patterns
refusal_probs = torch.stack(refusal_scores).max(dim=0)[0]  # [batch]

# 5. Consistency loss
consistency_loss = compute_consistency_loss(refusal_probs, similarity_matrix)
```

**Advantages:**
- ✅ Fully differentiable (soft matching via probabilities)
- ✅ No need to train classifier
- ✅ Interpretable (matches known refusal patterns)
- ✅ Efficient

#### **Option 1c: First Token Logits (Simplest - Fully Differentiable)**

```python
# 1. Define refusal token IDs
refusal_token_ids = [tokenizer.encode("I")[0], 
                     tokenizer.encode("can't")[0], 
                     tokenizer.encode("Sorry")[0], ...]

# 2. Get logits for first token (critical for refusal)
outputs = model(input_ids)
logits = outputs.logits[:, -1, :]  # [batch, vocab_size] - first generation token

# 3. Compute refusal probability (differentiable!)
refusal_logits = logits[:, refusal_token_ids]  # [batch, num_refusal_tokens]
refusal_probs = torch.softmax(refusal_logits, dim=-1).sum(dim=-1)  # [batch]

# 4. Consistency loss
consistency_loss = compute_consistency_loss(refusal_probs, similarity_matrix)
```

**Advantages:**
- ✅ Fully differentiable (uses logits directly)
- ✅ Interpretable (refusal probability)
- ✅ Efficient (no decoding needed)
- ⚠️ Only considers first token (may miss multi-token refusals)

## Deterministic Methods for Refusal Detection

All methods below are **deterministic** (same input → same output) and **differentiable**:

### **Method 1: Threshold-Based (Simplest)**

```python
# Deterministic: If refusal token probability > threshold → refusal
refusal_token_ids = [tokenizer.encode("I")[0], tokenizer.encode("can't")[0], ...]
logits = model(input_ids).logits[:, -1, :]  # First token logits
refusal_probs = torch.softmax(logits[:, refusal_token_ids], dim=-1).sum(dim=-1)

# Deterministic threshold (no randomness)
is_refusal = (refusal_probs > 0.5).float()  # Hard threshold
# Or soft (differentiable):
refusal_score = torch.sigmoid((refusal_probs - 0.5) * 10)  # Smooth step function
```

**Properties:**
- ✅ Deterministic: same logits → same refusal score
- ✅ Differentiable: uses softmax/sigmoid
- ✅ No training needed
- ⚠️ Simple (may miss complex refusals)

### **Method 2: Trained Classifier (Most Accurate)**

```python
# Train once, then deterministic
refusal_classifier = train_classifier_on_refusal_data()  # One-time training

# At inference/training time:
logits = model(input_ids).logits
probs = torch.softmax(logits, dim=-1)
expected_embeddings = compute_expected_embeddings(probs, token_embeddings)
refusal_prob = refusal_classifier(expected_embeddings)  # Deterministic output
```

**Properties:**
- ✅ Deterministic: same input → same output (once classifier is fixed)
- ✅ Differentiable: classifier is neural network
- ✅ More accurate: learns complex patterns
- ⚠️ Requires training data

### **Method 3: Rule-Based on Logits (Deterministic Rules)**

```python
# Define deterministic rules
def compute_refusal_deterministic(logits, refusal_tokens, thresholds):
    """
    Deterministic rules based on logit values.
    Same logits → same refusal score (no randomness).
    """
    first_token_logits = logits[:, -1, :]
    
    # Rule 1: Check if any refusal token is in top-k
    top_k_logits, top_k_indices = torch.topk(first_token_logits, k=5, dim=-1)
    has_refusal_in_topk = (top_k_indices.unsqueeze(-1) == refusal_tokens).any(dim=-1).any(dim=-1)
    
    # Rule 2: Sum of refusal token probabilities
    refusal_probs = torch.softmax(first_token_logits[:, refusal_tokens], dim=-1).sum(dim=-1)
    
    # Rule 3: Combine rules deterministically
    refusal_score = (has_refusal_in_topk.float() * 0.5 + 
                     torch.sigmoid((refusal_probs - 0.3) * 10) * 0.5)
    
    return refusal_score  # Deterministic: same logits → same score
```

**Properties:**
- ✅ Fully deterministic: hard-coded rules
- ✅ Differentiable: uses differentiable operations
- ✅ Interpretable: clear rules
- ⚠️ May need tuning

### **Method 4: Soft Pattern Matching (Deterministic Pattern Matching)**

```python
# Deterministic pattern matching (no randomness)
def compute_refusal_pattern_matching(logits, refusal_patterns):
    """
    Deterministic: same logits → same pattern match scores.
    """
    probs = torch.softmax(logits, dim=-1)
    refusal_scores = []
    
    for pattern in refusal_patterns:
        # For each position, compute pattern probability (deterministic)
        pattern_probs = []
        for pos in range(seq_len - len(pattern) + 1):
            pos_prob = torch.prod([probs[:, pos+i, pattern[i]] 
                                  for i in range(len(pattern))], dim=0)
            pattern_probs.append(pos_prob)
        # Max over positions (deterministic)
        refusal_scores.append(torch.stack(pattern_probs).max(dim=0)[0])
    
    # Max over patterns (deterministic)
    return torch.stack(refusal_scores).max(dim=0)[0]
```

**Properties:**
- ✅ Deterministic: pattern matching is deterministic
- ✅ Differentiable: uses probabilities (soft matching)
- ✅ No training needed
- ✅ Interpretable

### **Comparison**

| Method | Deterministic | Differentiable | Accuracy | Training Needed |
|--------|--------------|----------------|----------|-----------------|
| Threshold-based | ✅ | ✅ | Medium | ❌ |
| Trained classifier | ✅ | ✅ | High | ✅ |
| Rule-based | ✅ | ✅ | Medium | ❌ |
| Pattern matching | ✅ | ✅ | Medium-High | ❌ |

**Recommendation**: Use **trained classifier** for best accuracy, or **pattern matching** for no-training solution.

#### **Option 2: Refusal Probe (Also Differentiable)**

```python
# 1. Train refusal probe on hidden states
probe = nn.Linear(hidden_dim, 1)  # Outputs refusal probability
# Train probe separately first on (hidden_state, refusal_label) pairs

# 2. Get hidden states
outputs = model(input_ids, output_hidden_states=True)
h = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]
h_pooled = h.mean(dim=1)  # [batch, hidden_dim]

# 3. Get refusal probabilities (differentiable!)
refusal_probs = torch.sigmoid(probe(h_pooled)).squeeze()  # [batch]

# 4. Consistency loss (same as Option 1)
consistency_loss = compute_consistency_loss(refusal_probs, similarity_matrix)
```

**Advantages:**
- ✅ Differentiable through probe
- ✅ Can use any layer
- ✅ More flexible (learned representation)

#### **Option 3: Hidden State Similarity (Simplest)**

```python
# Direct hidden state consistency (no refusal probability)
outputs = model(input_ids, output_hidden_states=True)
h_i = outputs.hidden_states[layer][i].mean(dim=1)  # [hidden_dim]
h_j = outputs.hidden_states[layer][j].mean(dim=1)  # [hidden_dim]
consistency_loss += ||h_i - h_j||²
```

**Advantages:**
- ✅ Simplest implementation
- ✅ Fully differentiable
- ⚠️ Less interpretable (doesn't directly optimize refusal)

### Key Hyperparameters

- **Similarity threshold τ**: Start with 0.7-0.8 (cosine similarity)
- **Regularization weight λ**: Start with 0.1-1.0
- **Layer choice**: Mid-layers (12-24) often best for safety
- **Batch construction**: Ensure each batch has semantically similar pairs

## Experiments Needed

### 1. **Main Results**
- Compare to standard safety fine-tuning baseline
- Evaluate on:
  - Training set (should maintain performance)
  - Held-out harmful prompts (should improve)
  - Paraphrased jailbreaks (should improve significantly)
  - Safe prompts (should maintain helpfulness)

### 2. **Ablation Studies**
- Effect of similarity threshold τ
- Effect of regularization weight λ
- Which layer to use for consistency
- Effect of probe vs. no probe

### 3. **Baseline Comparisons**

#### **Essential Baselines (Must Include):**

1. **Standard LoRA Fine-Tuning on Safety Data** (Primary Baseline)
   ```python
   # Standard approach: Train LoRA on safety dataset
   lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
   model = get_peft_model(model, lora_config)
   # Train on: (harmful_prompt, refusal_response) pairs
   loss = CrossEntropyLoss(model_output, refusal_target)
   ```
   - **What it is**: Standard LoRA fine-tuning on safety datasets (BeaverTails, HarmBench)
   - **Why include**: Most common baseline, parameter-efficient
   - **Expected**: Good on training set, may overfit to specific phrasings

2. **Full Fine-Tuning on Safety Data**
   ```python
   # Fine-tune all parameters on safety data
   loss = CrossEntropyLoss(model_output, refusal_target)
   ```
   - **What it is**: Full model fine-tuning (not parameter-efficient)
   - **Why include**: Upper bound on performance (more parameters)
   - **Expected**: Best on training set, but may overfit more

3. **DPO (Direct Preference Optimization) for Safety**
   ```python
   # DPO loss: prefers safe responses over harmful
   loss = -log(σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x) 
                     - log π_θ(y_l|x) + log π_ref(y_l|x))))
   ```
   - **What it is**: Preference-based optimization (chosen vs. rejected responses)
   - **Why include**: Standard alignment method (widely used baseline)
   - **Expected**: Good generalization, but doesn't use semantic similarity
   - **Note**: Not state-of-the-art, but standard baseline
   
   **How DPO Works (Detailed Explanation Below)**

3b. **SafeDPO (Enhanced DPO for Safety)** ⭐ **More Recent**
   ```python
   # SafeDPO: DPO with explicit safety constraints
   loss = DPO_loss + λ * safety_constraint_loss
   ```
   - **What it is**: DPO enhanced with explicit safety alignment objectives
   - **Why include**: More recent, safety-specific version of DPO
   - **Expected**: Better safety than standard DPO
   - **Status**: More state-of-the-art than standard DPO

4. **SaLoRA (Safety-Alignment Preserved LoRA)**
   ```python
   # LoRA with safety module preservation
   loss = task_loss + λ * safety_preservation_loss
   ```
   - **What it is**: LoRA that preserves safety alignment (from paper)
   - **Why include**: Recent work on safety-preserving LoRA
   - **Expected**: Good safety preservation, but may not improve generalization

#### **Additional Baselines (Should Include):**

5. **Data Augmentation (Paraphrasing)**
   ```python
   # Augment training data with paraphrases
   augmented_data = [paraphrase(p) for p in harmful_prompts]
   # Then standard fine-tuning
   ```
   - **What it is**: Train on paraphrased harmful prompts
   - **Why include**: Similar goal (generalization), different method
   - **Expected**: Better generalization than standard, but may not be as systematic

6. **Contrastive Learning for Safety**
   ```python
   # Contrastive loss: pull similar prompts together
   loss = -log(exp(sim(h_i, h_j)/τ) / Σ_k exp(sim(h_i, h_k)/τ))
   ```
   - **What it is**: Contrastive learning on safety data
   - **Why include**: Similar to your approach (enforces similarity)
   - **Expected**: Good similarity learning, but may need negative sampling

7. **AlignGuard-LoRA** (If available)
   ```python
   # LoRA with Fisher Information Matrix regularization
   loss = task_loss + λ * FIM_regularization
   ```
   - **What it is**: LoRA that preserves alignment via FIM
   - **Why include**: Recent safety-preserving method
   - **Expected**: Good alignment preservation

#### **Ablation Baselines (For Your Method):**

8. **Your Method WITHOUT Semantic Similarity**
   ```python
   # Same LoRA + embedding MLP, but no consistency loss
   loss = L_safety + L_capability  # No L_consistency
   ```
   - **What it is**: Your method without the semantic similarity component
   - **Why include**: Ablation to show semantic similarity helps

9. **Your Method WITHOUT LoRA (Full Fine-Tuning)**
   ```python
   # Your method but fine-tune all parameters
   ```
   - **What it is**: Your method with full fine-tuning instead of LoRA
   - **Why include**: Ablation to show LoRA is sufficient

10. **Your Method WITHOUT Embedding MLP**
    ```python
    # Your method but only LoRA, no embedding transformation
    ```
    - **What it is**: Your method without embedding MLP
    - **Why include**: Ablation to show embedding MLP helps

#### **Summary Table of Baselines:**

| Baseline | Method | Parameters | Key Feature | Status |
|----------|--------|------------|-------------|--------|
| Standard LoRA | LoRA on safety data | ~0.1-1% | Standard approach | Basic baseline |
| Full Fine-Tuning | All params on safety | 100% | Upper bound | Basic baseline |
| DPO | Preference optimization | 100% | Standard alignment | Common baseline |
| **SafeDPO** | **Enhanced DPO for safety** | **100%** | **Safety-specific DPO** | **More SOTA** |
| SaLoRA | Safety-preserving LoRA | ~0.1-1% | Preserves safety | Recent work |
| AsFT | Anchoring safety | ~0.1-1% | Safety regularization | Recent work |
| Data Augmentation | Paraphrased training | ~0.1-1% | Generalization via data | Standard |
| Contrastive Learning | Contrastive loss | ~0.1-1% | Similarity learning | Standard |
| **Your Method** | **LoRA + Embedding MLP + Semantic Consistency** | **~1-2%** | **Semantic regularization** | **Proposed** |

#### **Evaluation Metrics for Baselines:**

For each baseline, measure:
- **Safety**: Refusal rate on harmful prompts (should be high)
- **Generalization**: Refusal rate on paraphrased/held-out harmful prompts (your method should excel)
- **Helpfulness**: Performance on safe prompts (should maintain)
- **Efficiency**: Training time, memory usage, parameter count

### 4. **Analysis**
- Visualize semantic clusters
- Show examples of improved generalization
- Analyze failure cases

## Alternative Approaches: LoRA vs. Other Methods

### **Option 1: LoRA Adapters (Current Proposal)**

**How it works:**
- Add low-rank matrices to attention layers: `W' = W + BA` where `r << d`
- Train only adapter parameters (~0.1-1% of model)
- Optimize via consistency loss: `L_consistency = ||h_i - h_j||²` for similar prompts

**Pros:**
- ✅ Parameter-efficient (train only ~4M params for 7B model)
- ✅ Preserves base model (can disable adapters)
- ✅ Fast training (fewer gradients)
- ✅ Modular (can stack multiple adapters)

**Cons:**
- ⚠️ Limited expressiveness (low-rank constraint)
- ⚠️ May not modify embeddings directly (only attention layers)
- ⚠️ Need to choose which layers to adapt

**Verdict**: **Good for efficiency, but may not directly optimize embeddings**

---

### **Option 2: Direct Embedding Layer Fine-Tuning (Better for Embeddings)**

**How it works:**
- Fine-tune only the embedding layer (token embeddings)
- Add small MLP on top: `h' = MLP(embedding(x))` to map to similar space
- Optimize: `L = ||MLP(e_i) - MLP(e_j)||²` for similar prompts

**Pros:**
- ✅ Directly optimizes embeddings (what you want!)
- ✅ Very parameter-efficient (only embedding layer + small MLP)
- ✅ Interpretable (embeddings directly modified)
- ✅ Fast (smaller layer)

**Cons:**
- ⚠️ May need to also tune early layers for best results
- ⚠️ Embedding layer is large (but still < 1% of model)

**Verdict**: **Best if goal is to optimize embeddings directly**

---

### **Option 3: Contrastive Learning (Most Theoretically Sound)**

**How it works:**
- Use contrastive loss: `L = -log(exp(sim(h_i, h_j)/τ) / Σ_k exp(sim(h_i, h_k)/τ))`
- Pull similar prompts together, push dissimilar apart
- Train with LoRA or full fine-tuning

**Pros:**
- ✅ Theoretically well-founded (contrastive learning)
- ✅ Explicitly optimizes similarity
- ✅ Works well in practice (SimCSE, etc.)
- ✅ Can combine with LoRA

**Cons:**
- ⚠️ Need negative examples (dissimilar prompts)
- ⚠️ More complex loss function
- ⚠️ Need to sample negatives carefully

**Verdict**: **Best theoretical foundation, but more complex**

---

### **Option 4: Metric Learning (Learns Distance Function)**

**How it works:**
- Learn a metric: `d(x_i, x_j) = ||M·h_i - M·h_j||²` where `M` is learned matrix
- Optimize: `L = d(similar_pairs) - d(dissimilar_pairs)`
- Use LoRA or small MLP to learn `M`

**Pros:**
- ✅ Learns optimal distance metric
- ✅ Can be combined with LoRA
- ✅ Theoretically sound

**Cons:**
- ⚠️ More complex than simple consistency loss
- ⚠️ Need negative pairs

**Verdict**: **Good but may be overkill**

---

### **Option 5: Hybrid: Embedding MLP + LoRA (Recommended)**

**How it works:**
1. Add small MLP on embeddings: `h' = MLP(embedding(x))`
2. Add LoRA to attention layers for fine-grained control
3. Optimize: `L = ||MLP(e_i) + LoRA(h_i) - MLP(e_j) - LoRA(h_j)||²`

**Pros:**
- ✅ Best of both worlds (embedding + attention)
- ✅ Still parameter-efficient
- ✅ Directly optimizes embeddings + hidden states
- ✅ More expressive than LoRA alone

**Cons:**
- ⚠️ Slightly more parameters than LoRA alone
- ⚠️ More hyperparameters to tune

**Verdict**: **Best overall approach - combines embedding optimization with attention tuning**

---

## Recommendation

**Best Approach: Embedding MLP + LoRA (Hybrid)**

```python
# 1. Embedding transformation (directly optimizes embeddings)
embedding_mlp = nn.Sequential(
    nn.Linear(embed_dim, embed_dim),
    nn.LayerNorm(embed_dim),
    nn.ReLU(),
    nn.Linear(embed_dim, embed_dim)
)

# 2. LoRA on attention layers (fine-grained control)
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # attention layers
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# 3. Forward pass
embeddings = model.get_input_embeddings()(input_ids)
transformed_embeddings = embedding_mlp(embeddings)
# Then pass through model with LoRA

# 4. Consistency loss
h_i = model(input_ids_i, inputs_embeds=transformed_embeddings_i)
h_j = model(input_ids_j, inputs_embeds=transformed_embeddings_j)
consistency_loss = ||h_i - h_j||²
```

**Why this is best:**
- Directly optimizes embeddings (what you want)
- LoRA fine-tunes attention for safety
- Still parameter-efficient (~1-2% of model)
- More expressive than LoRA alone
- Clear separation of concerns (embeddings vs. attention)

## Potential Extensions

1. **Hierarchical Similarity**: Use multiple similarity thresholds for different levels of regularization
2. **Adaptive Thresholds**: Learn similarity thresholds per prompt type
3. **Multi-Modal**: Extend to image+text prompts
4. **Theoretical Analysis**: Prove generalization bounds
5. **Contrastive Learning**: Combine with contrastive loss for better similarity learning

## Understanding Preference-Based Optimization (DPO)

### What is DPO?

**Direct Preference Optimization (DPO)** is a method to align LLMs with human preferences without training a separate reward model. It directly optimizes the model to prefer "good" responses over "bad" ones.

### How DPO Works for Safety

#### **1. Data Format**

Instead of single responses, you need **pairs of responses**:
- **Chosen response** (`y_w`): The safe/helpful response (what we want)
- **Rejected response** (`y_l`): The harmful/unhelpful response (what we don't want)

**Example:**
```
Prompt: "How to make a bomb?"
Chosen (y_w): "I can't help with that request. Making explosives is illegal and dangerous."
Rejected (y_l): "Here's how to make a bomb: [harmful instructions]"
```

#### **2. The DPO Loss Function**

```python
# DPO loss function
def dpo_loss(model, ref_model, prompt, chosen_response, rejected_response, beta=0.1):
    """
    model: Model being trained
    ref_model: Reference model (frozen, usually the base model)
    beta: Temperature parameter (controls strength of preference)
    """
    # Get log probabilities from both models
    log_pi_chosen = model.get_log_prob(prompt, chosen_response)
    log_pi_rejected = model.get_log_prob(prompt, rejected_response)
    log_pi_ref_chosen = ref_model.get_log_prob(prompt, chosen_response)
    log_pi_ref_rejected = ref_model.get_log_prob(prompt, rejected_response)
    
    # DPO loss: maximize preference for chosen, minimize for rejected
    # Relative to reference model (prevents model from deviating too much)
    loss = -log(sigmoid(
        beta * (
            (log_pi_chosen - log_pi_ref_chosen) - 
            (log_pi_rejected - log_pi_ref_rejected)
        )
    ))
    
    return loss
```

**Mathematical Formulation:**
$$L_{DPO} = -\log \sigma\left(\beta \left(\log \pi_\theta(y_w|x) - \log \pi_{ref}(y_w|x) - \log \pi_\theta(y_l|x) + \log \pi_{ref}(y_l|x)\right)\right)$$

Where:
- $\pi_\theta$: Model being trained
- $\pi_{ref}$: Reference model (frozen, prevents drift)
- $y_w$: Chosen (good) response
- $y_l$: Rejected (bad) response
- $\beta$: Temperature (higher = stronger preference)

#### **3. Intuition**

**What DPO does:**
1. **Increases probability** of chosen response (safe refusal)
2. **Decreases probability** of rejected response (harmful content)
3. **Relative to reference model** (prevents model from changing too much)

**Key insight**: Instead of training a reward model separately (like RLHF), DPO directly optimizes the model to prefer good responses.

#### **4. Comparison to Standard Fine-Tuning**

| Aspect | Standard Fine-Tuning | DPO |
|--------|---------------------|-----|
| **Data** | Single responses | Pairs (chosen vs. rejected) |
| **Loss** | Cross-entropy on target | Preference ranking |
| **Goal** | Match target response | Prefer good over bad |
| **Reference** | None | Uses frozen reference model |

**Example for Safety:**

**Standard Fine-Tuning:**
```python
# Train model to output refusal
loss = CrossEntropyLoss(model_output, "I can't help with that")
```

**DPO:**
```python
# Train model to prefer refusal over harmful response
loss = DPO_loss(
    chosen="I can't help with that",
    rejected="Here's how to make a bomb: ..."
)
```

#### **5. Advantages of DPO for Safety**

✅ **Explicit preference learning**: Model learns to prefer safe responses
✅ **No reward model needed**: Simpler than RLHF
✅ **Reference model prevents drift**: Model doesn't deviate too far from base
✅ **Works well in practice**: Standard alignment method

**Note**: DPO is **not** state-of-the-art for safety. More recent methods include:
- **SafeDPO**: DPO enhanced with explicit safety constraints
- **AsFT**: Anchoring safety during fine-tuning
- **Magic-Token-Guided Co-Training**: Multi-behavior safety control

#### **6. Disadvantages**

⚠️ **Needs paired data**: Requires (chosen, rejected) pairs (can be expensive)
⚠️ **Doesn't use semantic similarity**: Treats each example independently
⚠️ **May not generalize**: Like standard fine-tuning, may overfit to specific phrasings

#### **7. Why It's a Good Baseline**

- **State-of-the-art**: Currently one of the best alignment methods
- **Standard practice**: Widely used in safety research
- **Fair comparison**: Shows your method's advantage (semantic similarity)

### How to Implement DPO Baseline

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

# 1. Load base model and create reference (frozen)
base_model = AutoModelForCausalLM.from_pretrained("llama-3-8b")
ref_model = AutoModelForCausalLM.from_pretrained("llama-3-8b")
ref_model.eval()  # Freeze reference model
for param in ref_model.parameters():
    param.requires_grad = False

# 2. Add LoRA to trainable model
lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# 3. DPO training loop
for prompt, chosen, rejected in safety_dataset:
    # Get log probabilities
    log_pi_chosen = model.get_log_prob(prompt, chosen)
    log_pi_rejected = model.get_log_prob(prompt, rejected)
    log_pi_ref_chosen = ref_model.get_log_prob(prompt, chosen)
    log_pi_ref_rejected = ref_model.get_log_prob(prompt, rejected)
    
    # DPO loss
    loss = -torch.log(torch.sigmoid(
        beta * (
            (log_pi_chosen - log_pi_ref_chosen) - 
            (log_pi_rejected - log_pi_ref_rejected)
        )
    ))
    
    loss.backward()
    optimizer.step()
```

### Key Difference from Your Method

**DPO**: Learns to prefer good responses, but treats each example independently
**Your Method**: Enforces that semantically similar prompts produce similar responses (generalization)

**Your advantage**: Should generalize better to unseen harmful prompts that are semantically similar to training data.

## Related Work to Cite

- Safety fine-tuning (BeaverTails, HarmBench)
- DPO: Direct Preference Optimization (Rafailov et al., 2023)
- **SafeDPO**: Enhanced DPO for safety (2025) - More state-of-the-art
- **AsFT**: Anchoring Safety During Fine-Tuning (2025)
- SaLoRA: Safety-Alignment Preserved LoRA
- Contrastive learning (SimCSE, etc.)
- Semantic embeddings (Sentence-BERT, etc.)
- Data augmentation for safety
- Regularization techniques

## Current State-of-the-Art (2025)

**For Safety Fine-Tuning:**
1. **SafeDPO** - DPO with explicit safety constraints
2. **AsFT** - Anchoring safety during fine-tuning
3. **Magic-Token-Guided Co-Training** - Multi-behavior safety control
4. **Safe RLHF-V** - For multimodal models

**Your method should compare against:**
- **Primary**: Standard LoRA fine-tuning (most common baseline)
- **Strong baseline**: SafeDPO (more recent, safety-specific)
- **Additional**: DPO, AsFT, SaLoRA (if available)

**Note**: DPO alone is **not** state-of-the-art - it's a standard baseline. SafeDPO is more recent and safety-specific.

## Conclusion

**Verdict**: **Incremental contribution, moderate publication potential**

- ⚠️ **Not highly novel** - Similar work exists (semantic similarity for safety)
- ✅ **Still valuable** - If results are strong (20-30% improvement)
- ✅ **Practical** - Better generalization is useful
- ✅ **75-85% chance of working** empirically

**Honest Assessment**:
- This is **incremental research**, not groundbreaking
- Similar ideas have been explored (semantic similarity, contrastive learning)
- **Best for**: Mid-tier conference (ACL, EMNLP) or workshop
- **Not suitable for**: Top-tier without major novelty/breakthrough

**To Increase Novelty, Consider**:
1. **Novel problem setting**: E.g., few-shot safety, cross-lingual safety
2. **Theoretical contribution**: Prove generalization bounds or convergence
3. **Novel combination**: E.g., semantic similarity + adversarial training + something new
4. **Strong empirical results**: 40%+ improvement would be compelling
5. **New angle**: Focus on a specific aspect (e.g., refusal probability consistency)

**Next Steps**:
1. **Literature review**: Thoroughly review existing semantic similarity safety work
2. **Identify gap**: What exactly is new here?
3. **Implement and validate**: See if results are strong enough
4. **Target appropriate venue**: Mid-tier or workshop (not top-tier unless breakthrough)

**Key Selling Points (Revised)**:
- "Systematic study of semantic similarity regularization for safety" (not "first")
- "Improves generalization to unseen harmful prompts by 20-30%"
- "Parameter-efficient approach using LoRA + embedding MLP"
- "Empirical analysis of when semantic similarity assumption holds"
