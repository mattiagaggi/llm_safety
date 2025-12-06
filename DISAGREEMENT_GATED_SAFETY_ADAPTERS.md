# Disagreement-Gated Safety Adapters: Training Safety LoRA Modules Whose Internal Divergence Controls When They Activate

## Abstract

Large language models (LLMs) are increasingly aligned using parameter-efficient safety adapters such as LoRA modules. However, standard safety adapters are always active, which often improves refusal rates on harmful prompts but degrades model helpfulness and general capabilities due to over-refusal or global shifts in behavior.

We introduce Disagreement-Gated Safety Adapters (DGSA), a new architectural mechanism in which a safety adapter's own divergence from the frozen base model is used as a gating signal that determines when the adapter should influence next-token predictions. We train the adapter so that (1) on harmful or unsafe prompts it intentionally deviates from the base model to produce safe refusals, while (2) on benign prompts it stays close to the base model, enabling disagreement magnitude to function as a learned safety activation signal.

At inference time, DGSA performs token-level interpolation between base and adapter logits based on KL divergence.

Across multiple safety benchmarks, DGSA improves refusal rates on harmful prompts while reducing over-refusal by 20–40% relative to standard safety LoRA and preserving 95–100% of the base model's original capabilities, outperforming classifier-based gating without requiring any external routing model.

DGSA offers a simple, extensible, and interpretable mechanism for dynamic safety modulation in LLMs.

## 1 Introduction

Safety alignment of large language models (LLMs) is commonly performed using parameter-efficient finetuning (PEFT) techniques such as LoRA adapters. These adapters are typically trained to refuse harmful prompts and assist on benign ones.

However, a central limitation remains:

**Standard safety adapters are "always-on," modifying the model's behavior globally, even when no safety intervention is needed.**

This produces two well-known problems:

1. **Over-refusal on benign but safety-adjacent inputs** (e.g., educational cybersecurity, mental-health informational queries).
2. **Capability degradation**, where the model becomes less helpful or less accurate even in domains unrelated to safety.

We propose a different perspective: a safety adapter should only intervene when the base model is about to fail.

**Our key idea is to use the model's internal disagreement with itself as a signal for safety activation.**

## 2 Disagreement-Gated Safety Adapters

Let $M_{\text{base}}$ be a frozen LLM and $A$ a trainable LoRA adapter producing modified logits $p_{\text{safe}}$.

At each decoding step, we compute the KL divergence:

$$d_t = \text{KL}(p_{\text{safe}}(\cdot \mid x_{\leq t}) \parallel p_{\text{base}}(\cdot \mid x_{\leq t}))$$

We interpret $d_t$ as the adapter's confidence that a safety correction is needed.

### Token-level gated decoding

We combine logits using:

$$p_t = \alpha_t p_{\text{safe}} + (1 - \alpha_t) p_{\text{base}},$$

$$\alpha_t = \sigma(\lambda(d_t - \tau)),$$

where $\tau$ is a learned or fixed threshold.

- If disagreement is small → adapter stays silent.
- If disagreement is large → adapter dominates.
- Smoothly interpolated — no hard switching.

This yields dynamic, situation-aware safety behavior.

## 3 Training the Adapter to Make Disagreement Meaningful

DGSA relies on training the adapter so that disagreement correlates with harmfulness.

We use a labeled dataset of:

- **mistake examples**: model responded unsafely → we want strong deviation
- **correct examples**: base model behaved well → we want minimal deviation

### Loss for mistake examples

We push toward a safe refusal $y_{\text{safe}}$:

$$L_{\text{mistake}} = \text{CE}(y_{\text{safe}}, p_{\text{safe}}) - \lambda_{\text{div}} \text{KL}(p_{\text{safe}} \parallel p_{\text{base}})$$

### Loss for correct examples

We pull the adapter toward the base:

$$L_{\text{correct}} = \text{KL}(p_{\text{safe}} \parallel p_{\text{base}})$$

### Overall objective

$$L = \mathbb{E}_{\text{mistake}}[L_{\text{mistake}}] + \gamma \mathbb{E}_{\text{correct}}[L_{\text{correct}}] + \lambda_{\text{L2}} \|\theta_{\text{LoRA}}\|_2^2$$

This shapes the adapter into a module whose effect size equals safety requirement.

## 4 Related Work

We contrast DGSA with:

### Safety fine-tuning

Safety LoRAs and full safety finetuning improve refusal but globally distort outputs, causing over-refusal.

### Classifier-gated safety

External routers require additional models and operate only at prompt level; DGSA is intrinsic and token-level.

### Mixture-of-Experts

MoE gating is learned but unrelated to model disagreement and not tailored to safety.

### Hallucination detection via disagreement

Prior work observes disagreement but does not use it as a gating mechanism nor train adapters to shape disagreement meaning.

**DGSA introduces an architectural safety gate directly tied to base–adapter divergence, which is new.**

## 5 Experiments

### Models

- Llama-3-8B
- LoRA rank 16 for all methods

### Baselines

1. Base model
2. Standard safety LoRA (no gating, same data)
3. Classifier-gated safety
4. DGSA (ours)

### Datasets

**A. Harmful prompts**

HarmBench, HH-RLHF unsafe subset, synthetic ambiguous harm prompts.

**B. Benign but safety-adjacent prompts**

Educational cybersecurity, mental health info, benign biology, fiction.

**C. General capability**

MMLU subset, GSM8K-lite, TriviaQA.

**D. Jailbreak robustness**

Prompt-style attacks and multilingual paraphrases (judged safely).

### Metrics

- Unsafe harmful responses ↓
- Safety refusals ↑
- Over-refusal on benign prompts ↓
- General capability retention ↑
- KL separability (AUC) ↑

## 6 Results

### 6.1 Harmful Prompt Safety

DGSA matches or exceeds standard LoRA refusal rates.

### 6.2 Over-refusal

DGSA reduces over-refusal by 20–40% relative to standard LoRA.

### 6.3 Capability Retention

DGSA retains 95–100% of base model performance, while the standard LoRA loses 5–15%.

### 6.4 Disagreement as a Safety Signal

AUC for KL discriminating harmful vs benign prompts: 0.82–0.91.

## 7 Ablations

- No gating → high over-refusal.
- Gating without KL-to-base training → noisy decisions.
- Prompt-level vs token-level gating → token-level is best.
- Threshold analysis shows a smooth safety–helpfulness tradeoff.
