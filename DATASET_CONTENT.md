# Dataset Content Summary

This document details what each dataset contains beyond just text/conversations.

## 1. **BeaverTails** (Anthropic/hh-rlhf)
**Content Type:** Conversation pairs (helpful vs harmful responses)

**Fields:**
- `chosen` - Helpful/safe response text
- `rejected` - Harmful/unsafe response text
- (Note: The dataset contains pairs of responses to the same prompt)

**Metadata Extracted:**
- Source identifier
- Split (train/test)

---

## 2. **HarmfulQA** (declare-lab/HarmfulQA)
**Content Type:** Harmful questions with topic categorization

**Fields:**
- `question` - The harmful question text
- `topic` - Main topic category (self_harm, violence, drugs, cybercrime, etc.)
- `subtopic` - More specific subcategory

**Metadata Extracted:**
- Topic
- Subtopic
- Source identifier

---

## 3. **ToxiGen** (toxigen/toxigen-data)
**Content Type:** Machine-generated toxic/benign statements

**Fields:**
- `text` - The statement text
- `toxicity_ai` - AI-generated toxicity score (0-1)
- `toxicity_human` - Human-annotated toxicity score (0-1)
- `target_group` - Minority group targeted
- `stereotyping` - Stereotyping annotation
- `intent` - Intent annotation

**Metadata Extracted:**
- Toxicity scores (both AI and human)
- Target group
- Stereotyping information
- Intent information

---

## 4. **Measuring Hate Speech** (ucberkeley-dlab/measuring-hate-speech)
**Content Type:** Social media posts/comments with hate speech scores

**Fields:**
- `text` - Post/comment text
- `hate_speech_score` - Hate speech score (0-1)
- `violence` - Violence score (0-1)
- `respect` - Respect score (0-1)

**Metadata Extracted:**
- Hate speech score
- Violence score
- Respect score

---

## 5. **Alpaca** (tatsu-lab/alpaca)
**Content Type:** Instruction-following examples (safe/helpful)

**Fields:**
- `instruction` - The instruction/prompt
- `output` - The response/output
- `text` - Combined instruction + output (if available)

**Metadata Extracted:**
- Source identifier

---

## 6. **Red Team Attempts** (Anthropic/hh-rlhf, red-team-attempts)
**Content Type:** Red teaming conversation transcripts

**Fields:**
- `transcript` - Full conversation transcript
- `rating` - Success rating (1-5 scale, higher = more successful)
- `min_harmlessness_score_transcript` - Harmlessness score
- `task_description` - Description of the red team task
- `tags` - List of tags (e.g., self-harm, drugs, cybercrime)
- `model_type` - Type of model targeted
- `num_params` - Number of parameters in targeted model

**Metadata Extracted:**
- Rating
- Min harmlessness score
- Task description
- Tags
- Model type
- Number of parameters

---

## 7. **Mental Health Counseling** (Amod/mental_health_counseling_conversations)
**Content Type:** Mental health counseling conversations

**Fields:**
- `Context` - Conversation context
- `Response` - Counselor response
- (Combined into single text: context + response)

**Metadata Extracted:**
- Has self-harm keywords (boolean flag)

**Note:** Filtered by self-harm keywords: 'suicide', 'self-harm', 'hurt myself', 'kill myself', 'end my life', 'worthless'

---

## 8. **Self-Harm Synthetic Eval** (arianaazarbal/self-harm-synthetic-eval)
**Content Type:** Synthetic self-harm evaluation prompts

**Fields:**
- `instruction` - The prompt/instruction
- `label` - Safety label
- `category` - Category classification
- `prompt_type` - Type of prompt

**Metadata Extracted:**
- Label
- Category
- Prompt type

---

## 9. **Suicide Dataset** (jquiros/suicide)
**Content Type:** Suicide-related text examples

**Fields:**
- `text` - Text content
- `class` - Binary class label (0 = non-suicide, 1 = suicide)

**Metadata Extracted:**
- Class label

---

## 10. **Drug Reviews** (lewtun/drug-reviews)
**Content Type:** Drug review texts

**Fields:**
- `review` - Review text
- `drugName` - Name of the drug
- `condition` - Medical condition
- `rating` - Review rating

**Metadata Extracted:**
- Drug name
- Condition
- Rating

**Note:** Filtered for illegal substances: cocaine, heroin, meth, methamphetamine, crack, ecstasy, mdma, lsd, marijuana, cannabis

---

## 11. **SafetyBench** (thu-coai/SafetyBench)
**Content Type:** Multiple-choice safety evaluation questions

**Fields:**
- `question` - The safety question
- `options` - List of multiple-choice options
- `category` - Original category (Mental Health, Physical Health, Illegal Activities, Privacy and Property, Offensiveness, Ethics and Morality, Unfairness and Bias)
- `id` - Unique identifier

**Metadata Extracted:**
- ID
- Original category
- Config (test/dev)
- Split (en/zh/zh_subset)

**Note:** Questions are combined with options for full context

---

## Summary by Content Type

### Conversations/Dialogues:
- **BeaverTails** - Response pairs
- **Red Team Attempts** - Full transcripts
- **Mental Health Counseling** - Counseling conversations

### Questions/Prompts:
- **HarmfulQA** - Harmful questions
- **SafetyBench** - Multiple-choice safety questions
- **Self-Harm Synthetic** - Evaluation prompts

### Statements/Text:
- **ToxiGen** - Toxic/benign statements
- **Measuring Hate Speech** - Social media posts
- **Suicide Dataset** - Suicide-related text
- **Drug Reviews** - Drug review texts

### Instructions/Examples:
- **Alpaca** - Instruction-following examples (safe)

---

## Rich Metadata Datasets

Datasets with the most additional metadata:
1. **Red Team Attempts** - Ratings, scores, tags, model info, task descriptions
2. **ToxiGen** - Toxicity scores (AI + human), target groups, stereotyping, intent
3. **SafetyBench** - Categories, IDs, multiple-choice options
4. **Measuring Hate Speech** - Multiple scores (hate, violence, respect)
5. **Drug Reviews** - Drug names, conditions, ratings
