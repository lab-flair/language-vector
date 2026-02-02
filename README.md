# Language Steering for Multilingual In-Context Learning

Official implementation of "Language Steering for Multilingual In-Context Learning"

## Overview

This repository contains code for **language steering**, a training-free approach to improve multilingual in-context learning by leveraging activation differences between languages. The method:

1. **Computes language-specific steering vectors** from activation differences between source and target languages
2. **Applies steering during inference** by adding these vectors to intermediate activations
3. **Achieves consistent improvements** across diverse languages and tasks without any parameter updates

### Key Results

-  **Consistent improvements** across 19 languages on mathematical reasoning and NLI tasks
-  **Training-free**: No parameter updates or fine-tuning required
-  **Linguistically meaningful**: Steering vectors cluster by language families
-  **Task-transferable**: Vectors computed on one task can improve performance on others

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)


## Quick Start

### MGSM (Mathematical Reasoning)

Run steering experiments on the MGSM dataset:

```bash
python steer_mgsm.py
```

This will:
1. Load Llama-3.1-8B-Instruct model
2. Run experiments on 9 target languages (Chinese, Japanese, Thai, Swahili, Bengali, German, Spanish, French, Russian)
3. Save results to `steering_results/`

### Customize Configuration

Edit the configuration in `steer_mgsm.py`:

```python
config = SteeringConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",  # Model to use
    target_languages=["zh", "ja", "es"],             # Languages to test
    test_layers=[5, 10, 15, 20, 25, 30],             # Layers to try
    test_alphas=[0.5, 1.0, 2.0, 3.0],                # Scaling factors to try
    num_shots=6,                                      # Few-shot demonstrations
    seed=42                                           # Random seed
)
```

## Methodology

### 1. Steering Vector Computation

We compute language-specific steering vectors by extracting activation differences:

```
v(t) = mean(h_target(t) - h_source(t))
```

where:
- `h_target(t)`: Hidden states for target language examples at layer t
- `h_source(t)`: Hidden states for source language examples at layer t
- `v(t)`: Steering vector at layer t

### 2. Inference-Time Steering

During inference, we modify hidden states at specified positions:

```
h'(t) = h(t) + α · v(t)
```

where:
- `h(t)`: Original hidden state
- `α`: Scaling factor (hyperparameter)
- `v(t)`: Steering vector
- `h'(t)`: Steered hidden state

### 3. Three-Way Evaluation Split

We split the test set into three equal parts:
- **Compute**: For steering vector calculation (1/3)
- **Validation**: For hyperparameter selection (1/3)
- **Test**: For final evaluation (1/3)

This ensures no data leakage and fair hyperparameter selection.


## Datasets

### Supported Datasets

1. **MGSM** (Multilingual Grade School Math)
   - Task: Mathematical reasoning
   - Languages: 10 languages
   - Format: Question → Chain-of-thought → Answer

2. **XNLI** (Cross-lingual Natural Language Inference)
   - Task: Natural language inference
   - Languages: 15 languages
   - Format: Premise + Hypothesis → Label

3. **MSVAMP** (Multilingual Simple Variations on Arithmetic Math Problems)
   - Task: Mathematical word problems
   - Languages: 9 languages
   - Format: Question → Chain-of-thought → Answer

### Data Loading

Data is automatically downloaded from HuggingFace Datasets:

```python
from datasets import load_dataset

# MGSM
mgsm = load_dataset('jbross-ibm-research/mgsm', 'zh')

# XNLI
xnli = load_dataset('xnli', 'zh')
```


For questions or issues, please contact: kirtane.neeraja@gmail.com