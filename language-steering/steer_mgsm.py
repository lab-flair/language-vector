"""
Language Steering for Multilingual In-Context Learning - MGSM
===============================================================

This script implements the language steering approach for mathematical reasoning
on the MGSM (Multilingual Grade School Math) dataset.

The method:
1. Computes language-specific steering vectors from activation differences
2. Applies these vectors during inference to improve cross-lingual transfer
3. Evaluates across multiple target languages with validation-guided hyperparameter selection


"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import re
import os
import time
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
from dataclasses import dataclass, asdict
import pickle
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SteeringConfig:
    """Configuration for language steering experiments."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Dataset settings
    dataset_name: str = 'mgsm'
    source_language: str = 'en'
    target_languages: List[str] = None
    
    # Few-shot settings
    num_shots: int = 6
    seed: int = 42
    
    # Steering vector computation
    questions_per_sample: int = 6  # Number of Q&A pairs per sample
    
    # Hyperparameter search spaces
    test_layers: List[int] = None
    test_alphas: List[float] = None
    steer_locations: List[str] = None
    
    # Generation settings
    max_new_tokens: int = 512
    batch_size: int = 16
    
    # Output settings
    output_dir: str = 'steering_results'
    verbose_examples: int = 3
    
    def __post_init__(self):
        """Set default values for None fields."""
        if self.target_languages is None:
            self.target_languages = ["zh", "ja", "th", "sw", "bn", "de", "es", "fr", "ru"]
        if self.test_layers is None:
            self.test_layers = [5, 10, 15, 20, 25, 30]
        if self.test_alphas is None:
            self.test_alphas = [0.5, 1.0, 2.0, 3.0]
        if self.steer_locations is None:
            self.steer_locations = ['on_fewshot', 'after_fewshot', 'on_question', 'entire']


# Dataset mapping
DATASET_MAP = {
    'mgsm': 'jbross-ibm-research/mgsm'
}


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✓ Random seed set to {seed}")


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return path."""
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get formatted timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================

def extract_final_answer(text: str) -> Optional[float]:
    """
    Extract numerical answer from model output.
    
    Tries multiple patterns in order:
    1. "Final answer: X" format
    2. "#### X" format (GSM8K style)
    3. "The answer is X" format
    4. Last number in text (fallback)
    
    Args:
        text: Model generated text
        
    Returns:
        Extracted numerical answer or None if not found
    """
    # Pattern 1: "Final answer:" format (case-insensitive)
    final_re = re.compile(r"final\s*answer\s*[:\-]?\s*([-+]?\d+(?:[,\.]\d+)*)", re.IGNORECASE)
    match = final_re.search(text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Pattern 2: "####" format
    hash_re = re.compile(r"####\s*([-+]?\d+(?:[,\.]\d+)*)")
    match = hash_re.search(text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Pattern 3: "The answer is X" format
    answer_re = re.compile(r"(?:the\s+)?answer\s+is\s+([-+]?\d+(?:[,\.]\d+)*)", re.IGNORECASE)
    match = answer_re.search(text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Fallback: last number in text
    all_numbers = re.findall(r"[-+]?\d+(?:[,\.]\d+)*", text)
    if all_numbers:
        try:
            return float(all_numbers[-1].replace(',', ''))
        except ValueError:
            pass
    
    return None


def answers_match(pred: Optional[float], gold: Optional[float], tolerance: float = 1e-3) -> bool:
    """Check if predicted and gold answers match within tolerance."""
    if pred is None or gold is None:
        return False
    return abs(float(pred) - float(gold)) <= tolerance


# ============================================================================
# DATA LOADING AND SPLITTING
# ============================================================================

def load_mgsm_with_splits(src_lang: str, tgt_lang: str) -> Dict[str, Any]:
    """
    Load MGSM dataset and create three-way split.
    
    Splits the test set into three equal parts:
    - compute: for steering vector calculation
    - val: for hyperparameter selection
    - test: for final evaluation
    
    Args:
        src_lang: Source language code (e.g., 'en')
        tgt_lang: Target language code (e.g., 'zh')
        
    Returns:
        Dictionary containing train and split data
    """
    print("\n" + "="*80)
    print("📚 LOADING DATA")
    print("="*80)
    
    # Load datasets
    ds_src = load_dataset(DATASET_MAP['mgsm'], src_lang)
    ds_tgt = load_dataset(DATASET_MAP['mgsm'], tgt_lang)
    
    src_train = ds_src['train']
    tgt_train = ds_tgt['train']
    src_test = ds_src['test']
    tgt_test = ds_tgt['test']
    
    # Create three-way split
    total_test = len(src_test)
    split_size = total_test // 3
    
    compute_indices = list(range(0, split_size))
    val_indices = list(range(split_size, 2 * split_size))
    test_indices = list(range(2 * split_size, total_test))
    
    splits = {
        'train': {
            'source': src_train,
            'target': tgt_train
        },
        'compute': {
            'source': src_test.select(compute_indices),
            'target': tgt_test.select(compute_indices),
            'indices': compute_indices
        },
        'val': {
            'source': src_test.select(val_indices),
            'target': tgt_test.select(val_indices),
            'indices': val_indices
        },
        'test': {
            'source': src_test.select(test_indices),
            'target': tgt_test.select(test_indices),
            'indices': test_indices
        }
    }
    
    print(f"\nDataset: MGSM ({src_lang} → {tgt_lang})")
    print(f"  Train:   {len(src_train)} examples")
    print(f"  Compute: {len(compute_indices)} examples (for vector computation)")
    print(f"  Val:     {len(val_indices)} examples (for hyperparameter selection)")
    print(f"  Test:    {len(test_indices)} examples (for final evaluation)")
    
    return splits


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with appropriate settings.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print("\n" + "="*80)
    print("🤖 LOADING MODEL")
    print("="*80)
    print(f"Model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left',
        trust_remote_code=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    
    print(f"✓ Model loaded")
    print(f"  Device map: {model.hf_device_map}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    
    return model, tokenizer


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def format_fewshot_prompt(examples: List[Dict], system_instruction: str = "") -> str:
    """
    Format few-shot examples into a prompt.
    
    Args:
        examples: List of examples with 'question', 'answer', 'answer_number' fields
        system_instruction: Optional system instruction
        
    Returns:
        Formatted prompt string
    """
    if system_instruction:
        prompt = f"{system_instruction}\n\n"
    else:
        prompt = ""
    
    for ex in examples:
        prompt += f"Question: {ex['question']}\n"
        prompt += f"Answer: {ex['answer']}\n"
        prompt += f"Final answer: {ex['answer_number']}\n\n"
    
    return prompt


def create_multiq_sample(dataset: Any, indices: List[int], k: int) -> str:
    """
    Create a sample with k question-answer pairs concatenated.
    
    This is used for computing steering vectors with richer context.
    
    Args:
        dataset: Dataset to sample from
        indices: Indices to choose from
        k: Number of Q&A pairs per sample
        
    Returns:
        Formatted text with k Q&A pairs
    """
    sample_text = ""
    for idx in indices[:k]:
        ex = dataset[idx]
        sample_text += f"Question: {ex['question']}\n"
        sample_text += f"Answer: {ex['answer']}\n"
        sample_text += f"Final answer: {ex['answer_number']}\n\n"
    return sample_text.strip()


# ============================================================================
# STEERING VECTOR COMPUTATION
# ============================================================================

def compute_steering_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compute_split: Dict[str, Any],
    layers: List[int],
    questions_per_sample: int,
    batch_size: int,
    seed: int
) -> Dict[int, torch.Tensor]:
    """
    Compute language-specific steering vectors.
    
    Process:
    1. Create samples with multiple Q&A pairs
    2. Extract hidden states at specified layers
    3. Compute mean activation difference between target and source
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        compute_split: Data split for vector computation
        layers: List of layer indices to extract from
        questions_per_sample: Number of Q&A pairs per sample
        batch_size: Batch size for processing
        seed: Random seed
        
    Returns:
        Dictionary mapping layer index to steering vector
    """
    print("\n" + "="*80)
    print("🧮 COMPUTING STEERING VECTORS")
    print("="*80)
    
    np.random.seed(seed)
    
    src_dataset = compute_split['source']
    tgt_dataset = compute_split['target']
    n_examples = len(src_dataset)
    
    # Create samples
    print(f"\nCreating samples with {questions_per_sample} Q&A pairs each...")
    src_samples = []
    tgt_samples = []
    
    for i in range(n_examples):
        # Sample indices ensuring each example appears at least once
        indices = [i]
        remaining = questions_per_sample - 1
        if remaining > 0:
            other_indices = np.random.choice(
                [j for j in range(n_examples) if j != i],
                size=remaining,
                replace=True
            )
            indices.extend(other_indices.tolist())
        
        src_samples.append(create_multiq_sample(src_dataset, indices, questions_per_sample))
        tgt_samples.append(create_multiq_sample(tgt_dataset, indices, questions_per_sample))
    
    # Extract activations
    print(f"Extracting activations from {len(layers)} layers...")
    
    def extract_activations(texts: List[str], layers: List[int]) -> Dict[int, List[torch.Tensor]]:
        """Extract mean-pooled activations for each layer."""
        layer_activations = {layer: [] for layer in layers}
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                for layer in layers:
                    # Get hidden states for this layer [batch, seq_len, hidden_dim]
                    h = hidden_states[layer]
                    
                    # Mean pool over sequence length
                    # Create attention mask to exclude padding
                    mask = inputs['attention_mask'].unsqueeze(-1).to(h.dtype)
                    masked_h = h * mask
                    summed = masked_h.sum(dim=1)
                    counts = mask.sum(dim=1)
                    mean_h = summed / counts.clamp(min=1)
                    
                    layer_activations[layer].append(mean_h)
        
        # Concatenate all batches
        return {layer: torch.cat(acts, dim=0) for layer, acts in layer_activations.items()}
    
    print("  Extracting source activations...")
    src_acts = extract_activations(src_samples, layers)
    
    print("  Extracting target activations...")
    tgt_acts = extract_activations(tgt_samples, layers)
    
    # Compute steering vectors
    steering_vectors = {}
    for layer in layers:
        # Compute mean difference: target - source
        diff = tgt_acts[layer] - src_acts[layer]
        steering_vector = diff.mean(dim=0)
        steering_vectors[layer] = steering_vector
        
        print(f"  Layer {layer:2d}: vector shape {steering_vector.shape}, norm {steering_vector.norm():.4f}")
    
    print(f"\n✓ Computed steering vectors for {len(layers)} layers")
    
    return steering_vectors


# ============================================================================
# INFERENCE WITH STEERING
# ============================================================================

class SteeringHook:
    """Hook for applying steering during forward pass."""
    
    def __init__(self, steering_vector: torch.Tensor, alpha: float, positions: List[int]):
        """
        Initialize steering hook.
        
        Args:
            steering_vector: Vector to add to activations
            alpha: Scaling factor
            positions: Token positions to steer
        """
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.positions = set(positions)
        self.current_position = 0
    
    def __call__(self, module, input, output):
        """Apply steering to output hidden states."""
        # output is a tuple, hidden states are first element
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Apply steering to specified positions
        for pos in range(hidden_states.size(1)):
            if self.current_position + pos in self.positions:
                hidden_states[:, pos, :] = (
                    hidden_states[:, pos, :] + 
                    self.alpha * self.steering_vector.to(hidden_states.device)
                )
        
        self.current_position += hidden_states.size(1)
        
        return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states
    
    def reset(self):
        """Reset position counter."""
        self.current_position = 0


def get_steer_positions(
    tokenizer: AutoTokenizer,
    prompt: str,
    fewshot_prompt: str,
    question: str,
    location: str
) -> List[int]:
    """
    Determine which token positions to apply steering.
    
    Args:
        tokenizer: Tokenizer
        prompt: Full prompt
        fewshot_prompt: Few-shot portion of prompt
        question: Test question
        location: Steering location ('on_fewshot', 'after_fewshot', 'on_question', 'entire')
        
    Returns:
        List of token positions to steer
    """
    full_tokens = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
    
    if location == 'entire':
        return list(range(len(full_tokens)))
    
    elif location == 'on_fewshot':
        fewshot_tokens = tokenizer(fewshot_prompt, return_tensors='pt')['input_ids'][0]
        return list(range(len(fewshot_tokens)))
    
    elif location == 'after_fewshot':
        fewshot_tokens = tokenizer(fewshot_prompt, return_tensors='pt')['input_ids'][0]
        return [len(fewshot_tokens)]
    
    elif location == 'on_question':
        fewshot_tokens = tokenizer(fewshot_prompt, return_tensors='pt')['input_ids'][0]
        question_tokens = tokenizer(question, return_tensors='pt')['input_ids'][0]
        start = len(fewshot_tokens)
        return list(range(start, start + len(question_tokens)))
    
    else:
        raise ValueError(f"Unknown location: {location}")


def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    fewshot_prompt: str,
    question: str,
    layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
    location: str,
    max_new_tokens: int = 512
) -> str:
    """
    Generate text with steering applied.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Full input prompt
        fewshot_prompt: Few-shot portion (for position calculation)
        question: Test question (for position calculation)
        layer: Layer to apply steering
        steering_vector: Steering vector
        alpha: Scaling factor
        location: Where to apply steering
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
    """
    # Determine positions to steer
    positions = get_steer_positions(tokenizer, prompt, fewshot_prompt, question, location)
    
    # Create hook
    hook = SteeringHook(steering_vector, alpha, positions)
    
    # Register hook on appropriate layer
    target_layer = model.model.layers[layer]
    handle = target_layer.register_forward_hook(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
    finally:
        handle.remove()
    
    return generated_text


def generate_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512
) -> str:
    """Generate text without steering (baseline)."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_split(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    split: Dict[str, Any],
    fewshot_examples: List[Dict],
    steering_vectors: Optional[Dict[int, torch.Tensor]] = None,
    layer: Optional[int] = None,
    alpha: Optional[float] = None,
    location: Optional[str] = None,
    max_new_tokens: int = 512,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Evaluate on a data split with optional steering.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        split: Data split to evaluate
        fewshot_examples: Few-shot demonstration examples
        steering_vectors: Steering vectors (None for baseline)
        layer: Layer to steer (required if steering)
        alpha: Scaling factor (required if steering)
        location: Steering location (required if steering)
        max_new_tokens: Max tokens to generate
        verbose: Number of examples to print (0 for none)
        
    Returns:
        Dictionary with accuracy and detailed results
    """
    use_steering = steering_vectors is not None
    
    # Build few-shot prompt
    fewshot_prompt = format_fewshot_prompt(fewshot_examples)
    
    correct = 0
    total = 0
    results = []
    
    target_data = split['target']
    
    for i, example in enumerate(tqdm(target_data, desc="Evaluating")):
        question = example['question']
        gold_answer = example['answer_number']
        
        # Full prompt
        prompt = fewshot_prompt + f"Question: {question}\nAnswer:"
        
        # Generate
        if use_steering:
            generated = generate_with_steering(
                model, tokenizer, prompt, fewshot_prompt, question,
                layer, steering_vectors[layer], alpha, location, max_new_tokens
            )
        else:
            generated = generate_baseline(model, tokenizer, prompt, max_new_tokens)
        
        # Extract answer
        predicted = extract_final_answer(generated)
        is_correct = answers_match(predicted, gold_answer)
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'question': question,
            'gold_answer': float(gold_answer),
            'predicted_answer': float(predicted) if predicted is not None else None,
            'correct': is_correct,
            'generated_text': generated
        })
        
        # Print verbose examples
        if verbose > 0 and i < verbose:
            print(f"\n{'='*80}")
            print(f"Example {i+1}")
            print(f"Question: {question}")
            print(f"Generated: {generated[:200]}...")
            print(f"Gold: {gold_answer}, Predicted: {predicted}, Correct: {is_correct}")
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


# ============================================================================
# VALIDATION AND TEST
# ============================================================================

def run_validation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    steering_vectors: Dict[int, torch.Tensor],
    splits: Dict[str, Any],
    config: SteeringConfig,
    output_dir: str
) -> Dict[str, Any]:
    """
    Run validation to select best hyperparameters.
    
    Tests all combinations of (layer, alpha, location) and identifies
    configurations that improve over baseline.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        steering_vectors: Computed steering vectors
        splits: Data splits
        config: Configuration
        output_dir: Output directory
        
    Returns:
        Validation results with configurations above baseline
    """
    print("\n" + "="*80)
    print("🔍 VALIDATION")
    print("="*80)
    
    # Sample few-shot examples
    np.random.seed(config.seed)
    demo_indices = np.random.choice(len(splits['train']['source']), size=config.num_shots, replace=False)
    fewshot_examples = [splits['train']['source'][int(i)] for i in demo_indices]
    
    # Baseline evaluation
    print("\nEvaluating baseline (no steering)...")
    baseline_results = evaluate_split(
        model, tokenizer, splits['val'], fewshot_examples,
        max_new_tokens=config.max_new_tokens,
        verbose=config.verbose_examples
    )
    baseline_acc = baseline_results['accuracy']
    print(f"✓ Baseline accuracy: {baseline_acc:.2%}")
    
    # Oracle evaluation (target language demos)
    print("\nEvaluating oracle (target language few-shot)...")
    oracle_examples = [splits['train']['target'][int(i)] for i in demo_indices]
    oracle_results = evaluate_split(
        model, tokenizer, splits['val'], oracle_examples,
        max_new_tokens=config.max_new_tokens
    )
    oracle_acc = oracle_results['accuracy']
    print(f"✓ Oracle accuracy: {oracle_acc:.2%}")
    
    # Test all configurations
    print(f"\nTesting {len(config.test_layers)} layers × {len(config.test_alphas)} alphas × {len(config.steer_locations)} locations")
    print(f"Total configurations: {len(config.test_layers) * len(config.test_alphas) * len(config.steer_locations)}")
    
    configurations = {layer: {loc: {} for loc in config.steer_locations} for layer in config.test_layers}
    configs_above_baseline = []
    
    for layer in config.test_layers:
        for location in config.steer_locations:
            for alpha in config.test_alphas:
                
                results = evaluate_split(
                    model, tokenizer, splits['val'], fewshot_examples,
                    steering_vectors, layer, alpha, location,
                    max_new_tokens=config.max_new_tokens
                )
                
                acc = results['accuracy']
                configurations[layer][location][alpha] = acc
                
                if acc > baseline_acc:
                    configs_above_baseline.append({
                        'layer': layer,
                        'alpha': alpha,
                        'location': location,
                        'val_accuracy': acc,
                        'improvement': acc - baseline_acc
                    })
                    print(f"  ✓ Layer {layer:2d}, α={alpha:.1f}, {location:15s}: {acc:.2%} (+{acc-baseline_acc:+.2%})")
    
    # Sort by validation accuracy
    configs_above_baseline.sort(key=lambda x: x['val_accuracy'], reverse=True)
    
    print(f"\n✓ Found {len(configs_above_baseline)} configurations above baseline")
    
    return {
        'baseline': baseline_acc,
        'oracle': oracle_acc,
        'demo_indices': demo_indices.tolist(),
        'configurations': configurations,
        'configs_above_baseline': configs_above_baseline
    }


def run_test(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    steering_vectors: Dict[int, torch.Tensor],
    splits: Dict[str, Any],
    val_results: Dict[str, Any],
    config: SteeringConfig,
    output_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Run test evaluation on configurations that passed validation.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        steering_vectors: Computed steering vectors
        splits: Data splits
        val_results: Validation results
        config: Configuration
        output_dir: Output directory
        
    Returns:
        Test results or None if no valid configurations
    """
    print("\n" + "="*80)
    print("🎯 TEST EVALUATION")
    print("="*80)
    
    configs_to_test = val_results['configs_above_baseline']
    
    if not configs_to_test:
        print("❌ No configurations above baseline on validation set")
        return None
    
    print(f"Testing {len(configs_to_test)} configurations...")
    
    # Use same few-shot examples as validation
    demo_indices = val_results['demo_indices']
    fewshot_examples = [splits['train']['source'][int(i)] for i in demo_indices]
    
    # Baseline
    print("\nEvaluating baseline...")
    baseline_results = evaluate_split(
        model, tokenizer, splits['test'], fewshot_examples,
        max_new_tokens=config.max_new_tokens
    )
    baseline_acc = baseline_results['accuracy']
    print(f"✓ Baseline: {baseline_acc:.2%}")
    
    # Oracle
    print("\nEvaluating oracle...")
    oracle_examples = [splits['train']['target'][int(i)] for i in demo_indices]
    oracle_results = evaluate_split(
        model, tokenizer, splits['test'], oracle_examples,
        max_new_tokens=config.max_new_tokens
    )
    oracle_acc = oracle_results['accuracy']
    print(f"✓ Oracle: {oracle_acc:.2%}")
    
    # Test each configuration
    steered_results = []
    
    for cfg in configs_to_test:
        layer = cfg['layer']
        alpha = cfg['alpha']
        location = cfg['location']
        
        print(f"\nTesting: Layer {layer}, α={alpha}, {location}")
        
        results = evaluate_split(
            model, tokenizer, splits['test'], fewshot_examples,
            steering_vectors, layer, alpha, location,
            max_new_tokens=config.max_new_tokens,
            verbose=config.verbose_examples
        )
        
        test_acc = results['accuracy']
        improvement = test_acc - baseline_acc
        
        print(f"  Test accuracy: {test_acc:.2%} ({improvement:+.2%})")
        
        steered_results.append({
            'config': cfg,
            'test_accuracy': test_acc,
            'improvement_over_baseline': improvement,
            'gap_to_oracle': oracle_acc - test_acc,
            'detailed_results': results['results']
        })
    
    # Find best on test
    best = max(steered_results, key=lambda x: x['test_accuracy'])
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION ON TEST SET")
    print("="*80)
    print(f"Layer: {best['config']['layer']}")
    print(f"Alpha: {best['config']['alpha']}")
    print(f"Location: {best['config']['location']}")
    print(f"Test accuracy: {best['test_accuracy']:.2%}")
    print(f"Improvement: {best['improvement_over_baseline']:+.2%}")
    print(f"Gap to oracle: {best['gap_to_oracle']:.2%}")
    
    return {
        'baseline': baseline_acc,
        'oracle': oracle_acc,
        'steered_results': steered_results,
        'best_on_test': best
    }


# ============================================================================
# SINGLE LANGUAGE PIPELINE
# ============================================================================

def run_single_language(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    target_lang: str,
    config: SteeringConfig,
    base_output_dir: str
) -> Dict[str, Any]:
    """
    Run complete pipeline for a single target language.
    
    Steps:
    1. Load and split data
    2. Compute steering vectors
    3. Run validation
    4. Run test on validated configurations
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        target_lang: Target language code
        config: Configuration
        base_output_dir: Base output directory
        
    Returns:
        Summary results for this language
    """
    print("\n" + "="*80)
    print(f"🌍 LANGUAGE: {config.source_language} → {target_lang}")
    print("="*80)
    
    lang_start = time.time()
    
    # Create language-specific output directory
    lang_output_dir = ensure_dir(
        os.path.join(base_output_dir, f"{config.source_language}_to_{target_lang}")
    )
    
    # Load data
    splits = load_mgsm_with_splits(config.source_language, target_lang)
    
    # Compute steering vectors
    steering_vectors = compute_steering_vectors(
        model, tokenizer, splits['compute'],
        config.test_layers, config.questions_per_sample,
        config.batch_size, config.seed
    )
    
    # Save steering vectors
    vectors_file = os.path.join(lang_output_dir, f'steering_vectors_{get_timestamp()}.pkl')
    with open(vectors_file, 'wb') as f:
        cpu_vectors = {k: v.cpu() for k, v in steering_vectors.items()}
        pickle.dump(cpu_vectors, f)
    print(f"\n✓ Steering vectors saved: {vectors_file}")
    
    # Run validation
    val_results = run_validation(
        model, tokenizer, steering_vectors, splits, config, lang_output_dir
    )
    
    # Save validation results
    val_file = os.path.join(lang_output_dir, f'validation_results_{get_timestamp()}.json')
    with open(val_file, 'w') as f:
        save_data = {
            'baseline': float(val_results['baseline']),
            'oracle': float(val_results['oracle']),
            'demo_indices': [int(x) for x in val_results['demo_indices']],
            'configs_above_baseline': val_results['configs_above_baseline'],
            'all_configurations': {}
        }
        for layer in val_results['configurations']:
            save_data['all_configurations'][f'layer_{layer}'] = {}
            for loc in val_results['configurations'][layer]:
                save_data['all_configurations'][f'layer_{layer}'][loc] = {
                    str(a): float(acc) for a, acc in val_results['configurations'][layer][loc].items()
                }
        json.dump(save_data, f, indent=2)
    print(f"✓ Validation results saved: {val_file}")
    
    # Run test
    test_results = run_test(
        model, tokenizer, steering_vectors, splits, val_results, config, lang_output_dir
    )
    
    # Save test results
    if test_results:
        test_file = os.path.join(lang_output_dir, f'test_results_{get_timestamp()}.json')
        with open(test_file, 'w') as f:
            save_data = {
                'baseline': float(test_results['baseline']),
                'oracle': float(test_results['oracle']),
                'num_configs_tested': len(val_results['configs_above_baseline']),
                'steered_results': []
            }
            
            for sr in test_results['steered_results']:
                save_data['steered_results'].append({
                    'config': sr['config'],
                    'test_accuracy': float(sr['test_accuracy']),
                    'improvement_over_baseline': float(sr['improvement_over_baseline']),
                    'gap_to_oracle': float(sr['gap_to_oracle'])
                })
            
            if 'best_on_test' in test_results:
                save_data['best_on_test'] = {
                    'config': test_results['best_on_test']['config'],
                    'test_accuracy': float(test_results['best_on_test']['test_accuracy']),
                    'improvement_over_baseline': float(test_results['best_on_test']['improvement_over_baseline'])
                }
            
            json.dump(save_data, f, indent=2)
        print(f"✓ Test results saved: {test_file}")
    
    lang_time = time.time() - lang_start
    print(f"\n✅ {target_lang} complete ({lang_time/60:.1f} minutes)")
    
    # Create summary
    summary = {
        'source_language': config.source_language,
        'target_language': target_lang,
        'time_minutes': lang_time / 60,
        'validation': {
            'baseline': float(val_results['baseline']),
            'oracle': float(val_results['oracle']),
            'num_configs_above_baseline': len(val_results['configs_above_baseline'])
        }
    }
    
    if test_results and 'best_on_test' in test_results:
        summary['test'] = {
            'baseline': float(test_results['baseline']),
            'oracle': float(test_results['oracle']),
            'best_accuracy': float(test_results['best_on_test']['test_accuracy']),
            'improvement': float(test_results['best_on_test']['improvement_over_baseline']),
            'gap_to_oracle': float(test_results['best_on_test']['gap_to_oracle']),
            'best_config': test_results['best_on_test']['config']
        }
    
    return summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(config: Optional[SteeringConfig] = None):
    """
    Main execution function.
    
    Args:
        config: Configuration object (uses default if None)
    """
    if config is None:
        config = SteeringConfig()
    
    print("="*80)
    print("🚀 LANGUAGE STEERING FOR MULTILINGUAL ICL - MGSM")
    print("="*80)
    print(f"\nModel: {config.model_name}")
    print(f"Source: {config.source_language}")
    print(f"Targets: {config.target_languages}")
    
    overall_start = time.time()
    
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    base_output_dir = ensure_dir(f"{config.output_dir}/{get_timestamp()}")
    
    # Save config
    config_file = os.path.join(base_output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"\n✓ Configuration saved: {config_file}")
    
    # Load model once
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    
    # Run for each target language
    all_results = []
    
    for i, target_lang in enumerate(config.target_languages):
        print(f"\n{'='*80}")
        print(f"LANGUAGE {i+1}/{len(config.target_languages)}")
        print(f"{'='*80}")
        
        try:
            lang_results = run_single_language(
                model, tokenizer, target_lang, config, base_output_dir
            )
            all_results.append(lang_results)
        except Exception as e:
            print(f"❌ Error with {target_lang}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall summary
    summary_file = os.path.join(base_output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results_by_language': all_results
        }, f, indent=2)
    
    overall_time = time.time() - overall_start
    
    # Print final summary
    print("\n" + "="*80)
    print(f"🎉 ALL LANGUAGES COMPLETE ({overall_time/60:.1f} minutes)")
    print("="*80)
    print(f"\n📁 Results saved to: {base_output_dir}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Language':<10} {'Baseline':<10} {'Best':<10} {'Improvement':<12} {'Oracle':<10}")
    print("-"*80)
    
    for res in all_results:
        lang = res['target_language']
        if 'test' in res:
            baseline = res['test']['baseline']
            best = res['test']['best_accuracy']
            improvement = res['test']['improvement']
            oracle = res['test']['oracle']
            print(f"{lang:<10} {baseline:>9.2%} {best:>9.2%} {improvement:>+10.1f}pp {oracle:>9.2%}")
        else:
            print(f"{lang:<10} No valid results")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()