import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

def calculate_bleu(
    references: list[list[str]],
    hypotheses: list[list[str]],
    weights: tuple = (0.25, 0.25, 0.25, 0.25)
) -> float:
    """Calculate BLEU score for reference and hypothesis sentences."""
    refs = [[ref] for ref in references]  # NLTK expects list of references
    hyps = [' '.join(hyp) for hyp in hypotheses]
    
    return corpus_bleu(
        refs, hyps,
        weights=weights,
        smoothing_function=SmoothingFunction().method1
    )

def calculate_rouge(
    references: list[str],
    hypotheses: list[str],
    metrics: list[str] = ['rouge-1', 'rouge-2', 'rouge-l'],
    stats: list[str] = ['f']
) -> dict:
    """Calculate ROUGE scores for reference and hypothesis sentences."""
    rouge = Rouge(metrics=metrics)
    scores = rouge.get_scores(hyps=hypotheses, refs=references, avg=True)
    
    if stats != ['f', 'p', 'r']:
        return {
            metric: {k: v for k, v in values.items() if k in stats}
            for metric, values in scores.items()
        }
    
    return scores

def calculate_meteor(
    references: list[list[str]],
    hypotheses: list[list[str]]
) -> float:
    """Calculate METEOR score for reference and hypothesis sentences."""
    scores = []
    
    for ref_tokens, hyp_tokens in zip(references, hypotheses):
        ref = ' '.join(ref_tokens)
        hyp = ' '.join(hyp_tokens)
        scores.append(meteor_score([ref], hyp))
    
    return float(np.mean(scores))

def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Calculate accuracy between predictions and targets."""
    mask = (targets != ignore_index).float()
    correct = (predictions == targets).float() * mask
    total = mask.sum().item()
    
    if total == 0:
        return 0.0
    
    return correct.sum().item() / total

def calculate_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Calculate perplexity from model logits and targets."""
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
    return np.exp(loss)

def calculate_f1(
    predictions: list[list[str]],
    references: list[list[str]],
    average: str = 'micro'
) -> float:
    """Calculate F1 score for token-level classification."""
    assert average in ['micro', 'macro'], "Average must be 'micro' or 'macro'"
    
    if average == 'micro':
        pred_tokens = [token for seq in predictions for token in seq]
        ref_tokens = [token for seq in references for token in seq]
        
        tp = sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
        fp = len(pred_tokens) - tp
        fn = len(ref_tokens) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Macro average
    all_tokens = set(token for seq in predictions + references for token in seq)
    f1_scores = []
    
    for token in all_tokens:
        tp = sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == token and r == token)
        fp = sum(1 for p in pred_tokens if p == token) - tp
        fn = sum(1 for r in ref_tokens if r == token) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0

def compute_bleu(outputs, targets, vocab, max_n=4, weights=None):
    """
    Compute BLEU score between model outputs and targets.
    
    Args:
        outputs: List of predicted token IDs (batch_size, seq_len)
        targets: List of target token IDs (batch_size, seq_len)
        vocab: Vocabulary mapping indices to tokens
        max_n: Maximum n-gram order for BLEU score (default: 4)
        weights: Weights for n-gram precisions (default: uniform)
        
    Returns:
        BLEU score (float)
    """
    # Convert token IDs to strings
    def to_tokens(ids, vocab):
        tokens = []
        for idx in ids:
            token = vocab.get(idx.item(), '<unk>')
            if token == '<eos>':
                break
            if token not in ['<pad>', '<bos>', '<unk>']:
                tokens.append(token)
        return tokens
    
    # Prepare references and hypotheses
    references = []
    hypotheses = []
    
    for output, target in zip(outputs, targets):
        # Convert to tokens
        ref = [to_tokens(target, {v: k for k, v in vocab.items()})]
        hyp = to_tokens(output, {v: k for k, v in vocab.items()})
        
        references.append(ref)
        hypotheses.append(hyp)
    
    # Compute BLEU score
    return calculate_bleu(references, hypotheses, weights=weights or (1.0/max_n,) * max_n)

def compute_rouge(outputs, targets, vocab):
    """
    Compute ROUGE scores between model outputs and targets.
    
    Args:
        outputs: List of predicted token IDs (batch_size, seq_len)
        targets: List of target token IDs (batch_size, seq_len)
        vocab: Vocabulary mapping indices to tokens
        
    Returns:
        Dictionary of ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    """
    # Convert token IDs to strings
    def to_string(ids, vocab):
        tokens = []
        for idx in ids:
            token = vocab.get(idx.item(), '<unk>')
            if token == '<eos>':
                break
            if token not in ['<pad>', '<bos>', '<unk>']:
                tokens.append(token)
        return ' '.join(tokens)
    
    # Prepare references and hypotheses
    references = []
    hypotheses = []
    
    for output, target in zip(outputs, targets):
        # Convert to strings
        ref = to_string(target, {v: k for k, v in vocab.items()})
        hyp = to_string(output, {v: k for k, v in vocab.items()})
        
        references.append(ref)
        hypotheses.append(hyp)
    
    # Compute ROUGE scores
    return calculate_rouge(references, hypotheses)

def compute_meteor(outputs, targets, vocab):
    """
    Compute METEOR score between model outputs and targets.
    
    Args:
        outputs: List of predicted token IDs (batch_size, seq_len)
        targets: List of target token IDs (batch_size, seq_len)
        vocab: Vocabulary mapping indices to tokens
        
    Returns:
        METEOR score (float)
    """
    # Convert token IDs to strings
    def to_tokens(ids, vocab):
        tokens = []
        for idx in ids:
            token = vocab.get(idx.item(), '<unk>')
            if token == '<eos>':
                break
            if token not in ['<pad>', '<bos>', '<unk>']:
                tokens.append(token)
        return tokens
    
    # Prepare references and hypotheses
    references = []
    hypotheses = []
    
    for output, target in zip(outputs, targets):
        # Convert to tokens
        ref = [to_tokens(target, {v: k for k, v in vocab.items()})]
        hyp = to_tokens(output, {v: k for k, v in vocab.items()})
        
        references.append(ref)
        hypotheses.append(hyp)
    
    # Compute METEOR score
    return calculate_meteor(references, hypotheses)

def compute_accuracy(logits, targets, pad_idx=0):
    """
    Compute accuracy between model logits and targets.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        pad_idx: Padding token index to ignore
        
    Returns:
        Accuracy (float)
    """
    # Get predictions (indices of max logits)
    predictions = logits.argmax(dim=-1)
    
    # Create mask for non-padding tokens
    mask = (targets != pad_idx).float()
    
    # Count correct predictions
    correct = (predictions == targets).float() * mask
    
    # Compute accuracy
    accuracy = correct.sum() / mask.sum()
    
    return accuracy.item()

def compute_perplexity(loss):
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss (float)
        
    Returns:
        Perplexity (float)
    """
    return np.exp(loss)
