"""
Final evaluation script for the Transformer model.
Automatically adapts to different model architectures.
"""
import os
import json
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bleu(references, predictions):
    """Calculate BLEU score using nltk."""
    try:
        # Convert references to the format expected by NLTK
        refs = [[ref.split()] for ref in references]
        hyps = [pred.split() for pred in predictions]
        
        # Calculate BLEU-4 score with smoothing
        smoothie = SmoothingFunction().method4
        bleu_score = corpus_bleu(
            refs, 
            hyps,
            smoothing_function=smoothie,
            weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4
        )
        
        return bleu_score * 100  # Convert to percentage
        
    except Exception as e:
        logger.warning(f"Error calculating BLEU score: {str(e)}")
        return 0.0

def calculate_metrics(references, predictions):
    """Calculate various evaluation metrics."""
    # Calculate exact match accuracy
    exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)
    accuracy = exact_matches / len(predictions) if predictions else 0.0
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(references, predictions)
    
    # Calculate average length difference
    ref_lengths = [len(ref.split()) for ref in references]
    pred_lengths = [len(pred.split()) for pred in predictions]
    avg_length_diff = sum(p - r for p, r in zip(pred_lengths, ref_lengths)) / len(predictions) if predictions else 0.0
    
    # Calculate token-level accuracy (approximate)
    total_tokens = sum(len(ref.split()) for ref in references)
    correct_tokens = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        correct_tokens += sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
    
    token_accuracy = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
    
    return {
        'exact_match': accuracy * 100,  # as percentage
        'bleu': bleu_score,  # as percentage
        'token_accuracy': token_accuracy,  # as percentage
        'avg_length_diff': avg_length_diff,
        'num_examples': len(predictions),
        'valid_examples': sum(1 for p in predictions if p.strip())
    }

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Import your model and data loading code
try:
    from src.models.transformer import Transformer
    from src.data.dataset import TranslationDataset
    from src.data.collate import collate_fn
except ImportError as e:
    logger.error("Failed to import required modules. Make sure you're running from the project root.")
    logger.error(f"Error: {str(e)}")
    raise

class Config:
    """Configuration for evaluation."""
    # Model
    model_path = 'checkpoints/model_best.pt'
    
    # Data
    data_dir = 'data/multi30k'
    max_length = 100
    batch_size = 32
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inspect_model(checkpoint):
    """Inspect the model checkpoint structure."""
    logger.info("Inspecting model checkpoint...")
    
    # Get state dict
    state_dict = checkpoint['state_dict']
    
    # Print first 10 keys to understand the structure
    logger.info("First 10 state dict keys:")
    for i, k in enumerate(list(state_dict.keys())[:10]):
        logger.info(f"  {i+1}. {k} - {state_dict[k].shape if hasattr(state_dict[k], 'shape') else 'No shape'}")
    
    # Try to infer model architecture
    model_info = {
        'has_encoder': any('encoder' in k for k in state_dict),
        'has_decoder': any('decoder' in k for k in state_dict),
        'embedding_keys': [k for k in state_dict if 'embedding' in k]
    }
    
    return model_info

def load_model(model_path, device):
    """Load model and tokenizers with automatic architecture detection."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get tokenizers
    src_tokenizer = checkpoint.get('src_tokenizer')
    tgt_tokenizer = checkpoint.get('tgt_tokenizer')
    
    if src_tokenizer is None or tgt_tokenizer is None:
        raise ValueError("Tokenizer information not found in checkpoint")
    
    # Inspect model structure
    model_info = inspect_model(checkpoint)
    
    # Get model dimensions from the checkpoint state dict
    src_vocab_size = checkpoint['state_dict']['encoder_embedding.weight'].shape[0]
    tgt_vocab_size = checkpoint['state_dict']['decoder_embedding.weight'].shape[0]
    logger.info(f"Source vocab size from checkpoint: {src_vocab_size}")
    logger.info(f"Target vocab size from checkpoint: {tgt_vocab_size}")
    
    # Default model parameters (will be overridden by checkpoint if available)
    model_params = {
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,  # Changed from dim_feedforward to d_ff
        'max_seq_length': Config.max_length,
        'dropout': 0.1
    }
    
    # Try to get model parameters from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        for key in model_params:
            if key in config:
                model_params[key] = config[key]
    
    logger.info("Model parameters:")
    for k, v in model_params.items():
        logger.info(f"  {k}: {v}")
    
    # Initialize model
    model = Transformer(**model_params)
    
    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, src_tokenizer, tgt_tokenizer

def decode_tokens(tokenizer, tokens):
    """Decode tokens to string, handling different tokenizer types."""
    if not tokens:
        return ""
    
    if hasattr(tokenizer, 'decode'):
        return tokenizer.decode(tokens)
    elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'decode'):
        return tokenizer.tokenizer.decode(tokens)
    elif hasattr(tokenizer, 'decode_ids'):
        return tokenizer.decode_ids(tokens)
    else:
        # Fallback: try to join tokens as strings
        try:
            if isinstance(tokens[0], (list, np.ndarray)):
                tokens = tokens[0]  # Handle nested lists
            return ' '.join(str(t) for t in tokens if t not in [0, 1, 2, 3])
        except Exception as e:
            logger.warning(f"Error decoding tokens: {e}")
            return str(tokens)

def evaluate():
    """Run evaluation."""
    logger.info("=" * 60)
    logger.info("Starting Evaluation")
    logger.info("=" * 60)
    logger.info(f"Using device: {Config.device}")
    
    try:
        # Load model and tokenizers
        model, src_tokenizer, tgt_tokenizer = load_model(
            Config.model_path, Config.device
        )
        
        # Load test dataset
        test_src = os.path.join(Config.data_dir, 'test.de')
        test_tgt = os.path.join(Config.data_dir, 'test.en')
        
        if not os.path.exists(test_src) or not os.path.exists(test_tgt):
            raise FileNotFoundError(
                f"Test files not found in {Config.data_dir}. "
                "Make sure to run download_multi30k.py first."
            )
        
        logger.info("Loading test dataset...")
        test_dataset = TranslationDataset(
            src_file=test_src,
            tgt_file=test_tgt,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_length=Config.max_length,
            train=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=min(4, os.cpu_count() // 2),
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Loaded {len(test_dataset)} test examples")
        
        # Evaluation loop
        logger.info("\nStarting evaluation...")
        model.eval()
        
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                try:
                    src = batch['src'].to(Config.device)
                    tgt_input = batch['tgt_input'].to(Config.device)
                    tgt_output = batch['tgt_output'].to(Config.device)
                    
                    # Forward pass
                    output = model(src, tgt_input)
                    
                    # Get predictions (greedy decoding)
                    predictions = output.argmax(dim=-1)
                    
                    # Process batch
                    for i in range(predictions.size(0)):
                        # Get source, target, and prediction tokens
                        src_tokens = src[i].cpu().tolist()
                        tgt_tokens = tgt_output[i].cpu().tolist()
                        pred_tokens = predictions[i].cpu().tolist()
                        
                        # Filter out padding and special tokens
                        def filter_tokens(tokens):
                            return [t for t in tokens if t not in [0, 1, 2, 3]]
                        
                        src_filtered = filter_tokens(src_tokens)
                        tgt_filtered = filter_tokens(tgt_tokens)
                        pred_filtered = filter_tokens(pred_tokens)
                        
                        # Decode to strings
                        src_str = decode_tokens(src_tokenizer, src_filtered)
                        tgt_str = decode_tokens(tgt_tokenizer, tgt_filtered)
                        pred_str = decode_tokens(tgt_tokenizer, pred_filtered)
                        
                        all_references.append(tgt_str)
                        all_predictions.append(pred_str)
                        
                        # Log first few examples
                        if len(all_predictions) <= 3:
                            logger.info(f"\nExample {len(all_predictions)}:")
                            logger.info(f"  Source:     {src_str}")
                            logger.info(f"  Reference:  {tgt_str}")
                            logger.info(f"  Prediction: {pred_str}")
                
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                    continue
        
        # Calculate metrics
        metrics = calculate_metrics(all_references, all_predictions)
        
        # Prepare results
        results = {
            'metrics': metrics,
            'predictions': all_predictions[:100],  # Save only first 100 to avoid large files
            'references': all_references[:100]
        }
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Examples processed: {metrics['num_examples']}")
        logger.info(f"Valid examples: {metrics['valid_examples']}")
        logger.info("\n--- Metrics ---")
        logger.info(f"Exact Match: {metrics['exact_match']:.2f}%")
        logger.info(f"BLEU Score: {metrics['bleu']:.2f}")
        logger.info(f"Token Accuracy: {metrics['token_accuracy']:.2f}%")
        logger.info(f"Avg Length Difference: {metrics['avg_length_diff']:.2f} (pred - ref)")
        
        # Show first few examples
        logger.info("\n--- Example Translations ---")
        for i in range(min(3, len(all_predictions))):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"  Source:     {all_references[i]}")
            logger.info(f"  Reference:  {all_references[i]}")
            logger.info(f"  Prediction: {all_predictions[i]}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_file = 'results/evaluation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nResults saved to: {os.path.abspath(results_file)}")
        logger.info("Evaluation complete!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Transformer model')
    parser.add_argument('--model', type=str, default=Config.model_path,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default=Config.data_dir,
                       help='Directory containing test data')
    parser.add_argument('--batch-size', type=int, default=Config.batch_size,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Update config
    Config.model_path = args.model
    Config.data_dir = args.data_dir
    Config.batch_size = args.batch_size
    
    exit(evaluate())
