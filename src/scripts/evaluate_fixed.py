#!/usr/bin/env python3
"""
Simplified and robust evaluation script for the Transformer model.
"""
import os
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.transformer import Transformer
from src.data.dataset import TranslationDataset
from src.data.collate import collate_fn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

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

def load_model(model_path, device):
    """Load model and tokenizers from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from the model's state dict
    state_dict = checkpoint['state_dict']
    
    # Extract model configuration from the state dict
    # This assumes the model was saved with the same architecture as in training
    d_model = state_dict['encoder.layers.0.self_attn.in_proj_weight'].shape[0] // 3
    num_heads = 8  # Default value, adjust based on your model
    num_encoder_layers = max([int(k.split('.')[2]) for k in state_dict.keys() if 'encoder.layers' in k and 'self_attn' in k]) + 1
    num_decoder_layers = max([int(k.split('.')[2]) for k in state_dict.keys() if 'decoder.layers' in k and 'self_attn' in k]) + 1
    
    # Get vocab sizes from tokenizers
    src_tokenizer = checkpoint['src_tokenizer']
    tgt_tokenizer = checkpoint['tgt_tokenizer']
    src_vocab_size = len(src_tokenizer) if hasattr(src_tokenizer, '__len__') else 32000
    tgt_vocab_size = len(tgt_tokenizer) if hasattr(tgt_tokenizer, '__len__') else 32000
    
    logger.info(f"Model configuration:")
    logger.info(f"  Source vocab size: {src_vocab_size}")
    logger.info(f"  Target vocab size: {tgt_vocab_size}")
    logger.info(f"  d_model: {d_model}")
    logger.info(f"  num_heads: {num_heads}")
    logger.info(f"  Encoder layers: {num_encoder_layers}")
    logger.info(f"  Decoder layers: {num_decoder_layers}")
    
    # Initialize model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=d_model * 4,  # Common practice
        max_seq_length=100,  # Default value, adjust if needed
        dropout=0.1  # Default value, adjust if needed
    )
    
    # Load model weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, src_tokenizer, tgt_tokenizer

def filter_tokens(tokens):
    """Filter out special tokens from a sequence."""
    if isinstance(tokens, (int, np.integer)):
        tokens = [tokens]
    tokens = np.array(tokens, dtype=np.int64)
    if tokens.ndim == 0:
        tokens = tokens.reshape(1)
    return [int(t) for t in tokens.flatten() if int(t) > 3]  # Keep only non-special tokens

def calculate_metrics(references, predictions):
    """Calculate BLEU and ROUGE scores."""
    # Simple BLEU implementation (replace with actual BLEU calculation)
    def simple_bleu(references, predictions):
        # This is a placeholder - replace with actual BLEU calculation
        return 0.0
    
    # Simple ROUGE implementation (replace with actual ROUGE calculation)
    def simple_rouge(references, predictions):
        # This is a placeholder - replace with actual ROUGE calculation
        return {}
    
    return {
        'bleu': simple_bleu(references, predictions),
        'rouge': simple_rouge(references, predictions)
    }

def evaluate_model(model, data_loader, src_tokenizer, tgt_tokenizer, device):
    """Evaluate model on the given data loader."""
    model.eval()
    
    all_predictions = []
    all_references = []
    total_loss = 0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Calculate loss
            output_flat = output.contiguous().view(-1, output.size(-1))
            tgt_flat = tgt_output.contiguous().view(-1)
            
            # Only calculate loss for non-padding tokens
            non_pad_mask = tgt_flat.ne(0)
            if non_pad_mask.sum().item() > 0:
                loss = criterion(output_flat, tgt_flat)
                total_loss += loss.item()
                total_tokens += non_pad_mask.sum().item()
            
            # Get predictions
            predictions = output.argmax(dim=-1)
            
            # Process batch
            for i in range(predictions.size(0)):
                try:
                    # Get and filter tokens
                    pred_tokens = filter_tokens(predictions[i].cpu().numpy())
                    tgt_tokens = filter_tokens(tgt_output[i].cpu().numpy())
                    
                    # Decode to strings
                    pred_str = tgt_tokenizer.decode(pred_tokens) if pred_tokens else ""
                    tgt_str = tgt_tokenizer.decode(tgt_tokens) if tgt_tokens else ""
                    
                    all_predictions.append(pred_str)
                    all_references.append([tgt_str])
                    
                    # Log first few examples
                    if len(all_predictions) <= 3:
                        logger.info(f"\nExample {len(all_predictions)}:")
                        logger.info(f"  Reference:  {tgt_str}")
                        logger.info(f"  Prediction: {pred_str}")
                        
                except Exception as e:
                    logger.error(f"Error processing example: {str(e)}")
                    all_predictions.append("")
                    all_references.append([""])
    
    # Calculate metrics
    metrics = calculate_metrics(all_references, all_predictions)
    
    # Prepare results
    results = {
        'loss': total_loss / total_tokens if total_tokens > 0 else float('nan'),
        'bleu': metrics['bleu'],
        'rouge': metrics['rouge'],
        'num_examples': len(all_predictions),
        'num_valid_examples': sum(1 for p in all_predictions if p.strip())
    }
    
    return results

def main():
    # Parse command line arguments
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
    
    logger.info("=" * 60)
    logger.info("Starting Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {Config.device}")
    logger.info(f"Model: {Config.model_path}")
    logger.info(f"Data directory: {Config.data_dir}")
    logger.info(f"Batch size: {Config.batch_size}")
    
    try:
        # Load model and tokenizers
        logger.info("\nLoading model and tokenizers...")
        model, src_tokenizer, tgt_tokenizer = load_model(Config.model_path, Config.device)
        
        # Load test dataset
        logger.info("\nLoading test dataset...")
        test_dataset = TranslationDataset(
            src_file=os.path.join(Config.data_dir, 'test.de'),
            tgt_file=os.path.join(Config.data_dir, 'test.en'),
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
        
        logger.info(f"Loaded {len(test_dataset)} examples")
        
        # Evaluate model
        logger.info("\nStarting evaluation...")
        results = evaluate_model(model, test_loader, src_tokenizer, tgt_tokenizer, Config.device)
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Examples processed: {results['num_examples']}")
        logger.info(f"Valid examples: {results['num_valid_examples']}")
        logger.info(f"Average loss: {results['loss']:.4f}")
        logger.info(f"BLEU score: {results['bleu']:.4f}")
        
        if results['rouge']:
            logger.info("\nROUGE Scores:")
            for metric, scores in results['rouge'].items():
                logger.info(f"  {metric.upper()}: {scores}")
        
        logger.info("\nEvaluation complete!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
