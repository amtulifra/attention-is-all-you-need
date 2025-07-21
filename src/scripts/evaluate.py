import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src.models.transformer import Transformer
from src.data.dataset import TranslationDataset
from src.data.collate import collate_fn
from src.data.tokenizer import Tokenizer
from src.utils.metrics import calculate_bleu, calculate_rouge
from src.utils.visualization import plot_attention_weights
from src.utils import setup_logging

logger = setup_logging(log_dir='logs')

class Config:
    # Model
    model_path = 'checkpoints/model_best.pt'
    
    # Data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/multi30k')
    max_length = 100
    batch_size = 64
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device):
    """Load model and tokenizers from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load tokenizers
    src_tokenizer = checkpoint.get('src_tokenizer', Tokenizer())
    tgt_tokenizer = checkpoint.get('tgt_tokenizer', Tokenizer())
    
    # Initialize model
    model = Transformer(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        d_model=512,  # Should match training config
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_length=Config.max_length,
        dropout=0.1
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, src_tokenizer, tgt_tokenizer

def evaluate_model(model, data_loader, tgt_tokenizer, device):
    """Evaluate model on the given data loader."""
    model.eval()
    
    all_predictions = []
    all_references = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_idx, reduction='sum')
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            
            # Forward pass
            output = model(src, trg[:, :-1])
            
            # Calculate loss
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
            
            # Generate predictions
            predictions = model.generate(src, max_length=Config.max_length)
            
            # Convert predictions and references to tokens
            for i in range(predictions.size(0)):
                pred_tokens = [t for t in predictions[i].tolist() if t not in [tgt_tokenizer.pad_idx, tgt_tokenizer.eos_idx, tgt_tokenizer.bos_idx]]
                ref_tokens = [t for t in trg[i].tolist() if t not in [tgt_tokenizer.pad_idx, tgt_tokenizer.eos_idx, tgt_tokenizer.bos_idx]]
                
                pred_sentence = tgt_tokenizer.decode(pred_tokens)
                ref_sentence = tgt_tokenizer.decode(ref_tokens)
                
                all_predictions.append(pred_sentence.split())
                all_references.append([ref_sentence.split()])  # Note: BLEU expects list of references
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader.dataset)
    bleu_score = calculate_bleu(all_references, all_predictions)
    rouge_scores = calculate_rouge(
        [' '.join(ref[0]) for ref in all_references],
        [' '.join(pred) for pred in all_predictions]
    )
    
    return {
        'loss': avg_loss,
        'bleu': bleu_score,
        'rouge': rouge_scores,
        'predictions': all_predictions,
        'references': [ref[0] for ref in all_references]
    }

def main():
    # Load model and tokenizers
    logger.info(f"Loading model from {Config.model_path}...")
    model, src_tokenizer, tgt_tokenizer = load_model(Config.model_path, Config.device)
    
    # Load test dataset
    logger.info("Loading test dataset...")
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
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluate_model(model, test_loader, tgt_tokenizer, Config.device)
    
    # Print and save results
    logger.info(f"Test Loss: {results['loss']:.4f}")
    logger.info(f"BLEU Score: {results['bleu']:.4f}")
    logger.info("ROUGE Scores:")
    for metric, scores in results['rouge'].items():
        logger.info(f"  {metric}: {scores['f']:.4f} (P: {scores['p']:.4f}, R: {scores['r']:.4f})")
    
    # Save detailed results
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'loss': results['loss'],
            'bleu': results['bleu'],
            'rouge': results['rouge']
        }, f, indent=2)
    
    # Save predictions and references
    with open(os.path.join(output_dir, 'predictions.txt'), 'w', encoding='utf-8') as f:
        for pred, ref in zip(results['predictions'], results['references']):
            f.write(f"PRED: {' '.join(pred)}\n")
            f.write(f"REF: {' '.join(ref)}\n\n")
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
