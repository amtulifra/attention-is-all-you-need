import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from train import Config, DummyTranslationDataset, collate_fn
from torch.utils.data import DataLoader
from transformer import Transformer

nltk.download('punkt', quiet=True)

def calculate_bleu(model, data_loader, device):
    model.eval()
    references = []
    hypotheses = []
    attention_sources = []
    attention_weights_list = []
    smoothing = SmoothingFunction().method1
    
    with torch.no_grad():
        for src, tgt in tqdm(data_loader, desc="Calculating BLEU"):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            output, attn_weights = model(src=src, tgt=tgt_input, src_mask=None, tgt_mask=None, return_attn_weights=True)
            preds = output.argmax(dim=-1)
            
            for i in range(src.size(0)):
                src_tokens = src[i].tolist()
                tgt_tokens = tgt[i].tolist()
                pred_tokens = preds[i].tolist()
                
                src_tokens = [t for t in src_tokens if t not in [0, 1, 2]]
                tgt_tokens = [t for t in tgt_tokens if t not in [0, 1, 2]]
                pred_tokens = [t for t in pred_tokens if t not in [0, 1, 2]]
                
                references.append([tgt_tokens])
                hypotheses.append(pred_tokens)
                
                if len(attention_sources) < 3:
                    attention_sources.append((src_tokens, tgt_tokens, pred_tokens))
                    attention_weights_list.append(attn_weights if isinstance(attn_weights, tuple) else [attn_weights])
    
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        refs = [r for ref_list in ref for r in ref_list]
        score = sentence_bleu([refs], hyp, smoothing_function=smoothing)
        bleu_scores.append(score)
    
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    return avg_bleu, references, hypotheses, attention_sources, attention_weights_list

def plot_attention(attention_weights, src_tokens, tgt_tokens, layer=0, head=0):
    if isinstance(attention_weights, list):
        if layer >= len(attention_weights):
            return
        layer_weights = attention_weights[layer]
        
        if torch.is_tensor(layer_weights):
            layer_weights = layer_weights.detach().cpu().numpy()
            
        if len(layer_weights.shape) == 4:
            attn = layer_weights[0, head]
        elif len(layer_weights.shape) == 3:
            attn = layer_weights[head]
        else:
            return
    elif torch.is_tensor(attention_weights):
        weights = attention_weights.detach().cpu().numpy()
        if len(weights.shape) == 5:
            attn = weights[0, layer, head] if weights.shape[1] == 6 else weights[layer, 0, head]
        elif len(weights.shape) == 4:
            attn = weights[0, head]
        else:
            return
    else:
        return
        
    if len(attn.shape) > 2:
        attn = attn[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title(f'Attention Weights (Layer {layer}, Head {head})')
    
    os.makedirs('attention_plots', exist_ok=True)
    plt.savefig(f'attention_plots/attention_layer{layer}_head{head}.png')
    plt.close()

def evaluate_model(model_path, device='cuda'):
    print(f"Using device: {device}")
    print("Loading model and data...")
    
    model = Transformer(
        src_vocab_size=Config.VOCAB_SIZE,
        tgt_vocab_size=Config.VOCAB_SIZE,
        d_model=Config.D_MODEL,
        num_heads=Config.NUM_HEADS,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        d_ff=Config.D_FF,
        max_seq_length=Config.MAX_LENGTH,
        dropout=0.0
    ).to(device)
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # This is a full checkpoint with optimizer state, epoch, etc.
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # This is our training checkpoint format
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # This is just the raw state_dict (final_model.pt)
            model.load_state_dict(checkpoint)
            
        print(f"Successfully loaded model from {model_path}")
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        print("Trying to load just the state dict directly...")
        # Try to load the file as a direct state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    valid_dataset = DummyTranslationDataset(size=Config.VALID_SIZE, max_length=Config.MAX_LENGTH)
    valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn)
    
    print("Calculating BLEU score...")
    bleu, references, hypotheses, attention_sources, attention_weights_list = calculate_bleu(model, valid_loader, device)
    print(f"BLEU score: {bleu:.4f}")
    
    print("Generating attention visualizations...")
    for i, (src, tgt, pred) in enumerate(attention_sources[:2]):
        for layer in range(2):
            for head in range(4):
                plot_attention(attention_weights_list[i], src, tgt, layer=layer, head=head)
    
    print("Evaluation complete. Attention plots saved to 'attention_plots/' directory.")

if __name__ == '__main__':
    import argparse
    
    default_model = 'checkpoints/model_best.pt' if os.path.exists('checkpoints/model_best.pt') else 'checkpoints/final_model.pt'
    
    parser = argparse.ArgumentParser(description='Evaluate a trained Transformer model')
    parser.add_argument('--model', type=str, default=default_model,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run evaluation on')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Available checkpoints:")
        for file in os.listdir('checkpoints'):
            if file.endswith('.pt'):
                print(f"- checkpoints/{file}")
        exit(1)
        
    evaluate_model(args.model, args.device)
