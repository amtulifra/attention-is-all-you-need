import os
import torch
import argparse
from pathlib import Path

from src.models.transformer import Transformer
from src.data.tokenizer import Tokenizer
from src.utils import setup_logging

logger = setup_logging(log_dir='logs')

class Config:
    # Model
    model_path = 'checkpoints/model_best.pt'
    
    # Generation
    max_length = 100
    beam_size = 5
    length_penalty = 1.0
    temperature = 1.0
    
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
        dropout=0.0  # Disable dropout for inference
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, src_tokenizer, tgt_tokenizer

def translate_sentence(sentence, model, src_tokenizer, tgt_tokenizer, device, max_length=100):
    """Translate a single sentence."""
    model.eval()
    
    # Tokenize input
    tokens = src_tokenizer.encode(sentence)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]
    
    # Generate translation
    with torch.no_grad():
        output_tokens = model.generate(
            src_tensor,
            max_length=max_length,
            beam_size=Config.beam_size,
            length_penalty=Config.length_penalty,
            temperature=Config.temperature
        )
    
    # Convert tokens to text
    output_tokens = output_tokens[0].cpu().tolist()  # Get first (best) beam
    
    # Remove special tokens
    output_tokens = [t for t in output_tokens 
                    if t not in [tgt_tokenizer.pad_idx, 
                               tgt_tokenizer.bos_idx, 
                               tgt_tokenizer.eos_idx]]
    
    # Decode tokens to text
    translated_sentence = tgt_tokenizer.decode(output_tokens)
    
    return translated_sentence

def interactive_mode(model, src_tokenizer, tgt_tokenizer, device):
    """Run interactive translation mode."""
    print("\nInteractive translation mode (type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        try:
            # Get input sentence
            src_sentence = input("\nEnter source sentence: ").strip()
            
            if src_sentence.lower() in ['exit', 'quit']:
                break
                
            if not src_sentence:
                continue
            
            # Translate
            translation = translate_sentence(
                src_sentence, model, src_tokenizer, tgt_tokenizer, device, Config.max_length
            )
            
            # Print results
            print(f"\nSource: {src_sentence}")
            print(f"Translation: {translation}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            continue

def batch_mode(input_file, output_file, model, src_tokenizer, tgt_tokenizer, device):
    """Translate a batch of sentences from a file."""
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            src_sentences = [line.strip() for line in f if line.strip()]
        
        # Translate each sentence
        translations = []
        for sentence in tqdm(src_sentences, desc="Translating"):
            translation = translate_sentence(
                sentence, model, src_tokenizer, tgt_tokenizer, device, Config.max_length
            )
            translations.append(translation)
        
        # Write translations to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(f"{translation}\n")
        
        logger.info(f"Translated {len(translations)} sentences. Output saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in batch translation: {e}")
        raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Machine Translation Inference')
    parser.add_argument('--model', type=str, default=Config.model_path,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input file for batch translation')
    parser.add_argument('--output', type=str, default='translations.txt',
                       help='Path to output file for batch translation')
    parser.add_argument('--beam-size', type=int, default=Config.beam_size,
                       help='Beam size for beam search')
    parser.add_argument('--max-length', type=int, default=Config.max_length,
                       help='Maximum length of generated translations')
    parser.add_argument('--temperature', type=float, default=Config.temperature,
                       help='Temperature for sampling (lower = more greedy)')
    
    args = parser.parse_args()
    
    # Update config
    Config.model_path = args.model
    Config.beam_size = args.beam_size
    Config.max_length = args.max_length
    Config.temperature = args.temperature
    
    # Load model
    logger.info(f"Loading model from {Config.model_path}...")
    model, src_tokenizer, tgt_tokenizer = load_model(Config.model_path, Config.device)
    logger.info("Model loaded successfully!")
    
    # Run appropriate mode
    if args.input:
        # Batch translation mode
        logger.info(f"Translating sentences from {args.input}...")
        batch_mode(args.input, args.output, model, src_tokenizer, tgt_tokenizer, Config.device)
    else:
        # Interactive mode
        interactive_mode(model, src_tokenizer, tgt_tokenizer, Config.device)

if __name__ == '__main__':
    main()
