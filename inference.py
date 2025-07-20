import torch
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F
from transformer import Transformer
import math

class Translator:
    def __init__(self, model_path, vocab_path, max_length=100):
        # Load vocabularies
        vocab_data = torch.load(vocab_path)
        self.src_vocab = vocab_data['src_vocab']
        self.tgt_vocab = vocab_data['tgt_vocab']
        
        # Initialize tokenizers
        self.src_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.tgt_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        
        # Model configuration (should match training config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Transformer(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            max_seq_length=max_length,
            dropout=0.1
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Special tokens
        self.pad_idx = self.tgt_vocab['<pad>']
        self.sos_idx = self.tgt_vocab['<sos>']
        self.eos_idx = self.tgt_vocab['<eos>']
        
        self.max_length = max_length
    
    def preprocess_src(self, src_text):
        """Preprocess source text for the model"""
        tokens = ['<sos>'] + self.src_tokenizer(src_text) + ['<eos>']
        tokens = tokens[:self.max_length-1]
        
        # Convert tokens to indices
        numerical = [self.src_vocab[token] for token in tokens]
        
        # Pad sequence
        padding_length = self.max_length - len(numerical)
        numerical = numerical + [self.pad_idx] * padding_length
        
        return torch.LongTensor(numerical).unsqueeze(0).to(self.device)  # Add batch dimension
    
    def postprocess_tgt(self, indices):
        """Convert model output indices to text"""
        # Remove <sos> and everything after <eos>
        if self.eos_idx in indices:
            indices = indices[:indices.index(self.eos_idx)]
        
        # Convert indices to tokens
        tokens = [self.tgt_vocab.get_itos()[i] for i in indices]
        
        # Remove <sos> and <eos> if present
        tokens = [t for t in tokens if t not in ['<sos>', '<eos>', '<pad>']]
        
        return ' '.join(tokens)
    
    def translate(self, src_text, max_length=50, beam_size=5, length_penalty=0.6):
        """Translate source text using beam search"""
        # Preprocess source
        src = self.preprocess_src(src_text)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        
        # Encode source
        with torch.no_grad():
            memory = self.model.encode(src, src_mask)
        
        # Initialize beam search
        beam = [{
            'sequence': [self.sos_idx],
            'score': 0.0,
            'hidden': None,
            'finished': False
        }]
        
        # Beam search
        for _ in range(max_length):
            candidates = []
            
            for item in beam:
                if item['finished']:
                    candidates.append(item)
                    continue
                
                # Prepare decoder input
                tgt = torch.LongTensor([item['sequence']]).to(self.device)
                
                # Run decoder
                with torch.no_grad():
                    output = self.model.decode(
                        tgt, 
                        memory, 
                        src_mask,
                        create_mask(tgt, None, self.pad_idx).to(self.device)
                    )
                
                # Get log probabilities for the last token
                log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                
                # Get top k candidates
                topk_scores, topk_indices = torch.topk(log_probs, beam_size, dim=1)
                
                for i in range(beam_size):
                    token_idx = topk_indices[0, i].item()
                    score = item['score'] + topk_scores[0, i].item()
                    
                    # Apply length penalty
                    length = len(item['sequence']) + 1
                    score = score / (length ** length_penalty)
                    
                    # Check if sequence is finished
                    finished = (token_idx == self.eos_idx) or (len(item['sequence']) >= max_length - 1)
                    
                    candidates.append({
                        'sequence': item['sequence'] + [token_idx],
                        'score': score,
                        'hidden': None,  # Simplified for this example
                        'finished': finished
                    })
            
            # Keep top k candidates
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_size]
            
            # Check if all candidates are finished
            if all(candidate['finished'] for candidate in candidates):
                break
            
            beam = candidates
        
        # Get best translation
        best_translation = max(beam, key=lambda x: x['score'])
        return self.postprocess_tgt(best_translation['sequence'])

def load_model(model_path, vocab_path):
    """Load a trained model"""
    return Translator(model_path, vocab_path)

def translate_interactive(translator):
    """Interactive translation loop"""
    print("Enter 'q' to quit")
    print("-" * 50)
    
    while True:
        src_text = input("\nEnter German text: ")
        
        if src_text.lower() == 'q':
            break
        
        translation = translator.translate(src_text)
        print(f"\nTranslation: {translation}")
        print("-" * 50)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate German to English using a trained Transformer model')
    parser.add_argument('--model', type=str, default='checkpoints/model.pt',
                       help='path to the trained model')
    parser.add_argument('--vocab', type=str, default='vocab.pt',
                       help='path to the vocabulary file')
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = load_model(args.model, args.vocab)
    
    # Start interactive translation
    translate_interactive(translator)
