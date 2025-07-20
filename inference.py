import torch
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F
from transformer import Transformer

class Translator:
    def __init__(self, model_path, vocab_path, max_length=100):
        vocab_data = torch.load(vocab_path)
        self.src_vocab = vocab_data['src_vocab']
        self.tgt_vocab = vocab_data['tgt_vocab']
        
        self.src_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        self.tgt_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        
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
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.pad_idx = self.tgt_vocab['<pad>']
        self.sos_idx = self.tgt_vocab['<sos>']
        self.eos_idx = self.tgt_vocab['<eos>']
        self.max_length = max_length
    
    def preprocess_src(self, src_text):
        tokens = ['<sos>'] + self.src_tokenizer(src_text) + ['<eos>']
        tokens = tokens[:self.max_length-1]
        numerical = [self.src_vocab[token] for token in tokens]
        src_tensor = torch.LongTensor(numerical).unsqueeze(0).to(self.device)
        src_mask = self.make_src_mask(src_tensor)
        return src_tensor, src_mask
    
    def make_src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
    
    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        return (tgt_pad_mask & tgt_sub_mask).to(self.device)
    
    def greedy_decode(self, src, src_mask):
        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(self.sos_idx).type_as(src.data)
        
        for _ in range(self.max_length - 1):
            tgt_mask = self.make_tgt_mask(ys)
            out = self.model.decode(ys, memory, src_mask, tgt_mask)
            prob = self.model.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            
            if next_word == self.eos_idx:
                break
                
        return ys
    
    def postprocess(self, indices):
        tokens = [self.tgt_vocab.itos[idx] for idx in indices[0].tolist()]
        tokens = [t for t in tokens if t not in ['<sos>', '<eos>', '<pad>']]
        return ' '.join(tokens)
    
    def translate(self, src_text):
        src, src_mask = self.preprocess_src(src_text)
        tgt_tokens = self.greedy_decode(src, src_mask)
        return self.postprocess(tgt_tokens)

def load_model(model_path, vocab_path):
    return Translator(model_path, vocab_path)

def translate_interactive(translator):
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
    parser.add_argument('--model', type=str, default='checkpoints/model.pt', help='path to the trained model')
    parser.add_argument('--vocab', type=str, default='vocab.pt', help='path to the vocabulary file')
    
    args = parser.parse_args()
    translator = load_model(args.model, args.vocab)
    translate_interactive(translator)
