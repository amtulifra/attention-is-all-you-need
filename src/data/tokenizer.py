import os
import json
from collections import Counter
import sentencepiece as spm

class Tokenizer:
    """SentencePiece-based tokenizer for Transformer models."""
    
    def __init__(self, model_path=None, vocab_size=8000, character_coverage=0.9995, model_type='unigram'):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.sp_model = None
        
        if model_path and os.path.exists(model_path):
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(model_path)
    
    def train(self, input_file, model_prefix, vocab_size=None, character_coverage=None, model_type=None):
        """Train a new SentencePiece model."""
        vocab_size = vocab_size or self.vocab_size
        character_coverage = character_coverage or self.character_coverage
        model_type = model_type or self.model_type
        
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,        # <pad>
            unk_id=3,        # <unk>
            bos_id=1,        # <bos>
            eos_id=2,        # <eos>
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<bos>',
            eos_piece='<eos>',
            user_defined_symbols=[],
            split_digits=True,
            byte_fallback=True,
            add_dummy_prefix=False,
            normalization_rule_name='nmt_nfkc',
        )
        
        self.model_path = f"{model_prefix}.model"
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(self.model_path)
    
    def encode(self, text):
        """Convert text to token IDs."""
        if not self.sp_model:
            raise ValueError("Tokenizer not initialized. Call train() or load an existing model.")
        return self.sp_model.encode_as_ids(text)
    
    def decode(self, ids):
        """Convert token IDs back to text."""
        if not self.sp_model:
            raise ValueError("Tokenizer not initialized. Call train() or load an existing model.")
        return self.sp_model.decode_ids(ids)
    
    def tokenize(self, text):
        """Split text into subword tokens."""
        if not self.sp_model:
            raise ValueError("Tokenizer not initialized. Call train() or load an existing model.")
        return self.sp_model.encode_as_pieces(text)
    
    def save_vocab(self, vocab_path):
        """Save vocabulary to a JSON file."""
        if not self.sp_model:
            raise ValueError("Tokenizer not initialized. Call train() or load an existing model.")
        
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.get_piece_size())}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_vocab(cls, vocab_path):
        """Load vocabulary from a JSON file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def build_vocab(sentences, max_vocab_size=30000, min_freq=2, special_tokens=None):
    """Build vocabulary from tokenized sentences."""
    counter = Counter()
    for tokens in sentences:
        counter.update(tokens)
    
    special_tokens = special_tokens or ['<pad>', '<bos>', '<eos>', '<unk>']
    vocab = {token: i for i, token in enumerate(special_tokens)}
    
    # Add most frequent tokens that meet min_freq
    for token, count in counter.most_common(max_vocab_size - len(special_tokens)):
        if count >= min_freq:
            vocab[token] = len(vocab)
    
    return vocab

def load_tokenizer(model_path):
    """Load a pre-trained tokenizer."""
    return Tokenizer(model_path=model_path)

def save_tokenizer(tokenizer, model_prefix):
    """Save tokenizer model and vocabulary."""
    if not tokenizer.sp_model:
        raise ValueError("Tokenizer not trained. Call train() first.")
    
    tokenizer.sp_model.save(f"{model_prefix}.model")
    tokenizer.save_vocab(f"{model_prefix}.vocab")
