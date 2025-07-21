import torch
from torch.utils.data import Dataset
import random

class TranslationDataset(Dataset):
    """
    Dataset for parallel translation data.
    """
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_length=100, train=True):
        """Initialize dataset with source and target files.
        
        Args:
            src_file: Path to source language file
            tgt_file: Path to target language file
            src_tokenizer: Source language tokenizer
            tgt_tokenizer: Target language tokenizer
            max_length: Maximum sequence length
            train: Whether this is training data (affects shuffling)
        """
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        self.train = train
        
        # Load and tokenize data
        self.src_sentences = []
        self.tgt_sentences = []
        
        # Read the files
        with open(src_file, 'r', encoding='utf-8') as f_src, open(tgt_file, 'r', encoding='utf-8') as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                # Store the raw text - we'll tokenize on the fly in __getitem__
                src_text = src_line.strip()
                tgt_text = tgt_line.strip()
                
                # Simple length check on whitespace tokens as a first pass
                # The actual tokenization will happen in __getitem__
                if len(src_text.split()) <= max_length - 2 and len(tgt_text.split()) <= max_length - 2:
                    self.src_sentences.append(src_text.split())  # Store as list of words for now
                    self.tgt_sentences.append(tgt_text.split())
        
        # Special token indices - these are fixed in the Tokenizer class
        self.pad_idx = 0  # <pad>
        self.bos_idx = 1  # <bos>
        self.eos_idx = 2  # <eos>
        self.unk_idx = 3  # <unk>
        
        if train:
            # Shuffle the data for training
            combined = list(zip(self.src_sentences, self.tgt_sentences))
            import random
            random.shuffle(combined)
            self.src_sentences, self.tgt_sentences = zip(*combined)
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        # Get the source and target sentences (already split into words)
        src_words = self.src_sentences[idx]
        tgt_words = self.tgt_sentences[idx]
        
        # Join words with spaces for tokenization
        src_text = ' '.join(src_words)
        tgt_text = ' '.join(tgt_words)
        
        try:
            # Convert text to token IDs using the tokenizers
            src_indices = self.src_tokenizer.encode(src_text)
            tgt_indices = self.tgt_tokenizer.encode(tgt_text)
            
            # Add BOS and EOS to target
            tgt_indices = [self.bos_idx] + tgt_indices + [self.eos_idx]
            
            # Truncate if necessary
            if len(src_indices) > self.max_length - 2:  # Account for BOS/EOS
                src_indices = src_indices[:self.max_length-2]
            if len(tgt_indices) > self.max_length:
                tgt_indices = tgt_indices[:self.max_length-1] + [self.eos_idx]
            
            return {
                'src': torch.tensor(src_indices, dtype=torch.long),
                'tgt': torch.tensor(tgt_indices, dtype=torch.long),
                'src_len': len(src_indices),
                'tgt_len': len(tgt_indices)
            }
            
        except Exception as e:
            print(f"Error processing example {idx}:")
            print(f"Source: {src_text}")
            print(f"Target: {tgt_text}")
            print(f"Error: {str(e)}")
            # Return a dummy example
            return {
                'src': torch.tensor([self.pad_idx], dtype=torch.long),
                'tgt': torch.tensor([self.bos_idx, self.eos_idx], dtype=torch.long),
                'src_len': 1,
                'tgt_len': 2
            }


class DummyTranslationDataset(Dataset):
    """Dummy dataset for testing with random sequences."""
    
    def __init__(self, vocab_size=100, seq_len=10, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # Special token indices
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequences with variable lengths
        src_len = random.randint(1, self.seq_len)
        tgt_len = random.randint(1, self.seq_len)
        
        # Generate random tokens (starting from 4 to avoid special tokens)
        src = torch.randint(4, self.vocab_size, (src_len,))
        tgt = torch.randint(4, self.vocab_size, (tgt_len,))
        
        # Add BOS and EOS to target
        tgt = torch.cat([
            torch.tensor([self.bos_idx]),
            tgt,
            torch.tensor([self.eos_idx])
        ])
        
        return {
            'src': src,
            'tgt': tgt,
            'src_len': src_len,
            'tgt_len': tgt_len + 2  # +2 for BOS and EOS
        }
