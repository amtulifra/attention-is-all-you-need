import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_idx=0):
    """Collate function for the DataLoader.
    
    Args:
        batch: List of samples from the dataset
        pad_idx: Padding token index
    """
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # Get sequence lengths (exclude EOS for target input)
    src_lengths = torch.tensor([len(src) for src in src_batch])
    tgt_lengths = torch.tensor([len(tgt) - 1 for tgt in tgt_batch])
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    # Create target input and output (shifted right by one)
    tgt_input = tgt_padded[:, :-1]  # Exclude last token
    tgt_output = tgt_padded[:, 1:]  # Exclude first token (BOS)
    
    # Create attention masks
    src_mask = (src_padded != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=tgt_input.device)
    
    return {
        'src': src_padded,           # [batch_size, src_len]
        'tgt_input': tgt_input,      # [batch_size, tgt_len-1]
        'tgt_output': tgt_output,    # [batch_size, tgt_len-1]
        'src_mask': src_mask,        # [batch_size, 1, 1, src_len]
        'tgt_mask': tgt_mask,        # [tgt_len-1, tgt_len-1]
        'src_lengths': src_lengths,  # [batch_size]
        'tgt_lengths': tgt_lengths,  # [batch_size]
    }

def generate_square_subsequent_mask(sz, device='cpu'):
    """Generate a square mask for the sequence to prevent attending to future positions."""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
