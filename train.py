import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformer import Transformer

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

# Configuration
class Config:
    # Data parameters
    VOCAB_SIZE = 1000  # Dummy vocabulary size
    BATCH_SIZE = 32
    MAX_LENGTH = 20    # Reduced for faster training
    
    # Model hyperparameters
    D_MODEL = 128      # Reduced for faster training
    NUM_HEADS = 4      # Reduced from 8
    NUM_ENCODER_LAYERS = 2  # Reduced from 6
    NUM_DECODER_LAYERS = 2  # Reduced from 6
    D_FF = 512        
    DROPOUT = 0.1
    
    # Training parameters
    NUM_EPOCHS = 3     # Reduced for demonstration
    LEARNING_RATE = 0.0001
    BETA1 = 0.9
    BETA2 = 0.98
    EPSILON = 1e-9
    CLIP = 1.0         # Gradient clipping
    
    # Paths
    SAVE_DIR = 'checkpoints'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model.pt')
    
    # Dataset size
    TRAIN_SIZE = 100
    VALID_SIZE = 20

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class DummyTranslationDataset(Dataset):
    def __init__(self, size=100, max_length=20):
        self.size = size
        self.max_length = max_length
        self.vocab_size = Config.VOCAB_SIZE
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random sequences for source and target with the same length
        seq_len = torch.randint(5, self.max_length - 2, (1,)).item()  # Leave room for SOS and EOS
        
        # 0: <sos>, 1: <eos>, 2: <pad>, 3+: regular tokens
        src = torch.randint(3, self.vocab_size, (seq_len,))
        tgt = torch.randint(3, self.vocab_size, (seq_len,))
        
        # Add special tokens
        src = torch.cat([torch.tensor([0]), src, torch.tensor([1])])  # [SOS] + tokens + [EOS]
        tgt = torch.cat([torch.tensor([0]), tgt, torch.tensor([1])])  # [SOS] + tokens + [EOS]
        
        return src, tgt

def collate_fn(batch):
    """Process and pad a batch of sequences"""
    src_batch, tgt_batch = zip(*batch)
    
    # Convert to tensors if they're not already
    src_batch = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in src_batch]
    tgt_batch = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in tgt_batch]
    
    # Pad sequences with pad_idx=2
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=2, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=2, batch_first=True)
    
    return src_batch, tgt_batch

def train_epoch(model, iterator, optimizer, criterion, clip):
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(tqdm(iterator, desc='Training', leave=False)):
        # Move tensors to the correct device
        src = src.to(device)  # [batch_size, src_len]
        tgt = tgt.to(device)  # [batch_size, tgt_len]
        
        # Print shapes for the first batch
        if batch_idx == 0:
            print(f"Source shape: {src.shape}, Target shape: {tgt.shape}")
        
        # Create target sequence with shifted right (teacher forcing)
        tgt_input = tgt[:, :-1]  # Remove last token for input [batch_size, tgt_len-1]
        tgt_output = tgt[:, 1:]  # Remove first token for target [batch_size, tgt_len-1]
        
        # Forward pass
        optimizer.zero_grad()
        
        # Let the model handle mask creation
        output = model(
            src=src,
            tgt=tgt_input
        )
        
        # Print shapes for the first batch
        if batch_idx == 0:
            print(f"Model output shape: {output.shape}, Expected target shape: {tgt_output.shape}")
        
        # Calculate loss (ignore padding tokens)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)
        
        # Only compute loss on non-padding tokens
        loss = criterion(output, tgt_output)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """Evaluate the model on validation set"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(iterator, desc='Evaluating', leave=False):
            # Move tensors to the correct device
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Create target sequence with shifted right (teacher forcing)
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]   # Remove first token for target
            
            # Forward pass (let the model handle mask creation)
            output = model(
                src=src,
                tgt=tgt_input
            )
            
            # Calculate loss (ignore padding tokens)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            
            # Only compute loss on non-padding tokens
            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    """Calculate elapsed time for an epoch"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    # Create datasets
    print("Creating datasets...")
    train_dataset = DummyTranslationDataset(size=Config.TRAIN_SIZE, max_length=Config.MAX_LENGTH)
    valid_dataset = DummyTranslationDataset(size=Config.VALID_SIZE, max_length=Config.MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Initialize model
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=Config.VOCAB_SIZE,
        tgt_vocab_size=Config.VOCAB_SIZE,
        d_model=Config.D_MODEL,
        num_heads=Config.NUM_HEADS,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        d_ff=Config.D_FF,
        max_seq_length=Config.MAX_LENGTH,
        dropout=Config.DROPOUT
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model initialized with {total_params:,} trainable parameters')
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=2)  # 2 is <pad> token
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        betas=(Config.BETA1, Config.BETA2),
        eps=Config.EPSILON
    )
    
    # Training loop
    print("Starting training...")
    best_valid_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.CLIP)
        valid_loss = evaluate(model, valid_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Save the model if validation loss has improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:02d}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    print("Training complete!")

if __name__ == '__main__':
    # Ensure the save directory exists
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    main()
