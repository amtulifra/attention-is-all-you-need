import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformer import Transformer
import numpy as np
from datetime import datetime

# Set up directories
os.makedirs('checkpoints', exist_ok=True)

# Model and training configuration
class Config:
    # Data settings
    VOCAB_SIZE = 1000  # Size of the vocabulary (tokens)
    BATCH_SIZE = 32    # Number of samples per batch
    MAX_LENGTH = 20    # Maximum sequence length
    
    # Model architecture
    D_MODEL = 128           # Dimension of embeddings and hidden states
    NUM_HEADS = 4           # Number of attention heads
    NUM_ENCODER_LAYERS = 2  # Number of encoder layers
    NUM_DECODER_LAYERS = 2  # Number of decoder layers
    D_FF = 512        
    DROPOUT = 0.1
    
    # Training settings
    NUM_EPOCHS = 3        # Total training epochs
    LEARNING_RATE = 0.0001  # Base learning rate
    WARMUP_STEPS = 4000    # Warmup steps for learning rate
    BETA1 = 0.9           # Adam beta1
    BETA2 = 0.98          # Adam beta2
    EPSILON = 1e-9        # Adam epsilon
    CLIP = 1.0            # Gradient clipping threshold
    GRAD_ACCUM_STEPS = 4  # Accumulate gradients over multiple steps
    
    # Checkpoint paths
    SAVE_DIR = 'checkpoints'  # Directory to save checkpoints
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model.pt')  # Latest model
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pt')  # Best model
    
    # Logging settings
    LOG_INTERVAL = 10  # Log stats every N batches
    
    # Dataset size
    TRAIN_SIZE = 100
    VALID_SIZE = 20

# Set device (GPU if available, else CPU)
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
    """Process and pad a batch of sequences.
    
    Args:
        batch: List of (src, tgt) tuples where each is a sequence tensor
        
    Returns:
        Tuple of (padded_src, padded_tgt) tensors
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Convert to tensors if they're not already
    src_batch = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in src_batch]
    tgt_batch = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in tgt_batch]
    
    # Pad sequences with pad_idx=2
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=2, batch_first=True)
    
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=2, batch_first=True)
    
    return src_batch, tgt_batch

class NoamOpt:
    """Optimizer wrapper implementing the Noam learning rate schedule.
    
    Implements the learning rate schedule from the "Attention Is All You Need" paper.
    The learning rate increases linearly for the first warmup_steps, then decreases
    proportionally to the inverse square root of the step number.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """Update parameters and learning rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        """Calculate learning rate with warmup.
        
        Implements the learning rate schedule from the "Attention Is All You Need" paper.
        The learning rate increases linearly for the first warmup_steps, then decreases
        proportionally to the inverse square root of the step number.
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(model):
    return NoamOpt(model.d_model, 2, Config.WARMUP_STEPS,
                  optim.Adam(model.parameters(), lr=0, betas=(Config.BETA1, Config.BETA2), eps=Config.EPSILON))

def train_epoch(model, iterator, optimizer, criterion, epoch, clip=1.0):
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        iterator: DataLoader for training data
        optimizer: Optimizer with learning rate scheduling
        criterion: Loss function
        epoch: Current epoch number (for logging)
        clip: Gradient clipping threshold
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0
    total_steps = len(iterator)
    
    # Initialize progress bar
    pbar = tqdm(enumerate(iterator), total=total_steps, desc=f'Epoch {epoch+1:02d}')
    
    for batch_idx, (src, tgt) in pbar:
        # Move tensors to the correct device
        src = src.to(device)  # [batch_size, src_len]
        tgt = tgt.to(device)  # [batch_size, tgt_len]
        
        # Create target sequence with shifted right (teacher forcing)
        tgt_input = tgt[:, :-1]  # Remove last token for input [batch_size, tgt_len-1]
        tgt_output = tgt[:, 1:]   # Remove first token for target [batch_size, tgt_len-1]
        
        # Forward pass
        output = model(src=src, tgt=tgt_input)
        
        # Calculate loss (ignore padding tokens)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)
        
        loss = criterion(output, tgt_output) / Config.GRAD_ACCUM_STEPS
        loss.backward()
        
        # Update parameters every GRAD_ACCUM_STEPS batches
        if (batch_idx + 1) % Config.GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == total_steps:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # Update parameters and learning rate
            optimizer.step()
            optimizer.zero_grad()
        
        # Keep track of loss
        epoch_loss += loss.item() * Config.GRAD_ACCUM_STEPS
        
        # Update progress bar
        if (batch_idx + 1) % Config.LOG_INTERVAL == 0 or (batch_idx + 1) == total_steps:
            pbar.set_postfix({
                'loss': f'{loss.item() * Config.GRAD_ACCUM_STEPS:.3f}',
                'lr': f'{optimizer.rate():.6f}'
            })
    
    return epoch_loss / total_steps

def evaluate(model, iterator, criterion):
    """Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate
        iterator: DataLoader for validation data
        criterion: Loss function
        
    Returns:
        Average validation loss per token
    """
    model.eval()
    epoch_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        pbar = tqdm(iterator, desc='Evaluating', leave=False)
        for batch_idx, (src, tgt) in enumerate(pbar):
            # Move tensors to the correct device
            src = src.to(device)  # [batch_size, src_len]
            tgt = tgt.to(device)  # [batch_size, tgt_len]
            
            # Create target sequence with shifted right (teacher forcing)
            tgt_input = tgt[:, :-1]  # Remove last token for input [batch_size, tgt_len-1]
            tgt_output = tgt[:, 1:]  # Remove first token for target [batch_size, tgt_len-1]
            
            # Forward pass
            output = model(src=src, tgt=tgt_input)
            
            # Calculate loss (ignore padding tokens)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            
            # Calculate loss only for non-padding tokens
            loss = criterion(output, tgt_output)
            
            # Calculate number of non-padding tokens for proper averaging
            non_pad_mask = tgt_output != 2  # 2 is the padding index
            num_tokens = non_pad_mask.sum().item()
            
            epoch_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Update progress bar
            pbar.set_postfix({'val_loss': f'{loss.item():.3f}'})
    
    # Return average loss per token
    return epoch_loss / total_tokens if total_tokens > 0 else float('inf')

def epoch_time(start_time, end_time):
    """Calculate elapsed time in minutes and seconds.
    
    Args:
        start_time: Start time from time.time()
        end_time: End time from time.time()
        
    Returns:
        Tuple of (minutes, seconds)
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """Save model checkpoint.
    
    Args:
        state: Dictionary containing model state
        is_best: Whether this is the best model so far
        filename: Name for the checkpoint file
    """
    torch.save(state, os.path.join(Config.SAVE_DIR, filename))
    if is_best:
        best_path = os.path.join(Config.SAVE_DIR, 'model_best.pt')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, filename='checkpoint.pt'):
    """Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filename: Checkpoint filename
        
    Returns:
        Tuple of (start_epoch, best_val_loss)
    """
    checkpoint_path = os.path.join(Config.SAVE_DIR, filename)
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        return start_epoch, best_val_loss
    return 0, float('inf')

def main():
    """Main training function.
    
    Handles dataset creation, model initialization, training loop,
    validation, and checkpointing.
    """
    print("Creating datasets...")
    # Initialize datasets with random data
    train_dataset = DummyTranslationDataset(size=Config.TRAIN_SIZE, max_length=Config.MAX_LENGTH)
    valid_dataset = DummyTranslationDataset(size=Config.VALID_SIZE, max_length=Config.MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, 
                            collate_fn=collate_fn)
    
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
    
    # Count trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} trainable parameters")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=2)  # Ignore padding index
    
    # Use Noam optimizer with warmup
    optimizer = get_std_opt(model)
    
    # Load checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Create a summary writer for TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S')))
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, Config.CLIP)
        
        # Evaluate on validation set
        val_loss = evaluate(model, valid_loader, criterion)
        
        # Calculate epoch time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.rate(), epoch)
        
        # Print progress
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):.3f}')
        
        # Checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'val_loss': val_loss,
            'train_loss': train_loss,
        }, is_best, f'checkpoint_epoch_{epoch}.pt')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, 'final_model.pt'))
    writer.close()

if __name__ == '__main__':
    # Ensure the save directory exists
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    main()
