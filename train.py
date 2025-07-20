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

os.makedirs('checkpoints', exist_ok=True)

class Config:
    VOCAB_SIZE = 1000
    BATCH_SIZE = 32
    MAX_LENGTH = 20
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    D_FF = 512
    DROPOUT = 0.1
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.0001
    WARMUP_STEPS = 4000
    BETA1 = 0.9
    BETA2 = 0.98
    EPSILON = 1e-9
    CLIP = 1.0
    GRAD_ACCUM_STEPS = 4
    SAVE_DIR = 'checkpoints'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model.pt')
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pt')
    LOG_INTERVAL = 10
    TRAIN_SIZE = 1000
    VALID_SIZE = 100

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
        src_length = torch.randint(5, self.max_length, (1,)).item()
        tgt_length = torch.randint(5, self.max_length, (1,)).item()
        
        src = torch.randint(3, self.vocab_size, (src_length,))
        tgt = torch.randint(3, self.vocab_size, (tgt_length,))
        
        tgt = torch.cat([
            torch.tensor([1]),
            tgt,
            torch.tensor([2])
        ])
        
        return src, tgt

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    src_max_len = max(len(src) for src in src_batch)
    tgt_max_len = max(len(tgt) for tgt in tgt_batch)
    
    padded_src = torch.zeros(len(batch), src_max_len, dtype=torch.long)
    padded_tgt = torch.zeros(len(batch), tgt_max_len, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        padded_src[i, :len(src)] = src
        padded_tgt[i, :len(tgt)] = tgt
    
    return padded_src, padded_tgt

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
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
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in iterator:
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt[:, :-1])
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, os.path.join(Config.SAVE_DIR, filename))
    if is_best:
        best_path = os.path.join(Config.SAVE_DIR, 'model_best.pt')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, filename='checkpoint.pt'):
    checkpoint_path = os.path.join(Config.SAVE_DIR, filename)
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']}, val_loss {checkpoint.get('val_loss', 0):.4f})")
        
        return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, float('inf')

def main():
    train_dataset = DummyTranslationDataset(size=Config.TRAIN_SIZE, max_length=Config.MAX_LENGTH)
    valid_dataset = DummyTranslationDataset(size=Config.VALID_SIZE, max_length=Config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
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
    
    optimizer = NoamOpt(
        model.d_model,
        factor=1,
        warmup=Config.WARMUP_STEPS,
        optimizer=optim.Adam(
            model.parameters(),
            lr=0,
            betas=(Config.BETA1, Config.BETA2),
            eps=Config.EPSILON,
            weight_decay=0.0001
        )
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=2, reduction='mean')
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, 'checkpoint.pt')
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, Config.CLIP)
        val_loss = evaluate(model, valid_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f}')
        
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }, is_best)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, 'final_model.pt'))

if __name__ == '__main__':
    # Ensure the save directory exists
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    main()
