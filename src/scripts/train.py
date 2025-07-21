import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from src.models.transformer import Transformer
from src.data.dataset import TranslationDataset
from src.data.collate import collate_fn
from src.data.tokenizer import Tokenizer
from src.utils.metrics import calculate_bleu, calculate_rouge
from src.utils.visualization import plot_attention_weights
from src.utils import setup_logging

# Setup logging
logger = setup_logging(log_dir='logs')
writer = SummaryWriter('runs/transformer')

class Config:
    # Data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/multi30k')
    max_length = 100
    min_freq = 2
    
    # Model
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    dropout = 0.1
    
    # Training
    batch_size = 128
    num_epochs = 30
    learning_rate = 0.0001
    warmup_steps = 4000
    beta1 = 0.9
    beta2 = 0.98
    epsilon = 1e-9
    clip = 1.0
    
    # Checkpointing
    save_dir = 'checkpoints'
    checkpoint_freq = 1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        """Implement learning rate schedule."""
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()

def train_epoch(model, iterator, optimizer, criterion, clip, device, epoch):
    model.train()
    epoch_loss = 0
    
    progress = tqdm(iterator, desc=f'Epoch {epoch:02d} [Train]', leave=False)
    for i, batch in enumerate(progress):
        src = batch['src'].to(device)                  # [batch_size, src_len]
        tgt_input = batch['tgt_input'].to(device)      # [batch_size, tgt_len-1]
        tgt_output = batch['tgt_output'].to(device)    # [batch_size, tgt_len-1]
        
        optimizer.zero_grad()
        output = model(src, tgt_input)  # [batch_size, tgt_len-1, output_dim]
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)  # [batch_size * (tgt_len-1), output_dim]
        tgt_output = tgt_output.contiguous().view(-1)      # [batch_size * (tgt_len-1)]
        
        loss = criterion(output, tgt_output)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Log to tensorboard
        writer.add_scalar('train/loss', loss.item(), epoch * len(iterator) + i)
        
        # Update progress bar
        progress.set_postfix({'loss': f'{loss.item():.3f}'})
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc='Evaluating', leave=False):
            src = batch['src'].to(device)                  # [batch_size, src_len]
            tgt_input = batch['tgt_input'].to(device)      # [batch_size, tgt_len-1]
            tgt_output = batch['tgt_output'].to(device)    # [batch_size, tgt_len-1]
            
            output = model(src, tgt_input)  # [batch_size, tgt_len-1, output_dim]
            
            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)  # [batch_size * (tgt_len-1), output_dim]
            tgt_output = tgt_output.contiguous().view(-1)      # [batch_size * (tgt_len-1)]
            
            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    os.makedirs(Config.save_dir, exist_ok=True)
    torch.save(state, os.path.join(Config.save_dir, filename))
    if is_best:
        best_path = os.path.join(Config.save_dir, 'model_best.pt')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, filename='checkpoint.pt'):
    """Load model checkpoint if it exists, otherwise return default values."""
    checkpoint_path = os.path.join(Config.save_dir, filename)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=Config.device)
            
            # Load model state dict, ignoring mismatched keys
            model_state = checkpoint.get('state_dict', {})
            model.load_state_dict(model_state, strict=False)
            
            if optimizer is not None and 'optimizer' in checkpoint:
                try:
                    optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")
            
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch}, val_loss {best_val_loss:.4f})")
            
            return start_epoch, best_val_loss
            
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}. Starting fresh training.")
            return 0, float('inf')
    else:
        logger.info(f"No checkpoint found at '{checkpoint_path}'. Starting fresh training.")
        return 0, float('inf')

def main():
    # Create tokenizer directory
    tokenizer_dir = os.path.join(os.path.dirname(Config.data_dir), 'tokenizers')
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Initialize tokenizers
    logger.info("Initializing tokenizers...")
    
    # Source tokenizer (German)
    src_model_prefix = os.path.join(tokenizer_dir, 'de_tokenizer')
    src_tokenizer = Tokenizer()
    if not os.path.exists(f"{src_model_prefix}.model"):
        logger.info("Training source tokenizer...")
        src_tokenizer.train(
            input_file=os.path.join(Config.data_dir, 'train.de'),
            model_prefix=src_model_prefix,
            vocab_size=8000,
            model_type='bpe'
        )
    else:
        src_tokenizer = Tokenizer(f"{src_model_prefix}.model")
    
    # Target tokenizer (English)
    tgt_model_prefix = os.path.join(tokenizer_dir, 'en_tokenizer')
    tgt_tokenizer = Tokenizer()
    if not os.path.exists(f"{tgt_model_prefix}.model"):
        logger.info("Training target tokenizer...")
        tgt_tokenizer.train(
            input_file=os.path.join(Config.data_dir, 'train.en'),
            model_prefix=tgt_model_prefix,
            vocab_size=8000,
            model_type='bpe'
        )
    else:
        tgt_tokenizer = Tokenizer(f"{tgt_model_prefix}.model")
    
    # Load and process data
    logger.info("Loading datasets...")
    train_dataset = TranslationDataset(
        src_file=os.path.join(Config.data_dir, 'train.de'),
        tgt_file=os.path.join(Config.data_dir, 'train.en'),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=Config.max_length,
        train=True
    )
    
    val_dataset = TranslationDataset(
        src_file=os.path.join(Config.data_dir, 'val.de'),
        tgt_file=os.path.join(Config.data_dir, 'val.en'),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=Config.max_length,
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    # Get vocabulary sizes from tokenizers
    src_vocab_size = src_tokenizer.sp_model.get_piece_size()
    tgt_vocab_size = tgt_tokenizer.sp_model.get_piece_size()
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=Config.d_model,
        num_heads=Config.num_heads,
        num_encoder_layers=Config.num_encoder_layers,
        num_decoder_layers=Config.num_decoder_layers,
        d_ff=Config.d_ff,
        max_seq_length=Config.max_length,
        dropout=Config.dropout
    ).to(Config.device)
    
    # Initialize optimizer and criterion
    optimizer = NoamOpt(
        model.d_model,
        factor=1,
        warmup=Config.warmup_steps,
        optimizer=optim.Adam(
            model.parameters(),
            lr=0,
            betas=(Config.beta1, Config.beta2),
            eps=Config.epsilon,
            weight_decay=0.0001
        )
    )
    
    # Use fixed padding index (0) that matches our Tokenizer class
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is the padding index
    
    # Load checkpoint if exists
    start_epoch, best_val_loss = load_checkpoint(model, optimizer)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, Config.num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            Config.clip, Config.device, epoch
        )
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, criterion, Config.device)
        
        # Calculate epoch time
        epoch_mins, epoch_secs = divmod(time.time() - start_time, 60)
        
        # Log metrics
        logger.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f}')
        logger.info(f'\t Val. Loss: {val_loss:.3f}')
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'src_tokenizer': src_tokenizer,
            'tgt_tokenizer': tgt_tokenizer,
        }, is_best)
        
        # Log to tensorboard
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        
        # Save model at checkpoint frequency
        if (epoch + 1) % Config.checkpoint_freq == 0:
            checkpoint_path = os.path.join(Config.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'src_tokenizer': src_tokenizer,
                'tgt_tokenizer': tgt_tokenizer,
            }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(Config.save_dir, 'final_model.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'src_tokenizer': src_tokenizer,
        'tgt_tokenizer': tgt_tokenizer,
    }, final_model_path)
    logger.info(f"Training complete. Model saved to {final_model_path}")

if __name__ == '__main__':
    main()
