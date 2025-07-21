import math
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

class NoamOpt:
    """Optimizer wrapper implementing the Noam learning rate schedule."""
    
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
        """Calculate learning rate for current step."""
        step = step or self._step
        return self.factor * (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return {
            '_step': self._step,
            'warmup': self.warmup,
            'factor': self.factor,
            'model_size': self.model_size,
            '_rate': self._rate,
            'optimizer': self.optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == 'optimizer':
                self.optimizer.load_state_dict(value)
            else:
                setattr(self, key, value)

def get_optimizer(model, config):
    """Create and configure optimizer for the model."""
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('learning_rate', 0.0001)
    weight_decay = config.get('weight_decay', 0.0)
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adam':
        return Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer_name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler based on config."""
    scheduler_name = config.get('scheduler', 'noam').lower()
    
    if scheduler_name == 'noam':
        factor = config.get('warmup_factor', 1.0)
        warmup = config.get('warmup_steps', 4000)
        model_size = config.get('d_model', 512)
        return NoamOpt(model_size, factor, warmup, optimizer)
    
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=config.get('plateau_mode', 'min'),
            factor=config.get('plateau_factor', 0.1),
            patience=config.get('plateau_patience', 10),
            threshold=config.get('plateau_threshold', 1e-4),
            min_lr=config.get('min_lr', 1e-6),
            verbose=True
        )
    
    elif scheduler_name == 'linear':
        num_warmup_steps = config.get('warmup_steps', 4000)
        num_decay_steps = num_training_steps - num_warmup_steps
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_decay_steps)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    return LambdaLR(optimizer, lambda step: 1.0)  # Constant LR
