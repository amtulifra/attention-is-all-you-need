import os
import torch
import json
from pathlib import Path

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict,
    is_best: bool = False,
    checkpoint_dir: str = './checkpoints',
    filename: str = 'checkpoint.pth.tar',
    best_filename: str = 'model_best.pth.tar',
    config: dict = None,
    **kwargs
) -> str:
    """Save model checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(checkpoint, best_path)
    
    return checkpoint_path

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple:
    """Load model checkpoint from disk."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint at '{checkpoint_path}'")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        checkpoint,
        checkpoint.get('epoch', 0),
        checkpoint.get('train_loss', float('inf')),
        checkpoint.get('val_loss', float('inf')),
        checkpoint.get('metrics', {})
    )

def save_model(
    model: torch.nn.Module,
    model_dir: str = './models',
    model_name: str = 'model',
    config: dict = None,
    export_onnx: bool = False,
    onnx_input: torch.Tensor = None,
    **kwargs
) -> dict:
    """Save model weights and config to disk."""
    os.makedirs(model_dir, exist_ok=True)
    model_info = {}
    
    # Save PyTorch model
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path, **kwargs)
    model_info['pytorch'] = model_path
    
    # Export to ONNX if requested
    if export_onnx and onnx_input is not None:
        onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
        torch.onnx.export(
            model,
            onnx_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            **kwargs
        )
        model_info['onnx'] = onnx_path
    
    # Save config
    if config is not None:
        config_path = os.path.join(model_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        model_info['config'] = config_path
    
    return model_info

def load_model(
    model_path: str,
    model: torch.nn.Module = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple:
    """Load model weights from disk."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model at '{model_path}'")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model, config
    
    return None, config

def get_latest_checkpoint(checkpoint_dir: str, pattern: str = '*.pth.tar') -> str:
    """Get path to most recent checkpoint in directory."""
    checkpoint_files = list(Path(checkpoint_dir).glob(pattern))
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return str(checkpoint_files[0])
