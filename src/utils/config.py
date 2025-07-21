import os
import json
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load config from YAML or JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        raise ValueError(f"Unsupported format: {config_path.suffix}")

def save_config(
    config: dict, 
    config_path: str, 
    fmt: str = 'yaml',
    **kwargs
) -> None:
    """Save config to YAML or JSON file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if fmt.lower() in ('yaml', 'yml'):
            yaml.dump(config, f, default_flow_style=False, **kwargs)
        elif fmt.lower() == 'json':
            json.dump(config, f, indent=2, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge two config dicts."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def update_config(config: dict, updates: dict) -> dict:
    """Update config with new values using dot notation."""
    config = config.copy()
    
    for key_path, value in updates.items():
        keys = key_path.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return config

def get_config_value(config: dict, key_path: str, default=None):
    """Get value from nested config using dot notation."""
    current = config
    
    for key in key_path.split('.'):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current

def validate_config(config: dict, required_keys: list, name: str = 'config') -> None:
    """Check that config contains all required keys."""
    missing = []
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                missing.append(key_path)
                break
            current = current[key]
    
    if missing:
        raise ValueError(f"Missing required {name} keys: {', '.join(missing)}")
