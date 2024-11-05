from omegaconf import OmegaConf
from typing import Optional

def validate_config(config: OmegaConf) -> None:
    """Проверяет корректность конфигурации."""
    required_fields = {
        'model': ['name', 'params'],
        'training': ['batch_size', 'num_epochs', 'learning_rate', 'optimizer', 
                    'target_angle', 'log_interval', 'viz_interval', 'device'],
        'data': ['dataset', 'root_dir', 'train_batch_size', 'val_batch_size', 
                'num_workers'],
        'wandb': ['project']
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing section: {section}")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing field: {section}.{field}")

def load_and_validate_config(config_path: str) -> OmegaConf:
    """Загружает и проверяет конфигурацию."""
    config = OmegaConf.load(config_path)
    validate_config(config)
    return config