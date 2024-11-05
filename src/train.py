# src/train/train.py
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from model import RotateLayer
from dataset import CIFARDataModule
from trainer import Trainer
from omegaconf import OmegaConf
from dataclasses import asdict

def setup_wandb_run(config):
    """Настраивает wandb run с правильной структурой директорий"""
    # Создаем базовую директорию для эксперимента
    experiment_dir = Path("experiments")
    experiment_dir.mkdir(exist_ok=True)
    
    # Создаем директории для разных типов артефактов
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    vis_dir = experiment_dir / "visualizations"
    
    for dir_path in [checkpoints_dir, logs_dir, vis_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Инициализируем wandb с указанием директории для артефактов
    config_dict = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(
        project=config.wandb.project,
        config=config_dict,
        dir=str(logs_dir)
    )
    
    return run, {
        'checkpoints': checkpoints_dir,
        'logs': logs_dir,
        'visualizations': vis_dir
    }


def train(config_path):
    # Загрузка конфигурации
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    run, dirs = setup_wandb_run(config)

    try:
        # Обновляем конфигурацию из wandb
        for key, value in wandb.config.items():
            if key in config:
                OmegaConf.update(config, key, value, merge=True)
        
        # Подготовка данных
        data_module = CIFARDataModule(config)
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        # Инициализация моделей
        device = torch.device(config.training.device)
        model = RotateLayer(config.model.params.initial_angle).to(device)
        target_model = RotateLayer(config.training.target_angle).to(device)
        
        # Установка целевого угла
        with torch.no_grad():
            target_model.theta.data = torch.tensor(config.training.target_angle).to(device)
        target_model.eval()
        
        # Логирование архитектуры модели
        wandb.watch(model, log='all')
        
        # Инициализация оптимизатора и функции потерь
        optimizer = getattr(optim, config.training.optimizer)(
            model.parameters(),
            lr=config.training.learning_rate
        )
        
        criterion = nn.MSELoss()
        
        # Инициализация тренера
        trainer = Trainer(model, target_model, optimizer, criterion, config)

        
       # Обучение
        best_val_loss = float('inf')
        for epoch in range(config.training.num_epochs):
            train_loss, val_loss = trainer.train_epoch(train_loader, val_loader, epoch)
            
            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer._save_checkpoint(val_loss, is_best=True)
            
            if (epoch + 1) % config.training.viz_interval == 0:
                trainer._save_checkpoint(val_loss, is_best=False)
        
        # Сохраняем финальные артефакты
        wandb.save(str(dirs['checkpoints'] / "*.pth"), base_path=str(dirs['checkpoints'].parent))
        wandb.save(str(dirs['visualizations'] / "*.png"), base_path=str(dirs['visualizations'].parent))
        
        return model
        
    finally:
        # Завершаем wandb run
        wandb.finish()


if __name__ == "__main__":
    config_path = Path("configs/config.yaml")
    model = train(config_path)