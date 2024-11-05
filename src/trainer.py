# src/train/trainer.py
from pathlib import Path
import torch
import wandb
from utils import visualize_rotation
import torch.optim.lr_scheduler as lr_scheduler

class Trainer:
    def __init__(self, model, target_model, optimizer, criterion, config):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = config.training.device
        self.best_loss = float('inf')
        
        # Инициализация scheduler
        self.scheduler = self._initialize_scheduler()

        # Создаем директорию для чекпойнтов
        self.checkpoint_dir = Path(config.training.save_folder, "checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        
    def _initialize_scheduler(self):
        if not self.config.training.get('scheduler') or not self.config.training.scheduler.name:
            return None
            
        scheduler_name = self.config.training.scheduler.name
        scheduler_params = self.config.training.scheduler.get('params', {})
        
        return getattr(lr_scheduler, scheduler_name)(self.optimizer, **scheduler_params)
    
    def train_epoch(self, train_loader, val_loader, epoch):
        self.model.train()
        epoch_loss = 0
        
        for i, (images, _) in enumerate(train_loader):
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.no_grad():
                target_images = self.target_model(images)
            
            rotated_images = self.model(images)
            loss = self.criterion(rotated_images, target_images)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Логирование батча
            if i % self.config.training.log_interval == 0:
                self._log_batch(loss.item(), i, epoch, len(train_loader))
                
            # Визуализация
            if i == 0 and (epoch + 1) % self.config.training.viz_interval == 0:
                self._visualize(images[0:1], rotated_images[0:1], epoch)
        
        train_loss = epoch_loss / len(train_loader)
        
        # Вычисляем validation loss
        val_loss = self.validate(val_loader)
        
        # Обновляем scheduler
        self._step_scheduler()
        
        # Логируем learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        wandb.log({
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })
        
        return train_loss, val_loss
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(self.device)
                target_images = self.target_model(images)
                rotated_images = self.model(images)
                loss = self.criterion(rotated_images, target_images)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _step_scheduler(self):
        if self.scheduler is not None:
                self.scheduler.step()
    
    def _log_batch(self, loss, batch_idx, epoch, num_batches):
        wandb.log({
            "batch_loss": loss,
            "batch_radian": self.model.theta.item(),
            "batch_angle": torch.rad2deg(self.model.theta),
            "batch": batch_idx + epoch * num_batches,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        })
    
    def _visualize(self, original, rotated, epoch):
        visualize_rotation(original, rotated, torch.rad2deg(self.model.theta).item(), epoch+1, self.config)
    
    def _save_checkpoint(self, loss, is_best=False):
        """
        Сохраняет чекпойнт модели с правильной структурой директорий для wandb
        """
        # Создаем словарь с состоянием
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
        }
        
        # Определяем имя файла
        filename = 'best_model.pth' if is_best else f'checkpoint_last.pth'
        checkpoint_path = self.checkpoint_dir / filename
        
        # Сохраняем чекпойнт
        torch.save(checkpoint, checkpoint_path)
        
        # Сохраняем в wandb с указанием базового пути
        wandb.save(
            str(checkpoint_path),
            base_path=str(self.checkpoint_dir.parent),
            policy='now'
        )

    def load_checkpoint(self, checkpoint_path):
        """
        Загружает состояние модели из чекпойнта
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['loss']