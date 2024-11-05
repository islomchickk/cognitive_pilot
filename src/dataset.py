from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

class CIFARDataModule:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def setup(self):
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.config.data.root_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        self.val_dataset = torchvision.datasets.CIFAR100(
            root=self.config.data.root_dir,
            train=False,
            download=True,
            transform=self.transform
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
