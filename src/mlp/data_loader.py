from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir: str = "data/mnist",
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    validation_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, validation_loader 