import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms

from src.utils import setup_plotting


def get_mnist_loader(batch_size: int, data_dir: str = "data/mnist", return_dataset: bool = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
    if return_dataset:
        return dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def get_fashion_mnist_loader(batch_size: int, data_dir: str = "data/fashion_mnist", return_dataset: bool = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    if return_dataset:
        return dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader
