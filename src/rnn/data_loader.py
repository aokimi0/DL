from typing import Tuple, Dict, List
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import unicodedata
import string
from pathlib import Path

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

def find_files(path: str) -> List[Path]:
    return list(Path(path).glob("*.txt"))

def unicode_to_ascii(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )

def read_lines(filename: Path) -> List[str]:
    lines = filename.read_text(encoding="utf-8").strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]

def get_surname_data(data_dir: str = "data/names") -> Tuple[Dict[str, List[str]], List[str]]:
    category_lines: Dict[str, List[str]] = {}
    all_categories: List[str] = []

    for filename in find_files(data_dir):
        category = filename.stem
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines
    
    return category_lines, all_categories

def letter_to_index(letter: str) -> int:
    return ALL_LETTERS.find(letter)

def line_to_tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][letter_to_index(letter)] = 1
    return tensor

def get_mnist_loaders(
    batch_size: int,
    data_dir: str = "data/mnist",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    validation_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader
