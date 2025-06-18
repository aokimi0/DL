import random
import string
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


def unicode_to_ascii(s: str) -> str:
    """Turn a Unicode string to plain ASCII."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def letter_to_tensor(letter: str) -> torch.Tensor:
    """Converts a single letter to a one-hot tensor."""
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][ALL_LETTERS.find(letter)] = 1
    return tensor


def line_to_tensor(line: str) -> torch.Tensor:
    """Converts a line (name) to a tensor of one-hot vectors."""
    tensor = torch.zeros(len(line), N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][ALL_LETTERS.find(letter)] = 1
    return tensor


class SurnameDataset(Dataset):
    """Dataset for surname classification."""

    def __init__(self, data_path: str = "data/names/*.txt"):
        self.all_categories, self.data, self.category_map = self._load_data(data_path)
        self.n_categories = len(self.all_categories)

    def _load_data(self, data_path: str) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, int]]:
        category_lines = {}
        all_categories = []
        
        for filename in sorted(Path().glob(data_path)):
            category = filename.stem
            all_categories.append(category)
            lines = [unicode_to_ascii(line) for line in filename.read_text(encoding="utf-8").strip().split("\n")]
            category_lines[category] = lines

        data = []
        for category, lines in category_lines.items():
            for line in lines:
                data.append((line, category))
        
        category_map = {category: i for i, category in enumerate(all_categories)}
        return all_categories, data, category_map

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        line, category = self.data[idx]
        line_tensor = line_to_tensor(line)
        category_tensor = torch.tensor([self.category_map[category]], dtype=torch.long)
        return line_tensor, category_tensor, len(line)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates data into batches, padding sequences to the same length."""
    lines, categories, lengths = zip(*batch)
    
    # Pad line tensors
    lines_padded = pad_sequence(lines, batch_first=True, padding_value=0)
    
    # Concatenate category tensors
    categories_tensor = torch.cat(categories)
    
    # Create tensor of sequence lengths
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return lines_padded, categories_tensor, lengths_tensor


def category_from_output(output: torch.Tensor, all_categories: List[str]) -> Tuple[str, int]:
    """Helper function to get category from model output."""
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def get_random_training_example(
    all_categories: List[str], category_lines: Dict[str, List[str]]
) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
    """Gets a random training example for manual models."""
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # The old line_to_tensor had an extra dimension, which manual models expect
    line_tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        line_tensor[i][0][ALL_LETTERS.find(letter)] = 1
    return category, line, category_tensor, line_tensor

def load_surname_data_for_manual_models() -> Tuple[Dict[str, List[str]], List[str]]:
    """Loads data in the old format required by manual models."""
    category_lines: Dict[str, List[str]] = {}
    all_categories: List[str] = []

    for filename in sorted(Path().glob("data/names/*.txt")):
        category = filename.stem
        all_categories.append(category)
        lines = [unicode_to_ascii(line) for line in filename.read_text(encoding="utf-8").strip().split("\n")]
        category_lines[category] = lines

    return category_lines, all_categories 