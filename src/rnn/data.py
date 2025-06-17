import glob
import os
import random
import string
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


def find_files(path: str) -> List[str]:
    return glob.glob(path)


def unicode_to_ascii(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def read_lines(filename: str) -> List[str]:
    lines = Path(filename).read_text(encoding="utf-8").strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]


def load_surname_data(
    data_path: str = "data/names/*.txt",
) -> Tuple[Dict[str, List[str]], List[str]]:
    category_lines: Dict[str, List[str]] = {}
    all_categories: List[str] = []

    for filename in find_files(data_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories


def letter_to_index(letter: str) -> int:
    return ALL_LETTERS.find(letter)


def line_to_tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_choice(items: List) -> any:
    return items[random.randint(0, len(items) - 1)]


def get_random_training_example(
    all_categories: List[str], category_lines: Dict[str, List[str]]
) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor(
        [all_categories.index(category)], dtype=torch.long
    )
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor 