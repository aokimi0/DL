from typing import Tuple, Dict, List
import torch
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