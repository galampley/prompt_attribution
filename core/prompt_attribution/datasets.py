from __future__ import annotations

"""Utility helpers for loading built-in prompt datasets used in tests and demos."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Resolve project root (…/workspace)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "tests" / "data"

@dataclass
class PromptCase:
    """A single prompt example with metadata.
    
    Attributes
    ----------
    id : str
        Short identifier (e.g. "P-01").
    style : str
        Qualitative label of the prompting style.
    text : str
        Raw prompt text to feed to the attribution engine.
    """

    id: str
    style: str
    text: str


def _load_json(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_mystery_prompts() -> List[PromptCase]:
    """Load the built-in set of 10 mystery prompts used for dog-food tests."""
    data_path = DATA_DIR / "mystery_prompts.json"
    records = _load_json(data_path)
    return [PromptCase(**rec) for rec in records]


__all__ = [
    "PromptCase",
    "load_mystery_prompts",
] 