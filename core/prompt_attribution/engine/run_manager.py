"""Run manager for persisting attribution runs and results."""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..segmenter import Span
from ..settings import get_settings


@dataclass
class AblationResult:
    """Result of a single segment ablation test."""
    
    span_id: int
    delta_cos: float
    elapsed_ms: int
    # Per-sentence impact values (optional)
    sentence_deltas: List[float] = field(default_factory=list)


@dataclass
class Run:
    """Represents a full attribution run with results."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Input data
    prompt: str = ""
    completion: str = ""
    
    # Segmentation
    segments: List[Dict] = field(default_factory=list)
    
    # Results
    ablation_results: List[Dict] = field(default_factory=list)
    
    # Metadata
    settings: Dict = field(default_factory=dict)
    # Response sentence -> controlling segment mapping (index = sentence idx, value = segment id)
    response_control: List[int] = field(default_factory=list)
    response_sentence_deltas: List[float] = field(default_factory=list)
    
    # Rewrite suggestions - keyed by "seg{span_id}_sent{sentence_idx}"
    rewrite_suggestions: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_key_for_rewrite(self, span_id: int, sentence_idx: int) -> str:
        """Generate consistent key for rewrite suggestions dictionary.
        
        Args:
            span_id: ID of the prompt segment
            sentence_idx: Index of the response sentence
            
        Returns:
            String key for the rewrite_suggestions dictionary
        """
        return f"seg{span_id}_sent{sentence_idx}"
    
    def add_rewrite_suggestions(self, span_id: int, sentence_idx: int, suggestions: List[str]) -> None:
        """Add rewrite suggestions for a specific prompt segment and response sentence.
        
        Args:
            span_id: ID of the prompt segment
            sentence_idx: Index of the response sentence
            suggestions: List of rewrite suggestion strings
        """
        key = self.get_key_for_rewrite(span_id, sentence_idx)
        self.rewrite_suggestions[key] = suggestions


class RunManager:
    """Manages persistence of attribution runs."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the run manager.
        
        Args:
            base_dir: Base directory for storing runs
        """
        settings = get_settings()
        self.base_dir = Path(base_dir) if base_dir else Path("runs")
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_run(self, prompt: str, completion: str, segments: List[Span]) -> Run:
        """Create a new attribution run.
        
        Args:
            prompt: The input prompt
            completion: The model's baseline completion
            segments: List of prompt segments
            
        Returns:
            New Run object
        """
        # Create run object
        run = Run(
            prompt=prompt,
            completion=completion,
            segments=[asdict(segment) for segment in segments],
            settings={
                "completion_model": get_settings().completion_model,
                "embedding_model": get_settings().embedding_model,
            }
        )
        
        # Create run directory
        run_dir = self.base_dir / run.id
        os.makedirs(run_dir, exist_ok=True)
        
        # Save initial run data
        self._save_run(run)
        
        return run
    
    def add_ablation_result(self, run: Run, result: AblationResult) -> Run:
        """Add an ablation result to a run.
        
        Args:
            run: The run to update
            result: The ablation result to add
            
        Returns:
            Updated Run object
        """
        run.ablation_results.append(asdict(result))
        self._save_run(run)
        return run
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID.
        
        Args:
            run_id: Run ID to retrieve
            
        Returns:
            Run object if found, None otherwise
        """
        run_file = self.base_dir / run_id / "run.json"
        if not run_file.exists():
            return None
        
        try:
            with open(run_file, "r") as f:
                data = json.load(f)
                return Run(**data)
        except Exception:
            return None
    
    def _save_run(self, run: Run) -> None:
        """Save a run to disk.
        
        Args:
            run: Run to save
        """
        run_dir = self.base_dir / run.id
        run_file = run_dir / "run.json"
        
        # Ensure the run directory exists
        os.makedirs(run_dir, exist_ok=True)
        
        with open(run_file, "w") as f:
            json.dump(run.to_dict(), f, indent=2) 