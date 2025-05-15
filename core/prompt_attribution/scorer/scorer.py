"""Scorer for calculating impact of segment ablations."""

import numpy as np
from typing import List, Dict, Optional, Union

from ..engine import LLMWrapper
from ..settings import get_settings


class Scorer:
    """Scorer for calculating the impact of segment ablations.
    
    Features:
    - Uses embeddings to calculate semantic similarity
    - Normalizes scores to a 0-1 scale
    - Tracks cumulative impact for early stopping
    """
    
    def __init__(self, baseline_completion: str, llm: Optional[LLMWrapper] = None):
        """Initialize the scorer.
        
        Args:
            baseline_completion: The baseline model output to compare against
            llm: LLM wrapper instance
        """
        self.baseline_completion = baseline_completion
        self.llm = llm or LLMWrapper()
        self.settings = get_settings()
        
        # Cache for baseline embedding
        self._baseline_embedding = None
        
        # Track cumulative impact for early stopping
        self.cumulative_impact = 0.0
        self.early_stop_threshold = self.settings.early_stop_threshold
    
    async def _get_baseline_embedding(self) -> List[float]:
        """Get the embedding for the baseline completion.
        
        Returns:
            Baseline embedding vector
        """
        if self._baseline_embedding is None:
            self._baseline_embedding = await self.llm.get_embedding(self.baseline_completion)
        return self._baseline_embedding
    
    async def calculate_distance(self, ablated_completion: str) -> float:
        """Calculate the normalized cosine distance between completions.
        
        Args:
            ablated_completion: The completion with a segment ablated
            
        Returns:
            Normalized distance score (0-1)
        """
        # Get embeddings
        baseline_embedding = await self._get_baseline_embedding()
        ablated_embedding = await self.llm.get_embedding(ablated_completion)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(baseline_embedding, ablated_embedding)
        
        # Convert to distance and normalize to 0-1 scale
        # 1.0 means maximum difference (high impact)
        # 0.0 means identical (no impact)
        distance = (1.0 - similarity)
        
        # Update cumulative impact
        self.cumulative_impact += distance
        
        return distance
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        a = np.array(a)
        b = np.array(b)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def should_early_stop(self) -> bool:
        """Check if early stopping threshold has been reached.
        
        Returns:
            True if early stopping should occur
        """
        return self.cumulative_impact >= self.early_stop_threshold
    
    def get_normalized_scores(self, results: List[Dict]) -> List[Dict]:
        """Normalize a list of ablation results.
        
        Args:
            results: List of ablation results
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return []
        
        # Extract scores
        scores = [r["delta_cos"] for r in results]
        
        # Find min and max for normalization
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        score_range = max_score - min_score
        if score_range == 0:
            # All scores identical; assign 0.0 normalized_score uniformly
            return [{**r, "normalized_score": 0.0} for r in results]
        
        # Normalize scores to 0-1
        normalized_results = []
        for result in results:
            result_copy = result.copy()
            result_copy["normalized_score"] = (result["delta_cos"] - min_score) / score_range
            normalized_results.append(result_copy)
        
        return normalized_results 