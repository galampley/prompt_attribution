"""Hierarchical segmentation for efficient prompt attribution."""

import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from functools import lru_cache

from .segmenter import Segmenter, Span


class HierarchicalSegmenter:
    """Two-pass segmentation for efficient prompt attribution.
    
    First pass: Divide the prompt into coarse chunks (20% of prompt)
    Second pass: Re-segment the highest-impact chunk into finer sub-windows (10%)
    
    This approach dramatically reduces the number of API calls needed while
    still providing high-resolution attribution for the most important sections.
    """
    
    def __init__(
        self,
        base_segmenter: Optional[Segmenter] = None,
        coarse_segment_ratio: float = 0.2,
        fine_segment_ratio: float = 0.1,
        cache_size: int = 128
    ):
        """Initialize the hierarchical segmenter.
        
        Args:
            base_segmenter: Base segmenter to use (creates new one if None)
            coarse_segment_ratio: Target size of coarse segments as ratio of total prompt
            fine_segment_ratio: Target size of fine segments as ratio of total prompt
            cache_size: Size of the segmenter's LRU cache
        """
        self.base_segmenter = base_segmenter or Segmenter()
        self.coarse_ratio = coarse_segment_ratio
        self.fine_ratio = fine_segment_ratio
        self._segment_cache = {}
        
        # Enable memoization
        self.segment_with_cache = lru_cache(maxsize=cache_size)(self._segment_uncached)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text.
        
        Args:
            text: Text to hash
            
        Returns:
            Cache key string
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _segment_uncached(self, text: str) -> List[Span]:
        """Segment the text without using the cache.
        
        Args:
            text: Text to segment
            
        Returns:
            List of spans
        """
        return self.base_segmenter.segment(text)
    
    def segment_first_pass(self, prompt: str) -> List[Span]:
        """Perform first-pass segmentation with coarse chunks.
        
        Args:
            prompt: The prompt text to segment
            
        Returns:
            List of coarse-grained segments
        """
        # Try to use any natural segments first
        segments = self.segment_with_cache(prompt)
        
        # If we have very few segments, just return them
        if len(segments) <= 5:
            return segments
        
        # If we have too many segments, create coarser segments
        target_segment_count = max(5, int(len(segments) * self.coarse_ratio))
        
        # Group segments into coarser chunks
        coarse_segments = []
        segments_per_group = max(1, len(segments) // target_segment_count)
        
        for i in range(0, len(segments), segments_per_group):
            group = segments[i:i + segments_per_group]
            if not group:
                continue
                
            start = group[0].start
            end = group[-1].end
            text = prompt[start:end]
            
            coarse_span = Span(
                start=start,
                end=end,
                text=text,
                id=len(coarse_segments)
            )
            coarse_segments.append(coarse_span)
        
        return coarse_segments
    
    def segment_second_pass(
        self, 
        prompt: str, 
        high_impact_span: Span,
        original_segments: List[Span]
    ) -> List[Span]:
        """Re-segment the highest-impact span from the first pass.
        
        Args:
            prompt: The full prompt text
            high_impact_span: The span with highest impact from first pass
            original_segments: The original segments from first pass
            
        Returns:
            List of fine-grained segments for the high-impact region
        """
        # Extract the high-impact region
        region_text = high_impact_span.text
        
        # Find any original segments contained within this region
        contained_segments = [
            s for s in original_segments 
            if s.start >= high_impact_span.start and s.end <= high_impact_span.end
        ]
        
        # If we have contained segments, use them
        if contained_segments:
            return contained_segments
        
        # Otherwise, create finer segments
        # Use natural segmentation first
        segments = self.segment_with_cache(region_text)
        
        # Adjust segment positions to account for the offset
        offset = high_impact_span.start
        for segment in segments:
            segment.start += offset
            segment.end += offset
        
        # If still too few segments, use sliding window with finer granularity
        if len(segments) <= 3:
            # Create a temporary segmenter with smaller window size
            fine_segmenter = Segmenter(
                window_size=20,  # Smaller window size for finer granularity
                window_overlap=5
            )
            segments = fine_segmenter.segment(region_text)
            
            # Adjust segment positions
            for segment in segments:
                segment.start += offset
                segment.end += offset
                segment.id = None  # Will be assigned later
        
        # Assign new IDs
        for i, segment in enumerate(segments):
            segment.id = i
        
        return segments
    
    def combine_results(
        self,
        first_pass_results: List[Dict],
        second_pass_results: List[Dict],
        all_segments: List[Span]
    ) -> List[Dict]:
        """Combine results from both passes into unified results.
        
        Args:
            first_pass_results: Results from first pass
            second_pass_results: Results from second pass
            all_segments: All segments (first pass + second pass refined segments)
            
        Returns:
            Combined attribution results
        """
        # Start with first pass results
        combined = first_pass_results.copy()
        
        # Add second pass results, mapping segment IDs to the correct range
        segment_ids = [s.id for s in all_segments]
        max_id = max(segment_ids) if segment_ids else 0
        
        for result in second_pass_results:
            result_copy = result.copy()
            result_copy["span_id"] = result_copy["span_id"] + max_id + 1
            combined.append(result_copy)
        
        return combined
    
    def get_first_pass_high_impact(
        self,
        segments: List[Span],
        results: List[Dict]
    ) -> Optional[Span]:
        """Find the highest-impact segment from the first pass.
        
        Args:
            segments: First-pass segments
            results: First-pass results
            
        Returns:
            Highest-impact span or None if no results
        """
        if not results:
            return None
        
        # Sort results by impact
        sorted_results = sorted(results, key=lambda r: r.get("delta_cos", 0), reverse=True)
        
        # Get the highest-impact segment
        high_impact_id = sorted_results[0]["span_id"]
        
        # Find the corresponding span
        for segment in segments:
            if segment.id == high_impact_id:
                return segment
        
        return None 