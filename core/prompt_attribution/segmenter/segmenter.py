"""Segmenter for breaking prompts into analyzable chunks."""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Span:
    """A segment of text with start and end indices."""
    
    start: int
    end: int
    text: str
    id: Optional[int] = None


class Segmenter:
    """Split prompts into segments for ablation testing.
    
    The segmenter uses a heuristic approach:
    1. First tries to find markdown-style headings (e.g., ### Heading)
    2. Falls back to sliding window of approximately 40 tokens if no headings found
    """
    
    def __init__(self, window_size: int = 40, window_overlap: int = 5):
        """Initialize the segmenter.
        
        Args:
            window_size: Approximate number of tokens per window when no headings found
            window_overlap: Number of tokens to overlap between windows
        """
        self.window_size = window_size
        self.window_overlap = window_overlap
        
        # Heading pattern for markdown-style headers (e.g., ### Heading)
        self.heading_pattern = re.compile(r"^\s*(#{1,6})\s+(.+)$", re.MULTILINE)
        
        # Rough approximation of tokens = 4 chars
        self.chars_per_token = 4
    
    def segment(self, text: str) -> List[Span]:
        """Split the text into segments.
        
        Args:
            text: The prompt text to segment
            
        Returns:
            List of text spans
        """
        # First try to find markdown-style headings
        spans = self._segment_by_headings(text)
        
        # Fall back to sliding window if no headings found
        if not spans:
            spans = self._segment_by_window(text)
        
        # Add span IDs
        for i, span in enumerate(spans):
            span.id = i
            
        return spans
    
    def _segment_by_headings(self, text: str) -> List[Span]:
        """Segment the text by markdown-style headings.
        
        Args:
            text: The prompt text to segment
            
        Returns:
            List of text spans, empty if no headings found
        """
        matches = list(self.heading_pattern.finditer(text))
        if not matches:
            return []
        
        spans = []
        for i, match in enumerate(matches):
            start = match.start()
            # End is either the start of the next heading or the end of the text
            end = matches[i + 1].start() if i < len(matches) - 1 else len(text)
            span_text = text[start:end]
            spans.append(Span(start=start, end=end, text=span_text))
        
        # Check if there's content before the first heading
        if matches and matches[0].start() > 0:
            prefix_text = text[:matches[0].start()]
            spans.insert(0, Span(start=0, end=matches[0].start(), text=prefix_text))
        
        return spans
    
    def _segment_by_window(self, text: str) -> List[Span]:
        """Segment the text by sliding window.
        
        Args:
            text: The prompt text to segment
            
        Returns:
            List of text spans
        """
        spans = []
        chars_per_window = self.window_size * self.chars_per_token
        chars_overlap = self.window_overlap * self.chars_per_token
        
        start = 0
        while start < len(text):
            # Calculate end position, capping at text length
            end = min(start + chars_per_window, len(text))
            
            # Try to break at natural boundaries like periods or newlines
            if end < len(text):
                natural_breaks = [
                    text.rfind(". ", start, end),
                    text.rfind(".\n", start, end),
                    text.rfind("\n\n", start, end),
                ]
                natural_break = max(natural_breaks)
                if natural_break != -1 and natural_break > start + (chars_per_window // 2):
                    end = natural_break + 1  # Include the period
            
            span_text = text[start:end]
            spans.append(Span(start=start, end=end, text=span_text))
            
            # Move the start position, accounting for overlap
            start = end - chars_overlap if end < len(text) else len(text)
        
        return spans 