"""Ablation engine for testing segment impact."""

import asyncio
import time
import re
from typing import List, Dict, Optional, Union, Set, Any

from ..segmenter import Span
from ..settings import get_settings
from .llm_wrapper import LLMWrapper
from .run_manager import Run, RunManager, AblationResult


class AblationEngine:
    """Engine for performing ablation tests on prompt segments.
    
    Features:
    - Single-segment deletion strategy
    - Async batch processing with concurrency control
    - Cost guardrails to prevent API cost overruns
    """
    
    def __init__(self, llm: Optional[LLMWrapper] = None, run_manager: Optional[RunManager] = None):
        """Initialize the ablation engine.
        
        Args:
            llm: LLM wrapper instance
            run_manager: Run manager instance
        """
        self.llm = llm or LLMWrapper()
        self.run_manager = run_manager or RunManager()
        self.settings = get_settings()
        
        # Cost tracking
        self.estimated_cost_per_request = 0.003  # Rough estimate for GPT-4o
        self.max_cost = self.settings.max_cost_per_run
    
    def _remove_segment(self, prompt: str, span: Span) -> str:
        """Remove a segment from the prompt.
        
        Args:
            prompt: The full prompt
            span: The span to remove
            
        Returns:
            Prompt with the segment removed
        """
        return prompt[:span.start] + prompt[span.end:]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter based on punctuation."""
        # Split on ., !, ? followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]
    
    async def _process_segment(self, prompt: str, span: Span, scorer: Any,
                               baseline_sentences: List[str], baseline_embeddings: List[List[float]]) -> AblationResult:
        """Process a single segment ablation.
        
        Args:
            prompt: The full prompt
            span: The segment to ablate
            scorer: The scorer instance for calculating impact
            baseline_sentences: List of baseline sentences
            baseline_embeddings: List of baseline sentence embeddings
            
        Returns:
            Ablation result
        """
        start_time = time.time()
        
        # Create ablated prompt by removing the segment
        ablated_prompt = self._remove_segment(prompt, span)
        
        # Get completion for the ablated prompt
        ablated_completion = await self.llm.get_completion(ablated_prompt)
        
        # Split ablated completion into sentences and get embeddings
        ablated_sentences = self._split_sentences(ablated_completion)
        # Pad to length of baseline
        while len(ablated_sentences) < len(baseline_sentences):
            ablated_sentences.append("")
        
        embedding_tasks = []
        for s in ablated_sentences:
            if s:
                embedding_tasks.append(self.llm.get_embedding(s))
            else:
                embedding_tasks.append(None)
        # Gather embeddings for non-empty sentences
        embeddings_results = await asyncio.gather(*[t for t in embedding_tasks if t], return_exceptions=False)
        idx_res = 0
        ablated_embeddings = []
        for t in embedding_tasks:
            if t is None:
                ablated_embeddings.append([0.0])  # dummy vector for empty sentence
            else:
                ablated_embeddings.append(embeddings_results[idx_res])
                idx_res += 1
        
        # Compute per-sentence distances
        sentence_deltas = []
        for base_emb, abl_emb in zip(baseline_embeddings, ablated_embeddings):
            if len(abl_emb) == 1:  # dummy vector
                sentence_deltas.append(1.0)
            else:
                similarity = scorer._cosine_similarity(base_emb, abl_emb)
                sentence_deltas.append(1.0 - similarity)
        
        # Calculate impact using the scorer
        delta_cos = await scorer.calculate_distance(ablated_completion)
        
        # Calculate elapsed time
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return AblationResult(
            span_id=span.id,
            delta_cos=delta_cos,
            elapsed_ms=elapsed_ms,
            sentence_deltas=sentence_deltas
        )
    
    async def run_ablation_tests(
        self, 
        run: Run, 
        segments: List[Span] = None, 
        skip_ids: Set[int] = None,
        early_stop: bool = True,
        scorer: Optional[Any] = None,
    ) -> Run:
        """Run ablation tests for each segment.
        
        Args:
            run: The run to process
            segments: Segments to test (defaults to run's segments)
            skip_ids: IDs of segments to skip
            early_stop: Whether to enable early stopping
            scorer: Optional scorer instance (will be created if not provided)
            
        Returns:
            Updated run with ablation results
        """
        # Import here to avoid circular imports
        if scorer is None:
            from ..scorer import Scorer
            scorer = Scorer(run.completion, self.llm)
        
        prompt = run.prompt
        
        # Get segments if not provided
        if segments is None:
            segments = [
                Span(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    id=s["id"]
                )
                for s in run.segments
            ]
        
        # Filter segments to skip
        if skip_ids:
            segments = [s for s in segments if s.id not in skip_ids]
        
        # Check if projected cost exceeds limit
        segment_count = len(segments)
        projected_cost = segment_count * self.estimated_cost_per_request
        
        if projected_cost > self.max_cost:
            raise ValueError(
                f"Projected cost ${projected_cost:.2f} exceeds maximum allowed ${self.max_cost:.2f}. "
                f"Consider using hierarchical sampling to reduce costs."
            )
        
        # Process segments in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        cumulative_impact = 0.0
        early_stop_threshold = self.settings.early_stop_threshold
        
        # Precompute baseline sentences and embeddings
        baseline_sentences = self._split_sentences(run.completion)
        baseline_embeddings = await asyncio.gather(*[self.llm.get_embedding(s) for s in baseline_sentences])
        
        async def bounded_process(span):
            async with semaphore:
                return await self._process_segment(prompt, span, scorer,
                                                   baseline_sentences, baseline_embeddings)
        
        # Sort segments by ID to process in order
        sorted_segments = sorted(segments, key=lambda s: s.id)
        tasks = []
        
        for span in sorted_segments:
            # Add task to the list
            tasks.append(bounded_process(span))
            
            # Check cumulative impact for early stopping
            if (early_stop and 
                tasks and 
                cumulative_impact >= early_stop_threshold):
                print(f"Early stopping at {cumulative_impact:.2f} cumulative impact")
                break
        
        # Execute all tasks
        results = await asyncio.gather(*tasks)
        
        # Update run with results
        for result in results:
            cumulative_impact += result.delta_cos
            run = self.run_manager.add_ablation_result(run, result)
        
        # After gathering results compute controlling mapping
        num_sent = len(baseline_sentences)
        control = [-1] * num_sent
        max_scores = [-1.0] * num_sent
        for result in results:
            sid = result.span_id
            for idx, delta in enumerate(result.sentence_deltas):
                if idx < num_sent and delta > max_scores[idx]:
                    max_scores[idx] = delta
                    control[idx] = sid
        
        # Update run with control mapping
        run.response_control = control
        run.response_sentence_deltas = max_scores
        self.run_manager._save_run(run)
        
        return run 