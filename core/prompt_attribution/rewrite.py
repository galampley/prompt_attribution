"""Rewrite suggestions for prompt segments based on attribution results.

This module provides tools to generate suggested rewrites for prompt segments 
that have been identified as influencing problematic parts of an LLM's response.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

from .engine.llm_wrapper import LLMWrapper
from .settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RewriteRequest:
    """Request for a prompt segment rewrite suggestion."""
    
    # Core identifiers
    span_id: int
    sentence_idx: int
    
    # Content
    span_text: str
    sentence_text: str
    full_prompt: str
    full_response: str
    
    # Influence data
    delta_cos: float
    normalized_score: float
    sentence_delta: float
    
    # Optional
    user_comment: str = ""
    secondary_influences: List[Dict[str, Any]] = None
    matrix_snippet: str = ""
    full_matrix_json: str = ""  # Add field for full matrix in JSON format
    
    def __post_init__(self):
        """Initialize default values."""
        if self.secondary_influences is None:
            self.secondary_influences = []


async def generate_rewrite(request: RewriteRequest, llm: Optional[LLMWrapper] = None) -> List[str]:
    """Generate rewrite suggestions for a prompt segment.
    
    Args:
        request: The rewrite request with all context
        llm: Optional LLM wrapper to use (creates one if None)
        
    Returns:
        List of suggested rewrites for the segment
    """
    if llm is None:
        llm = LLMWrapper()
    
    # Print debug info about what we're working with
    # print(f"\n=== Rewrite Request Details ===")
    # print(f"Span ID: {request.span_id}")
    # print(f"Span Text: {request.span_text[:100]}...")
    # print(f"Sentence Text: {request.sentence_text[:100]}...")
    # print(f"User Comment: {request.user_comment}")
    # print(f"Full Matrix JSON: {request.full_matrix_json}")
    # print(f"Delta Cos: {request.delta_cos}")
    # print("=============================\n")
    
    # Create system and user prompts for the meta-LLM
    system_prompt = """You are a prompt engineering expert. You directly rewrite prompts and/or segments of prompts to improve LLM outputs."""
    
    user_prompt = f"""
    # Context
    ## Original prompt segment
    - <original_prompt_snippet> is the original prompt segment that needs rewriting

    <original_prompt_snippet>
    {request.span_text}
    </original_prompt_snippet>

    ## Problematic sentence
    - <problematic_portion> is the portion of the LLM output that is problematic
    - This is not what needs rewritten, but what needs to be fixed as a product of the prompt rewrite

    <problematic_portion>
    "{request.sentence_text}"
    </problematic_portion>

    ## User comment
    - <user_comment> is the user's comment on what's wrong with and/or what they would like to see in <problematic_sentence>

    <user_comment>
    {request.user_comment}
    </user_comment>

    ## Full prompt
    - <full_prompt> is the full prompt that was used to generate the LLM output
    
    <full_prompt>
    {request.full_prompt}
    </full_prompt>

    ## Full LLM Output
    - <full_llm_output> is the full LLM output that was generated
    
    <full_llm_output>
    {request.full_response}
    </full_llm_output>

    ## Influence matrix
    - <influence_matrix> is the full influence matrix data
    - Higher values indicate stronger influence between a segment and sentence
    - "segments" contains all prompt segments with IDs and text previews
    - "sentences" contains all response sentences with IDs and text previews
    - "values" shows the influence score for each segment-sentence pair
    - Focus on segments with high influence scores for the problematic sentence
    - Consider how your rewrite might affect other sentences too
    - The value for segment {request.span_id} and sentence {request.sentence_idx} is {request.delta_cos:.4f}

    <influence_matrix>
    {request.full_matrix_json}
    </influence_matrix>

    # Output format
    - Your output should be a JSON object with the following fields
    
    <output_format>
    {{
        "1": {{
            "rewrite": "<rewrite suggestion 1>",
            "explanation": "<explanation of why this rewrite helps, mention 'Influence matrix' and 'User comment'>"
        }},
        "2": {{
            "rewrite": "<rewrite suggestion 2>",
            "explanation": "<explanation of why this rewrite helps, mention 'Influence matrix' and 'User comment'>"
        }}
    }}
    </output_format>

    # Requirements
    - Don't output <output_format> in your response
    - Don't preface your output with ```json
    """
    
    # Call the LLM to get suggestions
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Debug print to show what's being sent to the LLM
    print("\n=== Messages being sent to LLM ===")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("===================================\n")
    
    try:
        response = await llm.get_chat_completion(messages)
        
        # Extract suggestions (one per line)
        suggestions = [line.strip() for line in response.strip().split("\n") if line.strip()]
        
        # Filter out any lines that look like explanations rather than rewrites
        suggestions = [s for s in suggestions if not s.startswith(("Here", "Option", "Rewrite", "Suggestion"))]
        
        return suggestions
    except Exception as e:
        logger.error(f"Error generating rewrite suggestions: {e}")
        return [f"Error generating suggestions: {str(e)}"]


def format_matrix_snippet(matrix, span_id: int, sentence_idx: int) -> str:
    """Format a small section of the influence matrix around the point of interest.
    
    Args:
        matrix: 2D list of influence values
        span_id: The span ID to center on
        sentence_idx: The sentence index to center on
        
    Returns:
        ASCII art representation of the matrix snippet
    """
    if not matrix or not matrix[0]:
        return ""
    
    # Get matrix dimensions
    n_spans = len(matrix)
    n_sentences = len(matrix[0])
    
    # Calculate view window (+-2 rows/cols around the point of interest)
    row_start = max(0, span_id - 2)
    row_end = min(n_spans, span_id + 3)
    col_start = max(0, sentence_idx - 2)
    col_end = min(n_sentences, sentence_idx + 3)
    
    # Build header
    result = "    " + " ".join(f"S{i:<2}" for i in range(col_start, col_end)) + "\n"
    result += "   +" + "-" * (4 * (col_end - col_start) + 1) + "\n"
    
    # Build rows
    for i in range(row_start, row_end):
        result += f"P{i:<2}|"
        for j in range(col_start, col_end):
            marker = "*" if i == span_id and j == sentence_idx else " "
            val = matrix[i][j]
            result += f"{val:.2f}{marker}"
        result += "\n"
    
    return result 


def format_full_matrix_json(matrix, segments, sentences) -> str:
    """Format the full influence matrix as JSON.
    
    Args:
        matrix: 2D list of influence values
        segments: List of prompt segments
        sentences: List of response sentences
        
    Returns:
        JSON string representation of the matrix with segment and sentence info
    """
    if not matrix or not matrix[0]:
        return "{}"
    
    matrix_data = {
        "segments": [],
        "sentences": [],
        "values": []
    }
    
    # Add segment info
    for i, seg in enumerate(segments):
        if isinstance(seg, dict):
            # Handle when segments are passed as dicts
            matrix_data["segments"].append({
                "id": seg.get("id", i),
                "preview": seg.get("text", "")[:50] + ("..." if len(seg.get("text", "")) > 50 else "")
            })
        else:
            # Handle when segments are Span objects
            matrix_data["segments"].append({
                "id": getattr(seg, "id", i),
                "preview": getattr(seg, "text", "")[:50] + ("..." if len(getattr(seg, "text", "")) > 50 else "")
            })
    
    # Add sentence info
    for i, sentence in enumerate(sentences):
        matrix_data["sentences"].append({
            "id": i,
            "preview": sentence[:50] + ("..." if len(sentence) > 50 else "")
        })
    
    # Add matrix values
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix_data["values"].append({
                "segment_id": i,
                "sentence_id": j,
                "value": matrix[i][j]
            })
    
    return json.dumps(matrix_data, indent=2)


async def generate_rewrites_for_run(
    run: Any,  # Avoiding circular import of Run type
    sentence_idx: int,
    user_comment: str = "",
    top_k_segments: int = 1,
    model: str = None  # Add model parameter
) -> Dict[int, List[str]]:
    """Generate rewrites for span(s) influencing a specific response sentence.
    
    Args:
        run: The Run object with ablation results
        sentence_idx: Index of the problematic sentence
        user_comment: Optional user explanation of what's wrong
        top_k_segments: Number of top segments to generate rewrites for
        model: Optional model override to use for rewrite generation
        
    Returns:
        Dict mapping span_id to list of rewrite suggestions
    """
    if not run.ablation_results or not hasattr(run, "response_control"):
        return {}
    
    # Make sure sentence index is valid
    if sentence_idx < 0 or sentence_idx >= len(run.response_control):
        return {}
    
    # Get the span ID(s) that most influence this sentence
    primary_span_id = run.response_control[sentence_idx]
    if primary_span_id < 0:
        return {}  # No controlling span found
    
    # Split response into sentences to get this specific one
    from re import split as re_split
    sentences = re_split(r'(?<=[.!?])\s+', run.completion.strip())
    if sentence_idx >= len(sentences):
        return {}
    sentence_text = sentences[sentence_idx]
    
    # Find the span text
    span_text = None
    for segment in run.segments:
        if segment["id"] == primary_span_id:
            span_text = segment["text"]
            break
    
    if not span_text:
        return {}
    
    # Find influence data
    delta_cos = 0.0
    normalized_score = 0.0
    sentence_delta = 0.0 if not hasattr(run, "response_sentence_deltas") else run.response_sentence_deltas[sentence_idx]
    
    for result in run.ablation_results:
        if result["span_id"] == primary_span_id:
            delta_cos = result.get("delta_cos", 0.0)
            normalized_score = result.get("normalized_score", 0.0)
            break
    
    # Prepare the matrix
    matrix_snippet = ""
    full_matrix_json = ""
    
    # Build influence matrix if we have the data
    if hasattr(run, "ablation_results") and run.ablation_results:
        # Build matrix from ablation results
        num_spans = len(run.segments)
        num_sentences = len(sentences)
        matrix = [[0.0 for _ in range(num_sentences)] for _ in range(num_spans)]
        
        for result in run.ablation_results:
            sid = result["span_id"]
            deltas = result.get("sentence_deltas", [])
            for j, d in enumerate(deltas):
                if j < num_sentences and sid < num_spans:
                    matrix[sid][j] = d
        
        # Format matrix snippet
        matrix_snippet = format_matrix_snippet(matrix, primary_span_id, sentence_idx)
        
        # Format full matrix as JSON
        full_matrix_json = format_full_matrix_json(matrix, run.segments, sentences)
    
    # Prepare the request
    request = RewriteRequest(
        span_id=primary_span_id,
        sentence_idx=sentence_idx,
        span_text=span_text,
        sentence_text=sentence_text,
        full_prompt=run.prompt,
        full_response=run.completion,
        delta_cos=delta_cos,
        normalized_score=normalized_score,
        sentence_delta=sentence_delta,
        user_comment=user_comment,
        matrix_snippet=matrix_snippet,
        full_matrix_json=full_matrix_json
    )
    
    # Get secondary influences if available
    if hasattr(run, "ablation_results") and len(run.ablation_results) > 1:
        # Find other spans with some influence on this sentence
        secondary_influences = []
        for result in run.ablation_results:
            if result["span_id"] != primary_span_id and "sentence_deltas" in result:
                if sentence_idx < len(result["sentence_deltas"]):
                    secondary_influences.append({
                        "span_id": result["span_id"],
                        "delta_cos": result["delta_cos"],
                        "sentence_delta": result["sentence_deltas"][sentence_idx]
                    })
        
        # Sort by sentence-specific delta if available, otherwise by overall delta
        secondary_influences.sort(
            key=lambda x: x.get("sentence_delta", 0.0), 
            reverse=True
        )
        request.secondary_influences = secondary_influences[:3]  # Top 3
    
    # Generate rewrites
    from .settings import get_settings
    
    # Create LLM wrapper, optionally with custom model
    llm = LLMWrapper()
    if model:
        # If a custom model is specified, modify the LLM to use it
        # We need to access the model directly on the wrapper
        # This keeps the original global settings intact
        original_model = llm.completion_model
        llm.completion_model = model
        print(f"Using custom model for rewrite: {model} (default was {original_model})")
    
    suggestions = await generate_rewrite(request, llm)
    
    # Restore original model if we changed it
    if model:
        llm.completion_model = original_model
    
    # Return as dict mapping span_id to suggestions
    return {primary_span_id: suggestions} 