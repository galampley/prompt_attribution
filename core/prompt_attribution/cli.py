"""Command-line interface for prompt attribution."""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from .engine import RunManager
from .engine.llm_wrapper import LLMWrapper
from . import rewrite
from .settings import get_settings


async def generate_rewrite_cmd(run_id: str, sentence_idx: int, user_comment: str = "", output_path: Optional[str] = None, model: Optional[str] = None):
    """Generate rewrite suggestions for a specific run and sentence.
    
    Args:
        run_id: The run ID to use
        sentence_idx: The sentence index to generate rewrites for
        user_comment: Optional comment about what's wrong with the sentence
        output_path: Optional path to save the suggestions to
        model: Optional model to use for rewrite generation
    """
    # Load the run
    run_manager = RunManager()
    run = run_manager.get_run(run_id)
    if not run:
        print(f"Run {run_id} not found")
        return
    
    # Generate rewrites
    suggestions_by_span = await rewrite.generate_rewrites_for_run(
        run=run,
        sentence_idx=sentence_idx,
        user_comment=user_comment,
        top_k_segments=1,
        model=model
    )
    
    if not suggestions_by_span:
        print(f"No rewrite suggestions generated for sentence {sentence_idx}")
        return
    
    # Store suggestions in the run
    for span_id, suggestions in suggestions_by_span.items():
        run.add_rewrite_suggestions(span_id, sentence_idx, suggestions)
    
    # Save the updated run
    run_manager._save_run(run)
    
    # Print the suggestions
    print("\nRewrite suggestions:")
    print("-------------------")
    for span_id, suggestions in suggestions_by_span.items():
        span_text = None
        for segment in run.segments:
            if segment["id"] == span_id:
                span_text = segment["text"]
                break
        
        print(f"\nFor span {span_id}:")
        print(f"Original: {span_text[:200]}{'...' if len(span_text) > 200 else ''}")
        # print("\nSuggestions:")
        # for i, suggestion in enumerate(suggestions, 1):
        #     print(f"{i}. {suggestion}")
        print("Suggestions\n", suggestions)
    
    # Optionally save to a file
    if output_path:
        with open(output_path, "w") as f:
            json.dump({
                "run_id": run_id,
                "sentence_idx": sentence_idx,
                "suggestions": suggestions_by_span
            }, f, indent=2)
        print(f"\nSaved suggestions to {output_path}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Prompt Attribution CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Rewrite command
    rewrite_parser = subparsers.add_parser("rewrite", help="Generate rewrite suggestions")
    rewrite_parser.add_argument("run_id", help="Run ID to generate rewrites for")
    rewrite_parser.add_argument("sentence_idx", type=int, help="Index of the sentence to rewrite")
    rewrite_parser.add_argument("--comment", "-c", default="", help="Comment about what's wrong with the sentence")
    rewrite_parser.add_argument("--output", "-o", help="Path to save suggestions to")
    rewrite_parser.add_argument("--model", "-m", help="Model to use for rewrite generation (overrides default)")
    
    # Parse args
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "rewrite":
        asyncio.run(generate_rewrite_cmd(
            run_id=args.run_id,
            sentence_idx=args.sentence_idx,
            user_comment=args.comment,
            output_path=args.output,
            model=args.model
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 