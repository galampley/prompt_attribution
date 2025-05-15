#!/usr/bin/env python3
"""Prompt Attribution - Visualization Example

This script demonstrates how to visualize prompt attribution results
with heat maps to identify which segments have the most impact.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the core package to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

# Direct imports from specific modules
from prompt_attribution.segmenter.segmenter import Segmenter
from prompt_attribution.engine.llm_wrapper import LLMWrapper
from prompt_attribution.engine.run_manager import RunManager
from prompt_attribution.engine.ablation_engine import AblationEngine
from prompt_attribution.scorer.scorer import Scorer
from prompt_attribution.visualizer.heatmap import HeatmapVisualizer
from prompt_attribution.visualizer.file_util import open_in_browser, save_visualization


async def main():
    """Run the main visualization example."""
    print("Prompt Attribution - Visualization Example")
    print("==========================================")
    
    # Initialize components
    print("\n1. Initializing components...")
    segmenter = Segmenter()
    llm = LLMWrapper()
    run_manager = RunManager()
    ablation_engine = AblationEngine(llm, run_manager)
    
    # Define a test prompt
    prompt = """
### Role
You are a helpful content writer who specializes in blog posts.

### Task
Write an engaging introduction paragraph about artificial intelligence.

### Style Guidance
Use simple language accessible to non-technical readers.
Avoid jargon and complex terminology.
Include at least one rhetorical question.

### Constraints
Maximum length: 150 words.
Include the phrase "digital revolution" somewhere.
Do not mention specific AI products by name.
"""

    # Segment the prompt
    print("\n2. Segmenting prompt...")
    segments = segmenter.segment(prompt)
    
    for segment in segments:
        print(f"  - Segment {segment.id}: {segment.text[:50].replace(chr(10), ' ')}...")
    
    # Get baseline completion
    print("\n3. Getting baseline completion...")
    baseline_completion = await llm.get_completion(prompt)
    print(f"  Completion: {baseline_completion[:100]}...")
    
    # Create a run
    print("\n4. Creating attribution run...")
    run = run_manager.create_run(
        prompt=prompt,
        completion=baseline_completion,
        segments=segments
    )
    print(f"  Created run with ID: {run.id}")
    
    # Run ablation tests
    print("\n5. Running ablation tests...")
    print("  This may take a minute as we test each segment...")
    
    # Create a scorer for the ablation tests
    scorer = Scorer(baseline_completion, llm)
    
    # Run the ablation tests with the scorer
    ablated_run = await ablation_engine.run_ablation_tests(
        run,
        scorer=scorer
    )
    
    # Analyze results
    print("\n6. Analyzing results...")
    results = ablated_run.ablation_results
    sorted_results = sorted(results, key=lambda r: r["delta_cos"], reverse=True)
    
    print("  Segments by impact (highest first):")
    for result in sorted_results:
        span_id = result["span_id"]
        impact = result["delta_cos"]
        segment_text = segments[span_id].text[:50].replace(chr(10), " ")
        print(f"  - Segment {span_id}: {impact:.4f} - {segment_text}...")
    
    # Normalize scores
    normalized_results = scorer.get_normalized_scores(sorted_results)
    
    # Create and display visualization
    print("\n7. Creating heat map visualization...")
    visualizer = HeatmapVisualizer(color_scheme="gradient")
    html_content = visualizer.visualize_run(ablated_run)
    
    # Save HTML to file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"heatmap_{ablated_run.id}.html"
    visualizer.save_html(ablated_run, str(output_path))
    print(f"  Saved heat map to: {output_path}")
    
    # Open in browser
    print("\n8. Opening visualization in browser...")
    tmp_path = open_in_browser(html_content)
    print(f"  Opened visualization from: {tmp_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    try:
        # First try to use nest_asyncio if we're in a Jupyter environment
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except (ImportError, RuntimeError):
        # Fall back to regular asyncio if we're not in Jupyter
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main()) 