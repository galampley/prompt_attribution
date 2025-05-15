#!/usr/bin/env python3
"""Prompt Attribution - Hierarchical Optimization Example

This script demonstrates the optimized hierarchical ablation approach 
for more efficient prompt attribution with fewer API calls.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the core package to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

# Direct imports from specific modules
from prompt_attribution.segmenter.segmenter import Segmenter
from prompt_attribution.segmenter.hierarchical import HierarchicalSegmenter
from prompt_attribution.engine.llm_wrapper import LLMWrapper
from prompt_attribution.engine.run_manager import RunManager
from prompt_attribution.engine.hierarchical_engine import HierarchicalAblationEngine
from prompt_attribution.scorer.scorer import Scorer
from prompt_attribution.visualizer.heatmap import HeatmapVisualizer
from prompt_attribution.visualizer.file_util import open_in_browser


async def run_standard_attribution(prompt, llm, run_manager, segmenter):
    """Run standard attribution for comparison."""
    print("\n=== Standard Attribution ===")
    start_time = time.time()
    
    print("1. Segmenting prompt...")
    segments = segmenter.segment(prompt)
    print(f"   Found {len(segments)} segments")
    
    print("2. Getting baseline completion...")
    baseline_completion = await llm.get_completion(prompt)
    
    print("3. Creating attribution run...")
    run = run_manager.create_run(
        prompt=prompt,
        completion=baseline_completion,
        segments=segments
    )
    
    print("4. Running ablation tests (this may take a while)...")
    from prompt_attribution.engine.ablation_engine import AblationEngine
    ablation_engine = AblationEngine(llm, run_manager)
    
    # Create a scorer for the ablation tests
    scorer = Scorer(baseline_completion, llm)
    
    # Run the ablation tests with the scorer
    ablated_run = await ablation_engine.run_ablation_tests(
        run,
        scorer=scorer
    )
    
    elapsed = time.time() - start_time
    print(f"Standard attribution completed in {elapsed:.2f} seconds")
    print(f"API calls made: approximately {len(segments)}")
    
    return ablated_run


async def run_hierarchical_attribution(prompt, llm, run_manager, hierarchical_segmenter):
    """Run hierarchical attribution with optimizations."""
    print("\n=== Hierarchical Attribution ===")
    start_time = time.time()
    
    print("1. Getting baseline completion...")
    baseline_completion = await llm.get_completion(prompt)
    
    print("2. Creating attribution run...")
    run = run_manager.create_run(
        prompt=prompt,
        completion=baseline_completion,
        segments=[]  # We'll populate these during hierarchical ablation
    )
    
    print("3. Running hierarchical ablation...")
    hierarchical_engine = HierarchicalAblationEngine(
        llm, run_manager, hierarchical_segmenter
    )
    
    # Create a scorer for the ablation tests
    scorer = Scorer(baseline_completion, llm)
    
    # Run the hierarchical ablation
    ablated_run = await hierarchical_engine.run_hierarchical_ablation(
        run,
        scorer=scorer
    )
    
    elapsed = time.time() - start_time
    print(f"Hierarchical attribution completed in {elapsed:.2f} seconds")
    print(f"Total segments tested: {len(ablated_run.ablation_results)}")
    
    return ablated_run


async def compare_and_visualize(standard_run, hierarchical_run):
    """Compare the results and visualize them."""
    print("\n=== Comparison ===")
    
    # Create visualizations for both methods
    visualizer = HeatmapVisualizer(color_scheme="gradient")
    
    # Generate HTML for standard run
    standard_html = visualizer.visualize_run(standard_run)
    
    # Generate HTML for hierarchical run
    hierarchical_html = visualizer.visualize_run(hierarchical_run)
    
    # Create a combined view
    combined_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Attribution Comparison</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 0; padding: 20px; }}
        h1 {{ font-size: 1.5em; margin-bottom: 0.5em; }}
        h2 {{ font-size: 1.2em; margin-top: 1.5em; margin-bottom: 1em; }}
        h3 {{ font-size: 1.1em; margin-bottom: 0.5em; }}
        .comparison {{ display: flex; flex-direction: column; gap: 2em; }}
        @media (min-width: 1200px) {{ .comparison {{ flex-direction: row; }} }}
        .method {{ flex: 1; }}
        .stats {{ margin-top: 1em; margin-bottom: 1em; background: #f5f5f5; padding: 1em; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Prompt Attribution Comparison</h1>
    <p>This comparison shows the difference between standard and hierarchical attribution approaches.</p>
    
    <div class="comparison">
        <div class="method">
            <h2>Standard Attribution</h2>
            <div class="stats">
                <p><strong>Segments:</strong> {len(standard_run.segments)}</p>
                <p><strong>API Calls:</strong> ~{len(standard_run.segments)}</p>
            </div>
            <div class="visualization">
                {standard_html}
            </div>
        </div>
        
        <div class="method">
            <h2>Hierarchical Attribution</h2>
            <div class="stats">
                <p><strong>Segments:</strong> {len(hierarchical_run.segments)}</p>
                <p><strong>API Calls:</strong> {len(hierarchical_run.ablation_results)}</p>
                <p><strong>API Call Reduction:</strong> {(1 - len(hierarchical_run.ablation_results) / len(standard_run.segments)) * 100:.1f}%</p>
            </div>
            <div class="visualization">
                {hierarchical_html}
            </div>
        </div>
    </div>
    
    <div style="margin-top: 2em; font-size: 0.8em; color: #666;">
        <p>Color intensity indicates segment impact on model output.</p>
    </div>
</body>
</html>"""
    
    # Save the comparison to a file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"comparison_{standard_run.id}_{hierarchical_run.id}.html"
    
    with open(output_path, "w") as f:
        f.write(combined_html)
    
    print(f"Saved comparison to: {output_path}")
    
    # Open in browser
    print("Opening comparison in browser...")
    open_in_browser(combined_html)


async def main():
    """Run the main optimization comparison example."""
    print("Prompt Attribution - Hierarchical Optimization Example")
    print("===================================================")
    
    # Initialize components
    print("\nInitializing components...")
    segmenter = Segmenter()
    hierarchical_segmenter = HierarchicalSegmenter()
    llm = LLMWrapper()
    run_manager = RunManager()
    
    # Define a test prompt (longer than usual to show the benefits of hierarchy)
    prompt = """
### System Context
You are an AI assistant tasked with providing helpful, accurate, and ethical responses to user inquiries. You should always prioritize user safety and well-being in your answers.

### User Context
The user is a software developer working on a machine learning project who needs help understanding the architecture of transformer models.

### Response Format Requirements
- Start with a brief summary (2-3 sentences)
- Use markdown formatting for headings and code blocks
- Include at least one code example in Python
- Keep the total response under 300 words
- End with a thought-provoking question

### Content Guidelines
- Use technical terms but briefly explain specialized concepts
- Focus on practical explanations rather than theoretical details
- Mention at least 2 specific applications of transformer models
- Avoid discussing specific commercial products or services
- Do not reference your own capabilities or limitations

### Tone and Style
- Professional but conversational
- Enthusiastic about the subject matter
- Encouraging of the user's learning journey
- Clear and concise explanations
- Use analogies where helpful for complex concepts

### Special Instructions
- If you are unsure of any technical details, indicate that clearly
- Do not make up information or provide potentially harmful advice
- Encourage best practices for model training and deployment
"""

    # Run both attribution methods
    standard_run = await run_standard_attribution(prompt, llm, run_manager, segmenter)
    hierarchical_run = await run_hierarchical_attribution(prompt, llm, run_manager, hierarchical_segmenter)
    
    # Compare and visualize results
    await compare_and_visualize(standard_run, hierarchical_run)
    
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