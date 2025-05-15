"""Prompt Attribution - Basic Example

This script demonstrates the core functionality of the Prompt Attribution toolkit 
to identify which parts of a prompt contribute most to the model's output.
"""

# %% [markdown]
# # Prompt Attribution - Basic Example
#
# This example demonstrates the core functionality of the Prompt Attribution toolkit
# to identify which parts of a prompt contribute most to the model's output.

# %% Setup
import asyncio
import os
import sys
from pathlib import Path

# Add the core package to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

# Direct imports from specific modules instead of package imports
from prompt_attribution.segmenter.segmenter import Segmenter
from prompt_attribution.engine.llm_wrapper import LLMWrapper
from prompt_attribution.engine.run_manager import RunManager, AblationResult
from prompt_attribution.engine.ablation_engine import AblationEngine
from prompt_attribution.scorer.scorer import Scorer

# %% [markdown]
# ## 1. Setup
#
# First, let's initialize our components:

# %% Initialize components
segmenter = Segmenter()
llm = LLMWrapper()
run_manager = RunManager()
ablation_engine = AblationEngine(llm, run_manager)

# %% [markdown]
# ## 2. Define a prompt
#
# Let's use a prompt with clearly defined sections to test the attribution:

# %% Define a test prompt
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

# %% [markdown]
# ## 3. Segment the prompt
#
# Now we'll break the prompt into segments for analysis:

# %% Segment the prompt
segments = segmenter.segment(prompt)

# Display segments
for segment in segments:
    print(f"Segment {segment.id}: {segment.text[:50]}...")

# %% [markdown]
# ## 4. Get baseline completion
#
# First, get the baseline output of the model for the full prompt:

# %% Get baseline completion
async def get_baseline():
    completion = await llm.get_completion(prompt)
    return completion

# Execute async function - avoid asyncio.run() in Jupyter environments
try:
    # Check if we're in a Jupyter/IPython environment with an existing event loop
    import nest_asyncio
    nest_asyncio.apply()  # Patch asyncio to allow nested event loops
    baseline_completion = asyncio.run(get_baseline())
except (ImportError, RuntimeError):
    # Alternative approach when nest_asyncio is unavailable or for other issues
    loop = asyncio.get_event_loop()
    baseline_completion = loop.run_until_complete(get_baseline())
    
print(baseline_completion)

# %% [markdown]
# ## 5. Create a run
#
# Now we'll create and persist a run with the baseline data:

# %% Create and persist a run
run = run_manager.create_run(
    prompt=prompt,
    completion=baseline_completion,
    segments=segments
)

print(f"Created run with ID: {run.id}")

# %% [markdown]
# ## 6. Run Ablation Tests
#
# Next, we'll run ablation tests by removing each segment and measuring the impact:

# %% Run ablation tests
async def run_ablation():
    # Create a scorer first
    scorer = Scorer(baseline_completion, llm)
    
    # Run ablation tests with the scorer
    updated_run = await ablation_engine.run_ablation_tests(
        run, 
        scorer=scorer
    )
    return updated_run

# Execute async function
try:
    import nest_asyncio
    nest_asyncio.apply()
    ablated_run = asyncio.run(run_ablation())
except (ImportError, RuntimeError):
    loop = asyncio.get_event_loop()
    ablated_run = loop.run_until_complete(run_ablation())

# %% [markdown]
# ## 7. Analyze Results
#
# Now let's look at which segments had the most impact:

# %% Analyze results
# Get results and sort by impact
results = ablated_run.ablation_results
sorted_results = sorted(results, key=lambda r: r["delta_cos"], reverse=True)

print("Segments by impact (highest first):")
for result in sorted_results:
    span_id = result["span_id"]
    impact = result["delta_cos"]
    segment_text = segments[span_id].text[:50].replace("\n", " ")
    print(f"Segment {span_id}: {impact:.4f} - {segment_text}...")

# %% [markdown]
# ## 8. Normalize Scores
#
# Finally, let's normalize the scores for visualization:

# %% Normalize scores
scorer = Scorer(baseline_completion, llm)
normalized_results = scorer.get_normalized_scores(sorted_results)

print("\nNormalized scores:")
for result in normalized_results:
    span_id = result["span_id"]
    norm_score = result["normalized_score"]
    segment_text = segments[span_id].text[:50].replace("\n", " ")
    print(f"Segment {span_id}: {norm_score:.4f} - {segment_text}...")

# %% [markdown]
# ## Next Steps
#
# In the next phase, we'll implement a heat map visualization of these results.
# %%
