# Prompt Attribution

A tool for analyzing which parts of a prompt most influence an LLM's output.

## Overview

Prompt Attribution helps debug and optimize complex prompts by systematically measuring which segments drive a model's behavior. It works by:

1. Breaking the prompt into logical segments
2. Performing ablation tests (removing each segment and measuring the impact)
3. Scoring the impact using embedding-based similarity
4. Visualizing the results with detailed HTML heat maps and impact tables
5. Generating rewrite suggestions for problematic prompt segments

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/prompt-attribution.git
cd prompt-attribution

# Install dependencies
poetry install
```

## Getting Started

1. Set up your environment variables:

```bash
# Create a .env file with your OpenAI API key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

2. Run the example script:

```bash
# Run the full attribution and visualization pipeline
python examples/visualize_attribution.py

# Or try the optimized hierarchical approach 
python examples/hierarchical_attribution.py

# Test with prepared mystery prompts
python examples/mystery_prompts.py P-01 --mode standard
```

The script will:
- Break down the prompt into segments
- Get a baseline completion from the model
- Run ablation tests for each segment 
- Calculate impact scores
- Generate an HTML heat map visualization
- Open the visualization in your default browser

## Visualization Features

The HTML visualization includes multiple interconnected views:

- **Prompt Heat Map**: Shows the original prompt with color-coded segments based on their impact.
- **Model Response**: Displays the model's output with sentences color-coded based on which prompt segment influenced each sentence most.
- **Influence Matrix**: A grayscale grid showing which prompt segments influence which response sentences.
- **Response Sentence Table**: Detailed mapping of each response sentence to the prompt segment that influenced it most, with impact scores and rewrite suggestion buttons.
- **Segment Impact Table**: Sortable table of all prompt segments with raw and normalized impact scores.

All tables are interactive and sortable, allowing you to analyze results from different perspectives.

## Project Structure

- `core/` - Core package with modular components
  - `prompt_attribution/segmenter/` - Breaks prompts into analyzable chunks
  - `prompt_attribution/engine/` - LLM interaction and ablation engine
  - `prompt_attribution/scorer/` - Calculates segment impact scores
  - `prompt_attribution/visualizer/` - Heat map and interactive table visualizations
  - `prompt_attribution/rewrite.py` - Auto-rewrite suggestions for problematic segments
  - `prompt_attribution/cli.py` - Command-line interface for attribution and rewrites
- `examples/` - Example usage and demo notebooks
- `ui/` - VS Code extension (coming soon)

## Features

- **Smart segmentation**: Automatically detects markdown headings or falls back to sliding windows
- **Efficient ablation**: Async batch processing with cost guardrails
- **Embedding-based scoring**: Uses OpenAI's embedding models to measure semantic differences
- **Sentence-level attribution**: Maps each response sentence to its most influential prompt segment
- **Impact visualization**: Color-coded heat maps showing the importance of each segment
- **Interactive tables**: Sortable tables with detailed impact metrics
- **Early stopping**: Automatically stops testing once enough impact is accounted for
- **Result caching**: Caches API calls to reduce costs during development
- **Hierarchical optimization**: Two-pass approach that reduces API calls by 60-80% while maintaining accuracy
- **Auto-rewrite suggestions**: Generate AI-powered rewrites for problematic prompt segments
- **CLI tool**: Command-line interface for generating rewrite suggestions
- **Custom model selection**: Choose different LLM models for rewrite generation

## Auto-Rewrite Suggestions

The tool includes functionality to get AI-powered rewrite suggestions for problematic prompt segments:

1. **Interactive UI**: Click the 💡 button in the response sentence table to request rewrites for the prompt segment influencing that sentence.

2. **CLI tool**: Generate rewrites directly from the command line:
   ```bash
   python -m core.prompt_attribution.cli rewrite <run_id> <sentence_idx> --comment "What needs improvement" --model "gpt-4o"
   ```

3. **Influence-informed rewrites**: The system uses the full influence matrix data to help the AI understand exactly how each part of the prompt affects the model's output.

4. **JSON output**: Rewrite suggestions include both the rewritten text and explanations of why the changes help.

5. **Custom models**: Specify which LLM to use for generating rewrite suggestions with the `--model` flag.

## Configuration

Configure the tool through environment variables or `.env` file:

- `OPENAI_API_KEY` - Your OpenAI API key
- `COMPLETION_MODEL` - Model for completions (default: "gpt-4o")
- `EMBEDDING_MODEL` - Model for embeddings (default: "text-embedding-3-small")
- `MAX_COST_PER_RUN` - Maximum cost per attribution run (default: 0.15)
- `MAX_CONCURRENT_REQUESTS` - Maximum concurrent API requests (default: 20)
- `EARLY_STOP_THRESHOLD` - Cumulative impact threshold for early stopping (default: 0.85)

## Performance Optimizations

The toolkit includes several performance optimizations:

### Hierarchical Windowing

Instead of testing each individual segment (which can be expensive for long prompts), the hierarchical approach:

1. First tests coarse segments (~20% of the prompt)
2. Identifies the highest-impact region
3. Only tests fine-grained segments within that region

This dramatically reduces API calls while maintaining accuracy for the most important parts of the prompt.

### Memoized Segmentation

The segmenter caches results based on prompt hash, preventing repeated segmentation of identical prompts.

### Disk Cache for Embeddings

API responses, including embeddings, are automatically cached to disk to prevent duplicate API calls.

## Use Cases

- **Prompt Debugging**: Identify which parts of your prompt are causing unexpected behavior
- **Prompt Optimization**: Trim or refine low-impact sections to reduce token usage and costs
- **Instruction Validation**: Confirm which instructions the model is actually following
- **Bias Detection**: Identify which prompt segments disproportionately influence the model's output
- **Educational Tool**: Learn about prompt engineering through visual feedback
- **Continuous Improvement**: Use rewrite suggestions to iteratively refine your prompts

## License

MIT 