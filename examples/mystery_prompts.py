import asyncio
import time
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "core"))

# Core helpers
from prompt_attribution.datasets import load_mystery_prompts
from prompt_attribution.segmenter.segmenter import Segmenter
from prompt_attribution.engine.llm_wrapper import LLMWrapper
from prompt_attribution.engine.run_manager import RunManager
from prompt_attribution.engine.ablation_engine import AblationEngine
from prompt_attribution.engine.hierarchical_engine import HierarchicalAblationEngine
from prompt_attribution.scorer.scorer import Scorer
from prompt_attribution.visualizer.heatmap import HeatmapVisualizer
from prompt_attribution.visualizer.file_util import open_in_browser

# Simple wrapper to count API calls
class CountingLLMWrapper(LLMWrapper):
    def __init__(self):
        super().__init__()
        self.completion_count = 0
        self.embedding_count = 0
    
    async def get_completion(self, *args, **kwargs):
        self.completion_count += 1
        return await super().get_completion(*args, **kwargs)
    
    async def get_embedding(self, *args, **kwargs):
        self.embedding_count += 1
        return await super().get_embedding(*args, **kwargs)

async def run_comparison(case_id: str = "P-01", mode: str = "hierarchical"):
    """Run attribution with selected engine mode(s).
    
    Args:
        case_id: Which mystery prompt to use (P-01 to P-10)
        mode: Engine mode to use ("standard", "hierarchical", or "both")
    """
    # Common setup
    case = next(c for c in load_mystery_prompts() if c.id == case_id)
    prompt_text = case.text
    segmenter = Segmenter()
    run_mgr = RunManager()
    segments = segmenter.segment(prompt_text)
    
    results = {}
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    viz = HeatmapVisualizer(color_scheme="gradient")
    
    # Run standard ablation if requested
    if mode in ["standard", "both"]:
        print(f"\n===== Running STANDARD ablation on {case_id} =====")
        std_llm = CountingLLMWrapper()
        std_engine = AblationEngine(std_llm, run_mgr)
        
        std_start = time.time()
        baseline = await std_llm.get_completion(prompt_text)
        std_run = run_mgr.create_run(prompt_text, baseline, segments)
        scorer = Scorer(baseline, std_llm)
        std_run = await std_engine.run_ablation_tests(std_run, scorer=scorer)
        std_time = time.time() - std_start
        
        # Save standard results
        results["standard"] = {
            "time_seconds": std_time,
            "completion_calls": std_llm.completion_count,
            "embedding_calls": std_llm.embedding_count,
            "total_calls": std_llm.completion_count + std_llm.embedding_count,
            "run": std_run
        }
        
        # Save standard visualization
        std_path = output_dir / f"heatmap_{case_id}_standard.html"
        viz.save_html(std_run, str(std_path))
        print(f"Standard heatmap saved to: {std_path}")
        
        # Open in browser if only running standard
        if mode == "standard":
            try:
                open_in_browser(std_path.read_text())
            except Exception as e:
                print("Could not open browser:", e)
    
    # Run hierarchical ablation if requested
    if mode in ["hierarchical", "both"]:
        print(f"\n===== Running HIERARCHICAL ablation on {case_id} =====")
        hier_llm = CountingLLMWrapper()
        hier_engine = HierarchicalAblationEngine(hier_llm, run_mgr)
        
        hier_start = time.time()
        baseline = await hier_llm.get_completion(prompt_text)
        hier_run = run_mgr.create_run(prompt_text, baseline, segments)
        scorer = Scorer(baseline, hier_llm)
        hier_run = await hier_engine.run_hierarchical_ablation(hier_run, scorer=scorer)
        hier_time = time.time() - hier_start
        
        # Save hierarchical results
        results["hierarchical"] = {
            "time_seconds": hier_time,
            "completion_calls": hier_llm.completion_count,
            "embedding_calls": hier_llm.embedding_count,
            "total_calls": hier_llm.completion_count + hier_llm.embedding_count,
            "run": hier_run
        }
        
        # Save hierarchical visualization
        hier_path = output_dir / f"heatmap_{case_id}_hierarchical.html"
        viz.save_html(hier_run, str(hier_path))
        print(f"Hierarchical heatmap saved to: {hier_path}")
        
        # Open in browser if only running hierarchical
        if mode == "hierarchical":
            try:
                open_in_browser(hier_path.read_text())
            except Exception as e:
                print("Could not open browser:", e)
    
    # Print comparison if running both
    if mode == "both" and "standard" in results and "hierarchical" in results:
        print("\n===== RESULTS COMPARISON =====")
        print(f"Prompt: {case_id} ({len(prompt_text)} chars, {len(segments)} segments)")
        print(f"Standard:     {results['standard']['time_seconds']:.2f}s, {results['standard']['total_calls']} API calls")
        print(f"Hierarchical: {results['hierarchical']['time_seconds']:.2f}s, {results['hierarchical']['total_calls']} API calls")
        print(f"Speedup: {results['standard']['time_seconds'] / results['hierarchical']['time_seconds']:.2f}x")
        print(f"API calls saved: {results['standard']['total_calls'] - results['hierarchical']['total_calls']} ({100 * (1 - results['hierarchical']['total_calls'] / results['standard']['total_calls']):.1f}%)")
        
        # Open hierarchical in browser for "both" mode since it's typically cleaner
        try:
            hier_path = output_dir / f"heatmap_{case_id}_hierarchical.html"
            open_in_browser(hier_path.read_text())
        except Exception as e:
            print("Could not open browser:", e)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run prompt attribution on mystery prompts")
    parser.add_argument("prompt_id", nargs="?", default="P-01", help="Mystery prompt ID (P-01 to P-10)")
    parser.add_argument("--mode", "-m", choices=["standard", "hierarchical", "both"], 
                        default="hierarchical", help="Ablation mode to run")
    args = parser.parse_args()
    
    print(f"Running attribution for prompt {args.prompt_id} in {args.mode} mode")
    asyncio.run(run_comparison(args.prompt_id, args.mode))

if __name__ == "__main__":
    main()