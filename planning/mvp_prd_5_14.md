# Prompt Attribution MVP – Product Requirements Document

**Version:** 0.1  
**Author:** Greyson Lampley (drafted with ChatGPT)  
**Date:** May 14 2025

## 1. Purpose

Debugging long, complex prompts is painful. Engineers need a fast way to see
*which* parts of a prompt drive unexpected model behavior.  
This MVP delivers an attribution heat‑map by perturbing the prompt (black‑box
ablation) and measuring the impact on the model's output.

## 2. Goals & Non‑Goals

### Goals
- Surface the top‑influential prompt segments for a single GPT‑4o call.  
- Provide a simple visual heat‑map inside a Jupyter notebook/VS Code panel.  
- Keep per‑run cost ≤ 5× a single GPT‑4o call (Target: ≤ $0.15).  
- Ship in one 7‑day sprint.

### Non‑Goals
- ✗ Multi‑turn chat attribution  
- ✗ Gradient‑based saliency on closed weights  
- ✗ Automatic prompt rewriting suggestions  
- ✗ Web production UI

## 3. Personas

- **Prompt Engineer** – iterates on prompts daily, wants quick feedback.  
- **LLM Researcher** – needs explainability data for papers/blogs.

## 4. Key Use Cases

**UC‑1** Engineer pastes a failing prompt into the notebook, clicks "Run Attribution",
and sees red‑hot segments highlighting contradictory instructions.

**UC‑2** Engineer edits a highlighted segment, re‑runs, and confirms the model
response now satisfies the spec.

## 5. Functional Requirements

### FR‑1 Segmenter  
- FR‑1.1 Autodetect "### Headings"; else fall back to 40‑token sliding windows.  

### FR‑2 Ablation Engine  
- FR‑2.1 Iteratively remove one segment, call GPT‑4o at T=0.  
- FR‑2.2 Async batch requests with retry/back‑off.  
- FR‑2.3 Cache identical calls (hash = SHA‑256(prompt)).

### FR‑3 Scorer  
- FR‑3.1 Embed baseline and ablated outputs with `embeddings‑3‑small`.  
- FR‑3.2 Compute cosine distance; higher = more impact.  
- FR‑3.3 Normalize scores 0‑1.

### FR‑4 Visualizer  
- FR‑4.1 Render inline HTML heat‑map; red = high impact, blue = low.  
- FR‑4.2 Hover tooltip shows numeric score and removed text.

### FR‑5 Cost Optimizer  
- FR‑5.1 Early‑stop ablation once cumulative impact ≥ 85%.  
- FR‑5.2 Hierarchical two‑pass windowing (20% coarse → hot chunk zoom).

## 6. Non‑Functional Requirements

- **Performance:** < 30s total latency for a 2,000‑token prompt on 150 Mbit/s.  
- **Reliability:** Retry 3× before surfacing error.  
- **Security:** Never log prompt contents outside local environment.  
- **Extensibility:** Clean class boundaries (Segmenter, AblationEngine, etc.).

## 7. Success Metrics

- **Hit Rate:** ≥ 60% of runs correctly flag the contradictory segment
  (measured on 10 internal "mystery" prompts).  
- **Cost per Run:** ≤ $0.15 (including embedding calls).  
- **Mean Time‑to‑Insight (MTTI):** User sees heat‑map in ≤ 30s.

## 8. Milestones (One‑Week Sprint)

- **Day 1:** Repo skeleton, baseline call persistence (`run.json`).  
- **Day 2:** Segmenter & basic scorer.  
- **Day 3:** Async ablation engine + caching.  
- **Day 4:** HTML heat‑map prototype.  
- **Day 5:** Cost optimizer (early‑stop, hierarchical).  
- **Day 6:** Polish UX & inline tooltips.  
- **Day 7:** Dog‑food, collect feedback, decide next steps.

## 9. Assumptions

- GPT‑4o temperature 0 is sufficiently deterministic.  
- Embedding distance is a proxy for semantic change.  
- Users are comfortable running a notebook/VS Code plug‑in.

## 10. Risks & Mitigations

- **R‑1:** API cost explosion → Early‑stop & hierarchical windows.  
- **R‑2:** Non‑determinism hides causal links → Fix seed, T=0.  
- **R‑3:** Large prompts exceed context window → Warn & truncate gracefully.

## 11. Open Questions

- ❓ What prompt length should trigger automatic hierarchical sampling?  
- ❓ Should we store user runs for future model fine‑tuning?  
- ❓ Which UI surface (VS Code panel vs. Jupyter) is first?

## 12. Appendix

- **Glossary:** ablation, heat‑map, cumulative impact.  
- **Cost table:** (tokens vs. $) to estimate budget impact for teams. 