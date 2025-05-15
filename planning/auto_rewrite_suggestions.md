# Auto-Rewrite Suggestions Feature

*Last updated: {{DATE}}*

---

## Goal
Automatically propose concise, clarity-oriented rewrites for the **highest-impact prompt segments** returned by the attribution engine. A secondary "meta-LLM" generates these suggestions, helping users tighten wording, remove ambiguity, and optimize token usage while preserving the original intent.

---

## 1  Triggering the Rewrite Workflow
1. Run attribution as usual and sort segments by *normalized impact score*.
2. Select either:
   - Top-k segments *(configurable, e.g. `k=5`)*, **or**
   - All segments whose normalized score ≥ `impact_threshold` *(default 0.25).*  
   This keeps latency & cost bounded.

---

## 2  Meta-LLM Prompt Design
```text
System:  Your job is to rewrite user-provided prompt snippets to improve clarity. Preserve meaning and tone.

User:
ORIGINAL_PROMPT_SNIPPET:
"""
{segment_text}
"""

CONSTRAINTS:
• ≤ {max_tokens} tokens
• Semantically equivalent
• Keep placeholders / keywords unchanged
• Reply with **rewrite only** – no commentary
```
Options:
- Request *N* alternatives: "Provide 3 variants ranked by clarity."
- Optionally include the baseline model completion for added context.

---

## 3  Data Flow / Code Changes
| Area | Update |
|------|--------|
| **`core/prompt_attribution/rewrite.py`** | New helper with `generate_rewrites(segments, model, n)` returning `{span_id: [str]}`. |
| **`engine.Run`** | Add `rewrite_suggestions: dict[int, list[str]]`. |
| **Visualizer** | Extend Segment Impact Table with a "Suggested Rewrite" column (tooltip or inline). |
| **Examples** | `--include-rewrites` flag to enable the extra step. |

---

## 4  UX Considerations
- Tooltip vs. inline column: tooltips keep the table compact; inline is copy-paste friendly.
- **Diff view**: highlight removed tokens (red) vs. additions (green).
- **"Apply & Re-score"** button: accept a rewrite, regenerate completion, and instantly show new impact.

---

## 5  Cost / Performance Safeguards
- Global `max_meta_cost_per_run` (USD or tokens).
- Cache suggestions by *prompt-hash + segment-hash*.
- Batch multiple segments per chat request when feasible.

---

## 6  Evaluation Loop (Optional)
1. After user applies a rewrite:
   1. Insert new text into prompt.
   2. Rerun a lightweight attribution or at least a fresh completion.
   3. Surface **before vs. after** diffs and updated impact scores.

---

## 7  Edge-Cases & Pitfalls
- **Long segments** → truncate or ask for summary-style rewrites.
- **Code/JSON blocks** → tag content-type so meta-LLM preserves formatting.
- **Conflicting rewrites** if multiple segments interact → consider staged application.

---

## 8  README Blurb
> **Auto-rewrite suggestions** — For each high-impact segment, a secondary LLM proposes concise, clarity-focused rewrites. Users can preview, accept, and instantly see how the change affects model behavior. 