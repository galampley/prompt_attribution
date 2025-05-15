# Auto-Rewrite Suggestions – v2 (Sentence-Driven Flow)

*Last updated: {{DATE}}*

---

## 0  Purpose
When a user spots an incorrect or low-quality **response sentence** *Sᵢ*, the visualizer already shows which **prompt span** *Pⱼ* most influenced it. We want a one-click workflow that asks a meta-LLM how to rewrite *Pⱼ* so that the next model run fixes *Sᵢ*, while preserving overall intent.

---

## 1  Minimal Context Package for Meta-LLM
1. **Original prompt span Pⱼ** (text to be rewritten)
2. **Problematic response sentence Sᵢ**
3. (Optional) **User note** explaining what's wrong
4. **Full prompt** (read-only)
5. **Full response** (read-only)
6. (Optional) Neighboring high-influence spans (top-k)
7. **Influence matrices**:
   - Segment → sentence influence matrix (heatmap data)
   - Sentence-level deltas showing how much each sentence changes when segments are removed
   - Raw delta cosine values showing magnitude of impact

Including quantitative influence measures provides critical context about which prompt components most strongly affect each response component, enabling more targeted rewrites.

---

## 2  Prompt Template
```text
SYSTEM
You are a prompt-engineering assistant. Rewrite a given prompt snippet so a target LLM produces a better answer.

USER
ORIGINAL PROMPT SNIPPET (to rewrite)
"""
{P_j}
"""

MODEL SENTENCE (problematic)
"""
{S_i}
"""

FULL PROMPT CONTEXT (read-only)
<<<
{full_prompt}
>>>

FULL MODEL RESPONSE (read-only)
<<<
{full_response}
>>>

INFLUENCE DATA (quantitative)
- Primary control: Span {P_j} has {delta_cos:.4f} influence on this sentence (normalized: {norm_score:.4f})
- Other influencing spans: {secondary_span_info}
- Sentence delta: {sentence_delta:.4f} (relative to other sentences in response)
- Full matrix excerpt:
{matrix_snippet}

WHAT'S WRONG (optional, provided by user)
"""
{user_comment}
"""

Rewrite the ORIGINAL PROMPT SNIPPET so that the next model run will:
• Correct or improve the given MODEL SENTENCE.
• Preserve the overall intent and style of the original prompt.
• Focus especially on aspects that had high influence scores.
Return 1-2 candidate rewrites. Do NOT change anything outside that span. Reply with the rewrites only, each on its own line.
```

---

## 3  Data Flow / Code Changes
| Area | Update |
|------|--------|
| **`core/prompt_attribution/rewrite.py`** | `generate_rewrite(case)` util taking span, sentence, prompt, response, note. |
| **`engine.Run`** | Add `rewrite_suggestions: Dict[str, List[str]]`, keyed `"seg{span}_sent{idx}"`. |
| **Visualizer** | In Response-Sentence table, add ⚙️ button → JS modal → call `/rewrite` endpoint → display suggestions. |
| **API/CLI** | FastAPI `/rewrite` (POST) or `pattribute rewrite --run RUN_ID --sent 3`. |

---

## 4  UX Flow
1. User clicks ⚙️ next to sentence *Sᵢ*.
2. Modal shows *Pⱼ* and *Sᵢ* side by side + optional textbox "What's wrong?"
3. "Ask for rewrite" → backend calls meta-LLM.
4. Suggestions appear instantly in modal.
5. (Stretch) "Apply & Re-score" button inserts chosen rewrite, reruns attribution.

---

## 5  Cost & Safety
* Only on-demand calls (no automatic batch).
* Trim full prompt/response if > 2 000 tokens.
* `MAX_META_COST_PER_RUN` env var guard.
* Cache suggestions by `(prompt_hash, span_hash, sentence_idx, user_note_hash)`.

---

## 6  Relationship to v1 Auto-Rewrite
v1 auto-rewrite blindly suggested edits for **top-impact spans**. v2 is **sentence-triggered**, providing richer context (why it's wrong) and focused on a specific error.

---

## 7  MVP Checklist
- [ ] Extend `Run` dataclass
- [ ] Implement `rewrite.py` helper (async OpenAI call)
- [ ] Add FastAPI endpoint
- [ ] JS modal + button in heat-map HTML
- [ ] Manual test on mystery prompts 