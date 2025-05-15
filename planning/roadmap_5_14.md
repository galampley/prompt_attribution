# Technical Roadmap – Prompt Attribution MVP

*(Scope: one‑week sprint, single‑turn attribution, notebook/VS Code panel)*  

---

## 0. Foundations (Day 0 – ½)

1. **Repo bootstrap**  
   - Create monorepo with packages: `core/`, `ui/`, `examples/`.  
   - Set up Poetry (or `pipenv`) and pre‑commit hooks (`black`, `ruff`).  
2. **Secrets & configs**  
   - `.env` template for OpenAI keys, model names, cost caps.  
   - Central `settings.py` with Pydantic validation.  
3. **CI**  
   - GitHub Actions: lint, unit tests, basic smoke notebook run with dummy model.

---

## 1. Core Engine (Day 1)

### 1.1 Segmenter
* Heuristic: split on `^###` headings; fall back to 40‑token sliding window.  
* Output: `List[Span(start, end, text)]`.

### 1.2 LLM wrapper
* Async client around `openai.ChatCompletion.acreate()` with:  
  - `temperature = 0`, seed pinning (`"seed": 42`).  
  - Idempotent caching (`hash(prompt)`).  
  - Automatic back‑off / retry (429, 5xx).

### 1.3 Baseline run
* Persist JSON (`runs/{run_id}/run.json`) — ground truth for scoring.

---

## 2. Ablation Loop (Day 2)

1. **Perturbation strategy** – single‑segment deletion.  
2. **Async batch dispatcher** – `asyncio.gather` ≤ 20 concurrent calls.  
3. **Cost guardrail** – abort if projected spend > configured limit.

**Output schema**

```json
{
  "span_id": 3,
  "delta_cos": 0.37,
  "elapsed_ms": 612
}
```

---

## 3. Scorer (Day 3)

* Embed outputs with `openai.embeddings‑3‑small`.  
* Compute cosine distance, normalize to 0‑1.  
* Track cumulative impact; **early‑stop** when ≥ 85 %.

---

## 4. Visualizer (Day 4)

1. **Notebook HTML heat‑map**  
   - Inline CSS spans; gradient `rgba(255,0,0,α)` → transparent.  
2. **VS Code webview panel** (stretch)  
   - Use `vscode.ExtensionContext` + `postMessage` to refresh heat‑map live.

---

## 5. Optimizations (Day 5)

* **Hierarchical windowing**  
  - Pass 1: 20 % coarse chunks.  
  - Pass 2: re‑segment hottest chunk into 10 % sub‑windows.  
* **Disk cache** for embeddings.  
* **Memoized segmenter** keyed by prompt hash.

---

## 6. QA & Dog‑food (Day 6)

* Assemble 10 “mystery” prompts; record hit‑rate & latency.  
* Unit tests  
  - Deterministic output (seed).  
  - Cost ceiling respected.  
  - Heat‑map HTML well‑formed.

---

## 7. Polish + Release (Day 7)

* README quick‑start, GIF demo.  
* Publish to PyPI (`prompt‑attributor‑mvp`) and VS Code marketplace (preview).  
* Tag `v0.1.0` and finalize release notes.

---

## Milestone Checklist

| Milestone            | Owner   | Due | Status |
|----------------------|---------|-----|--------|
| Repo + CI            | Greyson | D0  | ☐ |
| Segmenter            | Greyson | D1  | ☐ |
| Async Engine         | Greyson | D2  | ☐ |
| Scorer & Early‑stop  | Greyson | D3  | ☐ |
| Notebook Heat‑map    | Greyson | D4  | ☐ |
| Hierarchical pass    | Greyson | D5  | ☐ |
| QA metrics           | Greyson | D6  | ☐ |
| Release v0.1         | Greyson | D7  | ☐ |

*(Update owners if team expands.)*

---

## Post‑MVP Expansion Backlog

1. **Open‑weights “shadow model”** — fine‑tune Llama‑3‑8B on `<prompt, GPT‑4o‑output>` pairs for gradient saliency.  
2. **Composite attribution** — fuse ablation scores with integrated‑gradients for token‑level heat‑maps.  
3. **Multi‑turn chat support** — tree‑based ablation to trace which *turn* injects the bug.  
4. **Auto‑rewrite suggestions** — meta‑LLM proposes clarifying edits to high‑impact spans.  
5. **Rich web UI** — diff view, collapsible sections, cost estimator.  
6. **Version‑control hooks** — Git pre‑commit blocking low‑confidence prompts.  
7. **Telemetry‑driven heuristics** — train predictor to skip low‑impact segments, cutting cost.  
8. **IDE integrations** — JetBrains / Emacs plug‑ins; live heat‑map as you type.  
9. **Policy compliance scans** — flag segments triggering safety filters or style violations.  
10. **Team analytics** — dashboard of common contradictions, average fix time, cost savings.

---

**Next Action:** Confirm milestone dates, spin up the repo, and tackle Foundations (0) today.
