# Magic the Gathering Judge AI

## What does the app do?
An AI-powered MTG Judge for casual players. It takes a natural language board state or rules question, retrieves the exact Comprehensive Rules that apply, and outputs a cited ruling.

---

## Part 1: What's Hard About This AI Behavior

The goal sounds simple: answer a rules question and cite the right rule. The hard part is that three steps have to work correctly in sequence, and each one can fail silently, no crash, just a wrong answer that looks fine.

**The retrieval bottleneck.**
The system can only cite rules it actually retrieved. If step 2 returns the wrong chunks, step 3 either falls back on training data or admits it can't answer. The dangerous case is the first one, where a confidently wrong answer looks identical to a correct one unless you verify the citation yourself.

**Keyword extraction is difficult.**
Retrieval quality depends entirely on what step 1 extracts. MTG questions pack card types, game actions, and rule concepts into the same sentence. Pull the wrong term and you get wrong chunks with no error signal, the pipeline just keeps going.

**Cited rule ≠ correct ruling.**
A model can cite a real rule number and still misapply it. The eval only checks whether the citation string appears in the answer. An 80% score means 8/10 answers contained the expected rule number, it says nothing about whether the rulings were legally correct.

---

## Part 2: Iterations

### Version 1 — Flat keyword extraction prompt

**Change:** The `extract_keywords` prompt asked GPT to extract "2-3 important MTG keywords, mechanic names, or rule concepts" with examples like `'Deathtouch', 'Layer 7', 'Priority', 'Trample'`.

**Example:** Test case 10 — *"After a sorcery resolves, which player receives priority first?"* — consistently failed. GPT extracted `["Priority", "Sorcery"]`, which as an embedding landed closer to Section 307 (Sorceries) than Section 117 (Timing and Priority). All five retrieved chunks were about sorcery casting conditions. GPT responded: *"The provided rules do not specify which player receives priority first."*

**Delta:** 8/10 passing (80%). Test 10 failed every run.

**Conclusion:** Allowing card-type words like "Sorcery" as valid keywords pollutes the query and pulls the wrong rule section.

---

### Version 2 — Prompt focusing on rule section names

**Change:** Rewrote the prompt to prefer section headings over card types: *"prefer 'Timing and Priority' not 'Sorcery'."*

**Example:** Test 10 immediately passed — GPT extracted `["Timing and Priority", "Resolving Spells and Abilities"]`, retrieved rule 117.3b, and cited it verbatim.

**Delta:** Test 10 passed but overall accuracy dropped below the 80% baseline. For Deathtouch questions, the abstract phrase "Continuous Effects" retrieved layer-system rules instead of the tighter 702.2 sub-rules that the keyword "Deathtouch" found directly.

**Conclusion:** Fixing one broad question broke several specific ones. Version 1 was reverted and test 10 is documented as a known limitation.

---

### Version 3 — Reducing TOP_K_RESULTS from 5 to 3

**Change:** `TOP_K_RESULTS` in `main.py` line 38 reduced from 5 to 3. Hypothesis: the 4th and 5th chunks might be introducing noise, causing GPT to pick the wrong rule.

**Example:** Test 3 — *"A player is at 0 life but the turn has not ended. Have they lost the game yet?"* — was failing because GPT cited rule `104.3b` instead of `704.5a`. Both were in the retrieved context and GPT picked the wrong one. Fewer chunks might force `704.5a` to dominate.

**Delta:** Accuracy stayed at 80% (8/10). The same two tests failed. Chunk count didn't affect which rules ranked highest.

**Conclusion:** The test 3 failure is a ranking problem, not a noise problem. `104.3b` scores higher cosine similarity than `704.5a` because "losing the game" maps directly to section 104, not section 704. The answer GPT gives is factually correct — the expected citation in the test may need updating. TOP_K_RESULTS reverted to 5.

---


When a user writes the prompt, the request hits main.py and enters the ask_judge() function (Line 253). Success oif this trace depends on the offline datra prep handled by ingest.py (line 271).

## Part 3: Code Walkthrough

### `ingest.py` — `chunk_rules()` — lines 71–131

**What it does:** Reads the rules text line by line and starts a new chunk whenever a line matches `^\d+\.[\da-z]*\.?` — the pattern for MTG rule numbers like `702.2b.` or `100.`. One chunk per rule.

**Why:** Each rule is a self-contained statement. Splitting at rule boundaries means each vector in ChromaDB represents exactly one rule, so a top-5 retrieval gives five distinct relevant rules rather than five overlapping excerpts from the same paragraph.

**Alternative rejected:** Fixed-size character chunking (500 chars, 50-char overlap) is the standard RAG approach. Rejected because it ignores rule boundaries — a 500-character split would routinely embed the first sentence of 702.2b with the last sentence of 702.2a, producing a vector that doesn't cleanly represent either rule.

---

### `main.py` — `retrieve_rules()` — lines 148–195

**What it does:** Joins the extracted keywords into a single string, embeds it with `text-embedding-3-small`, and queries ChromaDB via `query_embeddings` for the 5 closest chunks by cosine similarity.

**Why the model must match:** ChromaDB stores raw float vectors with no model tag. If `ingest.py` used `text-embedding-3-small` and `main.py` used a different model, the vectors would be in different spaces and similarity scores would be meaningless. The `EMBEDDING_MODEL` constant at the top of both files is the only thing enforcing this — change one without the other and retrieval silently breaks.

---

### `main.py` — `ask_judge()` — lines 253–308

**What it does:** Calls `extract_keywords()`, `retrieve_rules()`, and `draft_answer()` in sequence. Each step has its own `try/except` returning an HTTP 500 with a step-specific message.

**Why separate blocks:** One outer catch returns `"something failed"` with no context. Separate blocks surface `"Step 1 failed — OpenAI error: 429"` vs `"Step 2 failed — ChromaDB error"` — you immediately know whether it's an API key issue or a database issue.

---

## Part 4: AI Disclosure & Safety

### AI Assistance Used

This project was made with Kiro as a coding assistant. Three specific failures:

**Failure 1 — Broken install.** Kiro pinned the packages we use explicitly but not their transitive dependencies. A fresh install pulled `httpx` 0.28 which had a breaking API change, crashing the app on startup with a `TypeError`. Kiro found the root cause and added `httpx==0.27.2` to requirements.txt.

**Failure 2 — Half the job.** Asked to build the frontend, Kiro added the route in `main.py` to serve `index.html` but never created the file. Server started fine, no errors, until you opened a browser and got a 500. Kiro wrote the file once the crash was reported.

**Failure 3 — Fix caused a regression.** Kiro rewrote the keyword extraction prompt to fix test 10. It worked, but the eval dropped overall because the change hurt other questions. We reverted and documented the failure instead.

---

### Real Risk: Confident Wrong Answers

When retrieval fails, GPT doesn't say "I don't know", it sounds like a judge anyway. In a live test, asked about Blood Moon + Urza's Saga, the system retrieved completely irrelevant chunks and GPT filled the gap with training data. It gave a wrong ruling, cited nothing suspicious, raised no errors. A player trusting that answer in a real game would make the wrong play. The fix is a disclaimer on every answer and showing retrieved chunks by default.
