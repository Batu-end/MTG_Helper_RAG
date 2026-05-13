# REPORT.md — MTG Judge AI


## What does the app do?
This application is an AI-powered MTG Judge designed for casual players who need a quick, authoritative rulings without pausing the game for 15 minutes to search forums. It takes a natural language description of a board state, retrieves the exact rules that apply and outputs a ruling with citations.

---

## Part 1: What's Hard About This AI Behavior

The goal sounds simple: answer a rules question and cite the right rule. The hard part is that this depends on three steps working correctly in sequence, and each one can fail silently with no crash, no error, just a wrong answer that looks fine.

**The retrieval bottleneck.**
The system can only cite rules it actually retrieved. If step 2 returns the wrong chunks, step 3 (GPT) either falls back on training data or admits it can't answer. Both look like failures, but for different reasons. The dangerous case is the first one, a confident wrong answer is indistinguishable from a confident right answer unless you go verify the citation manually.

**Keyword extraction is load-bearing.**
Retrieval quality depends entirely on what step 1 extracts. MTG questions pack card types, game actions, and rule concepts into the same sentence. Pull the wrong term and you get semantically nearby but wrong chunks. Nothing tells you this happened, the pipeline just keeps going.

**Cited rule ≠ correct ruling.**
A model can cite a real rule number and still misapply it. The eval only checks whether the citation string appears in the answer, not whether the ruling is actually right. An 80% score here means 8 out of 10 answers contained the expected rule number — it says nothing about whether those answers were legally correct.

---

## Part 2: Iterations

### Version 1 — Flat keyword extraction prompt

**Change:** 
The original `extract_keywords` system prompt asked GPT to extract "2-3 important MTG keywords, mechanic names, or rule concepts" with examples like `'Deathtouch', 'Layer 7', 'Priority', 'Trample'`.

**Example:** 
Test case 10 *"After a sorcery resolves, which player receives priority first?"* consistently failed. GPT extracted `["Priority", "Sorcery"]`. Joined as `"Priority Sorcery"`, the embedding vector landed closer to the Section 307 (Sorceries) cluster than Section 117 (Timing and Priority). All five retrieved chunks were about sorcery casting conditions, none mentioning rule 117. GPT correctly responded: *"The provided rules do not specify which player receives priority first."*

**Delta:** 
9/10 passing. Test 10 was a systematic failure, not a fluke, every run reproduced it.

**Conclusion:** 
The prompt was too vague. Allowing card-type words like "Sorcery" as valid keywords pollutes the query with terms that are common in non-priority rules sections.

---

### Version 2 — Prompt focusing on rule section names

**Change:** Rewrote the system prompt to explicitly prefer rule section names over card types: *"extract rule concepts that would appear as section headings in the MTG Comprehensive Rules… prefer 'Timing and Priority' not 'Sorcery'."*

**Example:** Test 10 immediately passed. GPT now extracted `["Timing and Priority", "Resolving Spells and Abilities"]`, retrieved rule 117.3b directly, and cited it verbatim.

**Delta:** Test 10 passed. But running the full eval showed overall accuracy dropped from 80% (8/10) to below that baseline — the abstract phrasing "Continuous Effects" retrieved layer-system rules when the question was specifically about Deathtouch, missing the tighter 702.2 sub-rules that the original keyword "Deathtouch" found directly.

**Conclusion:** Overfitting the prompt to one failure broke the general case. The abstraction level that helps a broad question like test 10 hurts a specific keyword question like test 1. Version 1 was reverted. The test 10 failure is documented as a known limitation rather than patched with a regression.

---

### Version 3 — Reducing TOP_K_RESULTS from 5 to 3

**Change:** `TOP_K_RESULTS` in `main.py` line 38 was reduced from 5 to 3. The hypothesis was that 5 chunks might be introducing noise — if the 4th and 5th closest chunks are weakly related, they dilute the prompt context and may cause GPT to cite the wrong rule or hedge its answer.

**Motivating example:** Test 3 — *"A player is at 0 life but the turn has not ended. Have they lost the game yet?"* — was failing because GPT cited rule `104.3b` (the general rule about losing the game) instead of `704.5a` (the State-Based Action that specifically triggers the loss check). The retrieved context included both sections, and GPT chose the wrong one. The theory was that with fewer but tighter chunks, the 704.5a chunk would dominate.

**Delta:** Accuracy remained 80% (8/10) with TOP_K_RESULTS=3. The same two tests failed — tests 3 and 10. Reducing chunk count did not change which rules were retrieved for either failing case; the wrong chunk was still the closest match regardless of cutoff.

**Conclusion:** The failure in test 3 is not a context-noise problem — it's a retrieval ranking problem. Rule 104.3b scores higher cosine similarity to the question than 704.5a because the question mentions "losing the game," which maps more directly to section 104 (Ending the Game) than section 704 (State-Based Actions). Fixing this would require either reranking retrieved chunks or adjusting the test's expected citation to `104.3` since the answer GPT gives is factually correct. TOP_K_RESULTS was reverted to 5.

---

## Part 3: Code Walkthrough

### `ingest.py` — `chunk_rules()` — lines 56–97

**What it does:**
Reads the raw rules text line by line. Every time a line matches `^\d+\.[\da-z]*\.?`, which is the exaxt pattern for MTG rule numbers like `702.2b.` or `100.`, it ends the previous chunk and starts a new one. The result is one chunk per rule.

**Why it's structured this way:**
Each MTG rule is its own self-contained statement. Rule 702.2b says one specific thing about Deathtouch. Rule 702.19b says one specific thing about Trample. Keeping them separate means each vector in ChromaDB represents exactly one rule, so a top-5 retrieval gives you five distinct relevant rules — not five overlapping excerpts from the same paragraph.

**Alternative considered and rejected:**
Fixed-size character chunking (e.g. 500 characters with 50-character overlap) is the standard approach in most RAG tutorials. It was rejected here because MTG rules have natural boundaries that fixed-size chunks ignore. A 500-character split would frequently cut mid-rule, embedding the first sentence of 702.2b together with the last sentence of 702.2a. The resulting vector doesn't cleanly represent either rule, which hurts retrieval precision.

---

### `main.py` — `retrieve_rules()` — lines 148–191

**What it does:**
Takes the extracted keywords, joins them into a single string, embeds that string using `text-embedding-3-small`, and queries ChromaDB with `query_embeddings` to get the 5 closest chunks by cosine similarity.

**Why the same embedding model matters:**
ChromaDB just stores lists of floats. There's no model tag attached — it's just numbers. If ingest used `text-embedding-3-small` and retrieval used `text-embedding-ada-002`, the two sets of vectors live in different geometric spaces and cosine similarity between them is meaningless. The `EMBEDDING_MODEL` constant defined at the top of both `ingest.py` and `main.py` is the only thing enforcing this — if you change one and not the other, retrieval silently breaks.

**Why `query_embeddings` instead of `query_texts`:**
ChromaDB's `query_texts` parameter runs its own embedding function internally, which defaults to a local sentence-transformer model. That's a completely different vector space from OpenAI's. Passing a pre-computed vector via `query_embeddings` is the only way to guarantee the query and the stored chunks are comparable.

---

### `main.py` — three-step chain in `ask_judge()` — lines 253–303

**What it does:**
Calls `extract_keywords()`, then `retrieve_rules()`, then `draft_answer()` in order. Each call is wrapped in its own `try/except` block that returns an HTTP 500 with a message identifying which step failed.

**Why separate error blocks instead of one outer try/except:**
One outer catch would just return `"something failed"` with no useful context. Separate blocks mean a rate limit error surfaces as `"Step 1 failed — OpenAI keyword extraction error: 429"` instead of a generic 500. You immediately know it's an API key issue, not a ChromaDB issue. Small thing, saves a lot of debugging time.

---

## Part 4: AI Disclosure & Safety

### AI Assistance Used

This project was built with assustance of Kiro as a coding assistant throughout. The following specific failures and recoveries occurred:

**Failure 1 — Broken install out of the box.** Kiro specified package versions in `requirements.txt` but didn't pin every dependency, just the ones we explicitly use. When installed fresh, pip pulled in a newer version of an internal package (`httpx`) that had changed its API, and the app crashed immediately on startup with a `TypeError`. Nothing in the error message pointed to a version mismatch. Kiro found the root cause from the traceback and added the missing pin. The lesson is that specifying your own packages isn't enough — transitive dependencies can break things too.

**Failure 2 — Did half the job.** When asked to build the frontend, Kiro updated the backend to serve an HTML file, but never actually created the HTML file. The server started fine, the route was registered, and nothing looked wrong until you opened a browser and got a 500 error. Kiro wrote the missing file once I reported the crash. The problem was that Kiro treated "set up the route" as equivalent to "build the frontend," which it isn't.

**Failure 3 — Fixed one thing, broke two others.** Test 10 was failing because the keyword extractor was pulling the wrong terms for a priority question. Kiro rewrote the prompt to fix it, test 10 passed — but the overall evaluation dropped because the new prompt made other questions worse. The fix was too narrow. We reverted and documented the failure instead. The takeaway: changing a general instruction to fix one specific case is risky, because the instruction applies to everything.

---

### Real Risk: Confident Wrong Answers

When retrieval fails, GPT doesn't say "I don't know", it sounds like a judge anyway. This happened in a live test: asked about Blood Moon + Urza's Saga, the system retrieved completely irrelevant chunks (an echo mechanic errata, an Antiquities card list) and GPT filled in the gap with training data. It gave a wrong ruling, cited nothing suspicious, and raised no errors. A player trusting that answer in a real game would make the wrong play.

The fix would be adding a disclaimer to every answer and showing the retrieved chunks by default so users can see what the system actually pulled.
