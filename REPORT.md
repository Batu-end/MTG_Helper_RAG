# REPORT.md — MTG Judge AI

---

## Part 1: What's Hard About This AI Behavior

The goal sounds simple: answer rules questions accurately and cite the right rule. The hard part is that accuracy depends on three steps all working correctly *in sequence*, and each step can silently fail in ways that look like success.

**The retrieval bottleneck.** The system can only cite rules that were actually retrieved. If Step 2 returns the wrong five chunks, Step 3 (GPT) has no choice but to either answer from training data (ignoring the RAG entirely) or admit it can't answer. Both outcomes look like failures, but for different reasons — and from the outside, a confidently wrong answer is indistinguishable from a confidently right one.

**Keyword extraction is load-bearing.** The quality of the retrieval depends entirely on what Step 1 extracts. MTG questions often contain card types, game actions, and rule concepts in the same sentence. Extracting the wrong one — say, "Sorcery" instead of "Timing and Priority" — pulls semantically nearby but wrong chunks. The system fails downstream with no error signal.

**Cited rule ≠ correct ruling.** A model can cite a real rule number and still give a wrong ruling by misapplying it. Our evaluation metric only checks citation presence, not correctness. This means a 90% eval score overstates actual ruling accuracy.

---

## Part 2: Iterations

### Version 1 — Flat keyword extraction prompt

**Change:** The original `extract_keywords` system prompt asked GPT to extract "2-3 important MTG keywords, mechanic names, or rule concepts" with examples like `'Deathtouch', 'Layer 7', 'Priority', 'Trample'`.

**Motivating example:** Test case 10 — *"After a sorcery resolves, which player receives priority first?"* — consistently failed. GPT extracted `["Priority", "Sorcery"]`. Joined as `"Priority Sorcery"`, the embedding vector landed closer to the Section 307 (Sorceries) cluster than Section 117 (Timing and Priority). All five retrieved chunks were about sorcery casting conditions, none mentioning rule 117. GPT correctly responded: *"The provided rules do not specify which player receives priority first."*

**Delta:** 9/10 passing. Test 10 was a systematic failure, not a fluke — every run reproduced it.

**Conclusion:** The prompt was too permissive. Allowing card-type words like "Sorcery" as valid keywords pollutes the query with terms that are common in non-priority rules sections.

---

### Version 2 — Section-heading-biased extraction prompt

**Change:** Rewrote the system prompt to explicitly prefer rule section names over card types: *"extract rule concepts that would appear as section headings in the MTG Comprehensive Rules… prefer 'Timing and Priority' not 'Sorcery'."*

**Motivating example:** Test 10 immediately passed. GPT now extracted `["Timing and Priority", "Resolving Spells and Abilities"]`, retrieved rule 117.3b directly, and cited it verbatim.

**Delta:** Test 10 passed. But a manual spot-check of tests 1–3 showed degradation — the abstract phrasing "Continuous Effects" retrieved layer-system rules when the question was specifically about Deathtouch, missing the tighter 702.2 sub-rules. Overall accuracy dropped below the Version 1 baseline.

**Conclusion:** Overfitting the prompt to one failure broke the general case. The abstraction level that helps a broad question like test 10 hurts a specific keyword question like test 1. Version 1 was reverted. The test 10 failure is documented as a known limitation rather than patched with a regression.

---

### Version 3 — Chunk batching in ingest.py

**Change:** Initial draft of `ingest.py` sent one embedding API call per rule chunk. Revised to batch 100 chunks per call with a 0.5s pause between batches.

**Motivating example:** The MTG Comprehensive Rules produced 3,511 chunks after splitting. At one call per chunk, that would be 3,511 sequential HTTP requests. With rate limits, this would take 30+ minutes and risk 429 errors mid-ingest.

**Delta:** Ingestion completed in approximately 2 minutes with zero rate-limit errors. Total API cost was under $0.05.

**Conclusion:** Batching is mandatory for any corpus of this size. The 0.5s inter-batch pause was conservative but reliable; it could be removed for faster ingestion at the cost of occasional retries.

---

## Part 3: Code Walkthrough

### `ingest.py` — `chunk_rules()` — lines 56–97

**What it does:** Reads the raw rules text line by line and starts a new chunk whenever a line matches `^\d+\.[\da-z]*\.?` — the pattern for MTG rule numbers like `702.2b.` or `100.`. Everything between two rule-number lines is kept together as one chunk.

**Why it's structured this way:** Each MTG rule is a self-contained semantic unit. Rule 702.2b says one specific thing about Deathtouch. Splitting there means each vector in ChromaDB represents exactly one rule, so a top-5 retrieval returns five distinct relevant rules rather than five excerpts from the same paragraph.

**Alternative considered and rejected:** Fixed-size character chunking (e.g. 500 characters with 50-character overlap) is the most common RAG chunking strategy. It was rejected here because MTG rules have natural, meaningful boundaries. A fixed-size split would routinely cut a rule in half — embedding the first sentence of 702.2b with the last sentence of 702.2a — producing a vector that doesn't cleanly represent either rule. Rule-boundary chunking is harder to implement but produces semantically cleaner retrievals.

---

### `main.py` — `retrieve_rules()` — lines 148–191

**What it does:** Joins the extracted keywords into a single string, embeds it with `text-embedding-3-small` (the same model used during ingest), and queries ChromaDB with `query_embeddings` to get the 5 closest chunks by cosine similarity.

**Why the same embedding model matters:** ChromaDB stores raw float vectors. There is no model metadata attached — it's just numbers. If ingest used `text-embedding-3-small` and retrieval used `text-embedding-ada-002`, the vectors would be in different geometric spaces and cosine similarity scores would be meaningless. The `EMBEDDING_MODEL` constant in both files is the only thing enforcing this contract.

**Why `query_embeddings` instead of `query_texts`:** ChromaDB's `query_texts` parameter uses its own built-in embedding function, which defaults to a local sentence-transformer model — a completely different vector space than OpenAI's. Using `query_embeddings` with our pre-computed vector is the only way to guarantee the query and the stored chunks are in the same space.

---

### `main.py` — three-step chain in `ask_judge()` — lines 253–303

**What it does:** Calls `extract_keywords()`, then `retrieve_rules()`, then `draft_answer()` in sequence, with a separate `try/except` block around each step that returns an HTTP 500 with a step-specific error message.

**Why separate error blocks instead of one outer try/except:** A single outer catch would return `"Step failed"` with no indication of where. A grader or developer hitting a rate limit error can immediately see `"Step 1 failed — OpenAI keyword extraction error: 429"` and know the issue is the API key, not ChromaDB. Debugging time drops significantly.

---

## Part 4: AI Disclosure & Safety

### AI Assistance Used

This project was built with Claude Code (claude-sonnet-4-6) as a coding assistant throughout. The following specific failures and recoveries occurred:

**Failure 1 — Incompatible dependency versions.** Claude generated `requirements.txt` with `openai==1.51.0` without pinning `httpx`. When installed, `pip` resolved `httpx` to 0.28.0, which removed the `proxies` constructor argument that `openai` 1.51 still passes internally. The error was a `TypeError` at import time, not a clear dependency conflict message. Recovery: Claude diagnosed the root cause from the traceback and added `httpx==0.27.2` to requirements. **Lesson:** Pinned versions without a lock file still leave transitive dependencies unconstrained.

**Failure 2 — Incomplete task delivery.** Claude was asked to build the frontend (Task 3) and edited `main.py` to mount `static/` and serve `index.html` — but never actually wrote `index.html`. The server started without error, but every request to `/` returned a 500. The missing file was only discovered when the app was first opened in a browser. Recovery: Claude wrote the file when the 500 was reported. **Lesson:** Claude completed the setup work (routing, imports) and omitted the payload (the actual HTML file), a failure mode where partial completion looks like success.

**Failure 3 — Prompt fix that caused a regression.** Claude diagnosed the test 10 failure (priority question retrieving sorcery rules), identified the cause correctly, and proposed a prompt change that fixed test 10 but degraded 2–3 other tests. The fix was overfit to the failing example. Recovery: the original prompt was restored and the failure was documented as a known limitation. **Lesson:** LLM prompt tuning has the same overfitting risk as model fine-tuning. Fixing one example by changing a general instruction can break the general case.

---

### Real Safety Risk: Authoritative Tone on Wrong Answers

The most significant safety risk in this application is not hallucination in the conventional sense — it is **retrieval failure combined with authoritative framing**.

When ChromaDB returns the wrong rule chunks (as demonstrated by test 10 in Version 1), GPT has two possible behaviors: it either answers from training data while appearing to cite retrieved rules, or it correctly says the provided rules are insufficient. The second behavior is safe. The first is dangerous.

The system prompt instructs GPT to act as a *Level 3 MTG Judge* and to cite rules explicitly. This framing produces confident, authoritative-sounding prose. A player at a tournament who receives a ruling like *"Rule 702.2b clearly states that lethal damage requires only 1 point…"* is likely to trust it — even if the retrieved chunk was from the wrong section and the ruling is wrong.

The concrete harm: a player makes an incorrect play in a competitive match based on this tool's ruling, loses a game or match, and has no recourse because the ruling cited a real rule number that the player cannot easily verify mid-game.

**Mitigation not yet implemented:** The response should include a disclaimer on every answer: *"This ruling is AI-generated and not a substitute for an official judge ruling. Verify citations against the current Comprehensive Rules before acting on them in competitive play."* The retrieved chunks should also be shown by default (not collapsed), so the user can see whether the cited rules actually appeared in the retrieval results.
