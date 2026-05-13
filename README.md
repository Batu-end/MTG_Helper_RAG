# MTG Judge AI

Ask any Magic: The Gathering rules question and get a cited ruling backed by the official Comprehensive Rules.

**Stack:** FastAPI · OpenAI (gpt-4o-mini + text-embedding-3-small) · ChromaDB (local)

**How it works:** Each question goes through a 3-step chain — extract MTG keywords → retrieve the 5 most relevant rule chunks from ChromaDB → GPT answers as a Level 3 Judge and cites the rule numbers it used.

---

## Setup (macOS, Python 3.11+)

**1. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key**
```bash
cp .env.example .env
```
Open `.env` and paste your key
```
OPENAI_API_KEY
```

**4. Ingest the rules** *(one-time, ~2 minutes)*
```bash
python3 ingest.py
```

**5. Start the server**
```bash
uvicorn main:app --reload
```

**6. Open the app**

Go to [http://localhost:8000](http://localhost:8000)

**7. Try a test question**

```bash
Does deathtouch work with trample?
```

You should see a response with keywords, retrieved_rules, and an answer that cites rule numbers.

---

## Evaluation

With the server running, open a second terminal and run:
```bash
source venv/bin/activate
python3 eval/run_eval.py
```

This tests 10 rules questions and checks whether the expected rule number appears in each answer. Example output:

```
MTG Judge Evaluation — 10 test cases
------------------------------------------------------------
  ✓ [01] PASS  rule=702.2  — Deathtouch damage assignment
  ✓ [02] PASS  rule=702.19 — Trample damage assignment over blockers
  ...
------------------------------------------------------------
  Results  : 9 passed / 1 failed / 0 errors
  Accuracy : 90.0%  (9/10 rules correctly cited)
```

---

## Project structure

```
MTG_Helper_RAG/
├── data/mtg_rules.txt       # MTG Comprehensive Rules (included in repo)
├── eval/
│   ├── test_cases.json      # 10 test questions with expected rule citations
│   └── run_eval.py          # Evaluation script
├── static/index.html        # Frontend (Tailwind CSS + vanilla JS)
├── chroma_db/               # Created by ingest.py
├── .env                     # Your API key (not committed)
├── .env.example             # Template — copy this to .env
├── ingest.py                # Chunks, embeds, and stores the rules
├── main.py                  # FastAPI backend
└── requirements.txt
```
