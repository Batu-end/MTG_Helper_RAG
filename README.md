# MTG Judge AI

An AI-powered Magic: The Gathering rules judge. Ask any rules question and get a cited ruling backed by the official Comprehensive Rules.

Built with **FastAPI**, **OpenAI** (gpt-4o-mini + text-embedding-3-small), and **ChromaDB** (local vector database).

---

## How it works

Every question goes through a 3-step reasoning chain:

1. **Decompose** — GPT extracts 2–3 MTG keywords from the question
2. **Retrieve** — Those keywords are embedded and used to query ChromaDB for the 5 most relevant rule chunks
3. **Draft** — GPT answers as a Level 3 MTG Judge, citing specific rule numbers from the retrieved chunks

---

## Prerequisites

- Python 3.11 or higher
- An OpenAI API key — get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- The MTG Comprehensive Rules plain-text file — download the `.txt` from [magic.wizards.com/en/rules](https://magic.wizards.com/en/rules)

---

## Setup — macOS

Open Terminal and run each block in order.

**1. Clone the repo and enter the project folder**
```bash
cd ~/Desktop
git clone <your-repo-url> MTG_Helper_RAG
cd MTG_Helper_RAG
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now show `(venv)` at the start.

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI API key**
```bash
cp .env.example .env
```
Open `.env` and replace `your-openai-api-key-here` with your actual key.

**5. Add the MTG rules file**

Save the downloaded `MagicCompRules_*.txt` file as:
```
MTG_Helper_RAG/data/mtg_rules.txt
```

**6. Run the ingestion script** *(one-time, takes ~2 minutes)*
```bash
python3 ingest.py
```

You should see: `✅ Ingestion complete! You can now run the API.`

**7. Start the web server**
```bash
uvicorn main:app --reload
```

**8. Open the app**

Go to [http://localhost:8000](http://localhost:8000) in your browser.

**9. Run the evaluation script** *(in a second terminal tab, with the server still running)*
```bash
source venv/bin/activate
python3 eval/run_eval.py
```

---

## Setup — Windows

Open **Command Prompt** (not PowerShell) and run each block in order.

**1. Clone the repo and enter the project folder**
```bat
cd %USERPROFILE%\Desktop
git clone <your-repo-url> MTG_Helper_RAG
cd MTG_Helper_RAG
```

**2. Create and activate a virtual environment**
```bat
python -m venv venv
venv\Scripts\activate
```

Your prompt should now show `(venv)` at the start.

**3. Install dependencies**
```bat
pip install -r requirements.txt
```

**4. Add your OpenAI API key**
```bat
copy .env.example .env
```
Open `.env` in Notepad and replace `your-openai-api-key-here` with your actual key.

**5. Add the MTG rules file**

Save the downloaded `MagicCompRules_*.txt` file as:
```
MTG_Helper_RAG\data\mtg_rules.txt
```

**6. Run the ingestion script** *(one-time, takes ~2 minutes)*
```bat
python ingest.py
```

You should see: `✅ Ingestion complete! You can now run the API.`

**7. Start the web server**
```bat
uvicorn main:app --reload
```

**8. Open the app**

Go to [http://localhost:8000](http://localhost:8000) in your browser.

**9. Run the evaluation script** *(open a second Command Prompt window, with the server still running)*
```bat
cd %USERPROFILE%\Desktop\MTG_Helper_RAG
venv\Scripts\activate
python eval/run_eval.py
```

---

## Project structure

```
MTG_Helper_RAG/
├── data/
│   └── mtg_rules.txt        # MTG Comprehensive Rules (you provide this)
├── eval/
│   ├── test_cases.json      # 10 test questions with expected rule citations
│   └── run_eval.py          # Evaluation script — measures citation accuracy
├── static/
│   └── index.html           # Frontend UI (vanilla JS + Tailwind CSS)
├── chroma_db/               # Created automatically by ingest.py
├── .env                     # Your OpenAI API key (never commit this)
├── ingest.py                # One-time script: chunks, embeds, and stores rules
├── main.py                  # FastAPI backend with the 3-step reasoning chain
└── requirements.txt
```

---

## Troubleshooting

**`OPENAI_API_KEY` not found**
Make sure `.env` is in the project root (same folder as `main.py`) and contains your key with no extra spaces.

**`proxies` TypeError on startup**
Run `pip install httpx==0.27.2` then retry. This is a known version conflict between openai and httpx 0.28+.

**`Collection 'mtg_rules' does not exist`**
The ingestion script hasn't been run yet, or it failed partway through. Run `python3 ingest.py` (macOS) or `python ingest.py` (Windows) first.

**`rules_in_db: 0` on `/health`**
Same as above — re-run the ingestion script.

**Eval script can't connect**
The web server must be running in a separate terminal before you run `run_eval.py`.
