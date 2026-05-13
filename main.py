"""
main.py — MTG Judge FastAPI Backend
=====================================
Exposes a single /ask endpoint that answers MTG rules questions using a
3-step reasoning chain:

    Step 1 — Decomposition : Extract 2-3 MTG keywords from the question.
    Step 2 — Retrieval     : Embed those keywords and query ChromaDB for the
                             top 5 most relevant rule chunks.
    Step 3 — Drafting      : Feed the original question + retrieved rules to
                             GPT, which answers as a Level 3 MTG Judge and
                             cites specific rule numbers.

Run with:
    uvicorn main:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Any

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()  # Read OPENAI_API_KEY (and any other vars) from .env

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "mtg_rules"
EMBEDDING_MODEL = "text-embedding-3-small"   # Must match what ingest.py used
CHAT_MODEL = "gpt-4o-mini"
TOP_K_RESULTS = 5  # How many rule chunks to retrieve from ChromaDB


# ── Pydantic Models ───────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """The JSON body the client sends to /ask."""
    question: str = Field(
        ...,
        min_length=5,
        description="A Magic: The Gathering rules question.",
        examples=["Does deathtouch work with trample?"],
    )


class AskResponse(BaseModel):
    """The JSON body returned by /ask."""
    question: str = Field(description="The original question, echoed back.")
    keywords: list[str] = Field(
        description="Keywords extracted from the question in Step 1."
    )
    retrieved_rules: list[str] = Field(
        description="The raw rule chunks retrieved from ChromaDB in Step 2."
    )
    answer: str = Field(
        description="The judge's answer with rule citations from Step 3."
    )


# ── App Lifespan (startup / shutdown) ─────────────────────────────────────────
#
# We initialise the OpenAI client and ChromaDB collection once at startup and
# store them on `app.state` so every request can reuse the same connections.
# This is far more efficient than creating new clients per request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Called once when the server starts; cleans up when it stops."""
    # --- Startup ---
    app.state.openai = OpenAI()  # Reads OPENAI_API_KEY from environment

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    # NOTE: The collection must already exist — run ingest.py first.
    app.state.collection = chroma_client.get_collection(name=COLLECTION_NAME)

    print(f"✓ ChromaDB collection '{COLLECTION_NAME}' loaded")
    print(f"  Contains {app.state.collection.count():,} rule chunks")

    yield  # Hand control back to FastAPI; the server is now running.

    # --- Shutdown (nothing special needed here) ---
    print("Server shutting down.")


app = FastAPI(
    title="MTG Judge API",
    description="Answers Magic: The Gathering rules questions using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Step 1: Decomposition ─────────────────────────────────────────────────────

def extract_keywords(question: str, openai_client: OpenAI) -> list[str]:
    """
    Ask GPT to identify the 2-3 most important MTG keywords in the question.

    We ask for JSON output so we can parse it reliably instead of trying to
    split a freeform sentence. The `json_object` response format guarantees
    GPT returns valid JSON (it will never add markdown fences, etc.).

    Returns:
        A list of keyword strings, e.g. ["Deathtouch", "Trample"].

    Raises:
        OpenAIError: if the API call fails (caught by the endpoint).
    """
    system_prompt = (
        "You are a Magic: The Gathering rules expert. "
        "Given a rules question, extract the 2-3 most important MTG keywords, "
        "mechanic names, or rule concepts (e.g. 'Deathtouch', 'Layer 7', "
        "'Priority', 'Trample'). "
        "Respond ONLY with a JSON object in this exact format: "
        '{"keywords": ["keyword1", "keyword2"]}'
    )

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},  # Guarantees valid JSON back
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,  # We want deterministic extraction, not creativity
    )

    import json
    data = json.loads(response.choices[0].message.content)
    keywords: list[str] = data.get("keywords", [])

    if not keywords:
        # Fallback: use the raw question if GPT returned an empty list
        keywords = [question]

    return keywords


# ── Step 2: Retrieval ─────────────────────────────────────────────────────────

def retrieve_rules(
    keywords: list[str],
    collection: Any,
    openai_client: OpenAI,
) -> list[str]:
    """
    Embed the extracted keywords and query ChromaDB for the most relevant rules.

    WHY embed the keywords rather than the full question?
    Because our ingest.py stored embeddings of individual rule chunks, which
    are dense with keywords. Matching keyword-to-keyword tends to score
    higher similarity than matching a conversational sentence to a rule chunk.

    We join the keywords into one string so we make a single embedding call.

    Returns:
        A list of up to TOP_K_RESULTS rule text strings.

    Raises:
        OpenAIError: if the embedding API call fails.
    """
    # Join keywords into a single search phrase, e.g. "Deathtouch Trample"
    query_text = " ".join(keywords)

    # Embed the query using the SAME model used in ingest.py.
    # If you use a different model here, the vector spaces won't match and
    # similarity scores will be meaningless.
    embed_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query_text],
    )
    query_vector = embed_response.data[0].embedding

    # Query ChromaDB — returns the n_results closest chunks by cosine distance
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_RESULTS,
        include=["documents"],  # We only need the text, not metadata/distances
    )

    # results["documents"] is a list-of-lists (one list per query vector).
    # Since we sent one query, we take index [0].
    rule_chunks: list[str] = results["documents"][0]
    return rule_chunks


# ── Step 3: Drafting & Citation ───────────────────────────────────────────────

def draft_answer(
    question: str,
    rule_chunks: list[str],
    openai_client: OpenAI,
) -> str:
    """
    Ask GPT to answer the question acting as a Level 3 MTG Judge.

    We include the retrieved rule chunks directly in the prompt so GPT can
    cite them. The system prompt strictly instructs it NOT to use outside
    knowledge — this keeps answers grounded in the actual rules text and
    prevents hallucination of non-existent rules.

    Returns:
        A string containing the judge's ruling with cited rule numbers.

    Raises:
        OpenAIError: if the API call fails.
    """
    # Format the retrieved rules as a numbered list for clarity in the prompt
    formatted_rules = "\n\n".join(
        f"[Rule {i + 1}]\n{chunk}" for i, chunk in enumerate(rule_chunks)
    )

    system_prompt = (
        "You are a Level 3 Magic: The Gathering Judge. "
        "Answer the player's question based STRICTLY on the rules provided below. "
        "Do NOT use knowledge from outside the provided rules. "
        "In your answer, explicitly cite each rule you rely on by its rule number "
        "(e.g. 'Rule 702.2b states...'). "
        "If the provided rules are insufficient to answer the question, say so clearly."
    )

    user_message = (
        f"Player's question:\n{question}\n\n"
        f"Relevant rules:\n{formatted_rules}"
    )

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # Slight creativity so the answer reads naturally
    )

    return response.choices[0].message.content.strip()


# ── Endpoint: POST /ask ───────────────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the MTG Judge a rules question",
)
async def ask_judge(request: AskRequest) -> AskResponse:
    """
    3-step reasoning chain:
      1. Decompose the question into MTG keywords.
      2. Retrieve the most relevant rule chunks from ChromaDB.
      3. Draft a cited ruling using GPT.
    """
    openai_client: OpenAI = app.state.openai
    collection = app.state.collection

    # ── Step 1: Keyword extraction ────────────────────────────────────────────
    try:
        keywords = extract_keywords(request.question, openai_client)
    except OpenAIError as e:
        # OpenAIError is the base class for all openai-sdk exceptions
        # (AuthenticationError, RateLimitError, APIConnectionError, etc.)
        raise HTTPException(
            status_code=500,
            detail=f"Step 1 failed — OpenAI keyword extraction error: {str(e)}",
        )

    # ── Step 2: Rule retrieval ────────────────────────────────────────────────
    try:
        rule_chunks = retrieve_rules(keywords, collection, openai_client)
    except OpenAIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Step 2 failed — OpenAI embedding error: {str(e)}",
        )
    except Exception as e:
        # Catches ChromaDB errors (e.g. collection not found, disk issue)
        raise HTTPException(
            status_code=500,
            detail=f"Step 2 failed — ChromaDB retrieval error: {str(e)}",
        )

    # ── Step 3: Answer drafting ───────────────────────────────────────────────
    try:
        answer = draft_answer(request.question, rule_chunks, openai_client)
    except OpenAIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Step 3 failed — OpenAI answer drafting error: {str(e)}",
        )

    return AskResponse(
        question=request.question,
        keywords=keywords,
        retrieved_rules=rule_chunks,
        answer=answer,
    )


# ── Endpoint: GET /health ─────────────────────────────────────────────────────

@app.get("/health", summary="Check that the server and DB are ready")
async def health_check() -> dict:
    """
    Quick sanity check. Returns the number of rules stored in ChromaDB.
    Useful for confirming ingest.py ran successfully before making /ask calls.
    """
    try:
        count = app.state.collection.count()
        return {"status": "ok", "rules_in_db": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB error: {str(e)}")


# ── Serve the frontend UI ─────────────────────────────────────────────────────
#
# This must come AFTER all API routes are defined. FastAPI matches routes in
# registration order, so the API endpoints above take priority over the static
# file mount. If we registered the mount first, "/" would be claimed by
# StaticFiles before our /ask and /health routes were reachable.

@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    """Serve the single-page frontend at the root URL."""
    return FileResponse("static/index.html")


# Mount CSS/JS/image assets under /static so index.html can reference them
# with paths like /static/logo.png if needed in the future.
app.mount("/static", StaticFiles(directory="static"), name="static")
