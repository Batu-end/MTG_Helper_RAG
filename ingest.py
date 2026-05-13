"""
ingest.py — MTG Rules Ingestion Script
=======================================
Run this script ONCE (or whenever you update the rules file) to:
  1. Read and chunk the raw MTG rules text.
  2. Embed each chunk using OpenAI.
  3. Store the chunks + embeddings in a local ChromaDB collection.

Usage:
    python ingest.py

Prerequisites:
    - A .env file with OPENAI_API_KEY=sk-...
    - The rules text file at data/mtg_rules.txt
"""

import os
import re
import time

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

# Path to the raw MTG rules text file you provide.
RULES_FILE = "data/mtg_rules.txt"

# ChromaDB will persist its data here (a folder on disk, not a server).
CHROMA_DIR = "./chroma_db"

# Name of the collection inside ChromaDB where we store the rule chunks.
COLLECTION_NAME = "mtg_rules"

# OpenAI embedding model.
# "text-embedding-3-small" is fast, cheap, and plenty accurate for this use case.
EMBEDDING_MODEL = "text-embedding-3-small"

# How many chunks to send to OpenAI in a single API call.
# OpenAI can handle batches — this avoids making one HTTP request per chunk.
BATCH_SIZE = 100


# ── Step 1: Load environment variables ───────────────────────────────────────

load_dotenv()  # Reads your .env file and puts OPENAI_API_KEY into os.environ

openai_client = OpenAI()  # Automatically picks up OPENAI_API_KEY from the environment


# ── Step 2: Read the rules file ───────────────────────────────────────────────

def load_rules_text(filepath: str) -> str:
    """Read the entire MTG rules file and return it as a single string."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Rules file not found at '{filepath}'.\n"
            "Please download the MTG Comprehensive Rules and save them there."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"✓ Loaded rules file ({len(text):,} characters)")
    return text


# ── Step 3: Chunk the text by rule numbers ────────────────────────────────────

def chunk_rules(text: str) -> list[dict]:
    """
    Split the rules text into individual rule chunks.

    MTG rules are numbered like:
        100.      ← section header
        100.1.    ← rule
        100.1a.   ← lettered sub-rule
        702.15c.  ← deeper sub-rule

    Strategy:
        We scan line by line. Whenever a line *starts* with a rule number
        (digits, dot, optional digits/letters, dot), that line begins a new
        chunk. Everything up to the next rule number is part of that chunk.

    Returns:
        A list of dicts, each with:
            "id"   — a unique string ID (e.g. "rule_0042")
            "text" — the full text of that rule chunk
    """
    # This pattern matches lines that start with a rule number.
    # Examples that match:  "100.", "100.1.", "702.15c.", "1."
    # Examples that don't:  "Credits", "Glossary", blank lines
    rule_start_pattern = re.compile(r"^\d+\.[\da-z]*\.?", re.MULTILINE)

    lines = text.splitlines(keepends=True)

    chunks = []
    current_chunk_lines = []

    for line in lines:
        if rule_start_pattern.match(line.strip()):
            # This line starts a new rule — save the previous chunk first.
            if current_chunk_lines:
                chunk_text = "".join(current_chunk_lines).strip()
                if chunk_text:  # skip empty chunks
                    chunks.append(chunk_text)
            # Start collecting the new chunk.
            current_chunk_lines = [line]
        else:
            # This line is a continuation of the current chunk.
            current_chunk_lines.append(line)

    # Don't forget the very last chunk after the loop ends.
    if current_chunk_lines:
        chunk_text = "".join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append(chunk_text)

    # Wrap each chunk in a dict with a stable string ID.
    result = [
        {"id": f"rule_{i:04d}", "text": chunk}
        for i, chunk in enumerate(chunks)
    ]

    print(f"✓ Split rules into {len(result):,} chunks")
    return result


# ── Step 4: Embed chunks using OpenAI ────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Call the OpenAI Embeddings API to turn each chunk's text into a vector.

    We send chunks in batches to be efficient. Each chunk dict gets a new
    "embedding" key added, which is a list of floats (the vector).

    OpenAI rate limits are generous for small projects, but we add a small
    sleep between batches just to be polite.
    """
    texts = [chunk["text"] for chunk in chunks]
    total = len(texts)
    print(f"Embedding {total} chunks in batches of {BATCH_SIZE}...")

    all_embeddings = []

    for batch_start in range(0, total, BATCH_SIZE):
        batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, total)

        print(f"  → Embedding chunks {batch_start + 1}–{batch_end} / {total}")

        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts,
        )

        # The API returns embeddings in the same order as the input.
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # Small pause to avoid hammering the API (not strictly needed here).
        if batch_end < total:
            time.sleep(0.5)

    # Attach each embedding back to its chunk dict.
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    print(f"✓ Finished embedding {total} chunks")
    return chunks


# ── Step 5: Store in ChromaDB ─────────────────────────────────────────────────

def store_in_chromadb(chunks: list[dict]) -> None:
    """
    Persist all chunks (text + embeddings) into a local ChromaDB collection.

    ChromaDB stores data on disk at CHROMA_DIR so it survives between runs.
    We use get_or_create_collection so re-running the script won't crash —
    but be aware it will ADD duplicates if you run it twice on the same data.
    For a clean re-ingest, delete the ./chroma_db folder first.
    """
    # PersistentClient saves everything to disk automatically.
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # get_or_create_collection: creates the collection if it doesn't exist yet.
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        # IMPORTANT: tell ChromaDB we're supplying our own embeddings.
        # If we omitted this, ChromaDB would try to embed the text itself
        # using its default model, which would be a different vector space.
        metadata={"hnsw:space": "cosine"},  # cosine similarity works well for text
    )

    # ChromaDB's add() expects three parallel lists.
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]

    # Add in batches to avoid memory issues with very large rule sets.
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(chunks))
        print(f"  → Storing chunks {batch_start + 1}–{batch_end} into ChromaDB")

        collection.add(
            ids=ids[batch_start:batch_end],
            documents=documents[batch_start:batch_end],
            embeddings=embeddings[batch_start:batch_end],
        )

    print(f"✓ Stored {len(chunks):,} chunks in collection '{COLLECTION_NAME}'")
    print(f"  Database saved to: {os.path.abspath(CHROMA_DIR)}")


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    print("=== MTG Rules Ingestion ===\n")

    # 1. Read the raw rules text from disk.
    raw_text = load_rules_text(RULES_FILE)

    # 2. Split the text into one chunk per rule.
    chunks = chunk_rules(raw_text)

    # 3. Embed each chunk using OpenAI (this costs a small amount of API credit).
    chunks = embed_chunks(chunks)

    # 4. Save everything to the local ChromaDB database.
    store_in_chromadb(chunks)

    print("\n✅ Ingestion complete! You can now run the API.")


if __name__ == "__main__":
    main()
