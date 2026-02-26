# ============================================================
# HyDE Legal Assistant - Indian Constitution
# Hypothetical Document Embeddings for Legal Retrieval
# ============================================================
# Install: pip3 install sentence-transformers requests scikit-learn numpy
# Run:     python3 chatbot.py
# Requires: Ollama running locally with llama3.2 pulled
# ============================================================

import os
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.2"
EMBED_MODEL   = "all-MiniLM-L6-v2"
KNOWLEDGE_DIR = "knowledge_base"
CHUNK_SIZE    = 300
TOP_K         = 3


# ── STEP 1: LOAD & CHUNK DOCUMENTS ──────────────────────────
def load_chunks(directory):
    """Read all .txt files and split into chunks."""
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in paragraphs:
                for i in range(0, len(para), CHUNK_SIZE):
                    chunk = para[i:i + CHUNK_SIZE].strip()
                    if chunk:
                        chunks.append(chunk)
    print(f"[LOADER] {len(chunks)} chunks loaded from '{directory}'")
    return chunks


# ── STEP 2: BUILD VECTOR STORE ───────────────────────────────
def build_vector_store(chunks, embedder):
    """Embed all chunks once at startup and store as numpy array."""
    print("[EMBEDDER] Embedding constitution... (one-time setup)")
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    print(f"[EMBEDDER] Done. {embeddings.shape[0]} chunks embedded.")
    return embeddings


# ── STEP 3: CALL OLLAMA ──────────────────────────────────────
def call_ollama(prompt, max_tokens=300):
    """Send a prompt to local Ollama and return the response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7}
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"].strip()


# ── STEP 4: NORMAL RAG RETRIEVAL (for comparison) ────────────
def normal_retrieve(question, chunks, chunk_embeddings, embedder):
    """Standard RAG: embed the raw question directly."""
    raw_embedding = embedder.encode([question])
    similarities = cosine_similarity(raw_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:TOP_K]
    return [(chunks[i], similarities[i]) for i in top_indices]


# ── STEP 5: HyDE RETRIEVAL ───────────────────────────────────
def hyde_retrieve(question, chunks, chunk_embeddings, embedder):
    """
    HyDE logic:
    1. Ask LLM to write a hypothetical legal answer.
    2. Embed that hypothetical answer (not the raw question).
    3. Find real chunks most similar to that embedding.
    """

    # 5a. Generate hypothetical legal answer
    hypo_prompt = (
        f"You are a legal expert on the Indian Constitution. "
        f"Write a short, detailed legal paragraph that directly answers: '{question}'. "
        f"Use formal legal language and reference constitutional provisions if possible. "
        f"Do not say you don't know — write your best answer.\n\nAnswer:"
    )
    print("\n[HyDE] Generating hypothetical legal answer...")
    hypothetical_answer = call_ollama(hypo_prompt, max_tokens=150)
    print(f"[HyDE] Hypothetical answer:\n  {hypothetical_answer}\n")

    # 5b. Embed the hypothetical answer
    hypo_embedding = embedder.encode([hypothetical_answer])

    # 5c. Cosine similarity against all chunk embeddings
    similarities = cosine_similarity(hypo_embedding, chunk_embeddings)[0]

    # 5d. Pick top-k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:TOP_K]
    retrieved = [(chunks[i], similarities[i]) for i in top_indices]

    return retrieved, hypothetical_answer


# ── STEP 6: GENERATE FINAL LEGAL ANSWER ──────────────────────
def generate_final_answer(question, retrieved_chunks):
    """Use retrieved constitutional articles as context for final answer."""
    context = "\n\n".join([f"- {chunk}" for chunk, _ in retrieved_chunks])
    final_prompt = (
        f"You are a legal assistant specializing in the Indian Constitution. "
        f"Use ONLY the constitutional articles provided below to answer the question. "
        f"Cite the relevant article numbers in your answer.\n\n"
        f"Constitutional Context:\n{context}\n\n"
        f"Question: {question}\n\nLegal Answer:"
    )
    return call_ollama(final_prompt, max_tokens=300)


# ── MAIN CHAT LOOP ───────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HyDE Legal Assistant — Indian Constitution")
    print("  Powered by HyDE + Ollama + sentence-transformers")
    print("=" * 60)

    chunks = load_chunks(KNOWLEDGE_DIR)
    if not chunks:
        print("ERROR: No .txt files found in knowledge_base/")
        return

    print(f"[EMBEDDER] Loading '{EMBED_MODEL}'...")
    embedder = SentenceTransformer(EMBED_MODEL)
    chunk_embeddings = build_vector_store(chunks, embedder)

    print("\nAsk any legal question (e.g. 'arrest?', 'education rights?', 'equality?')")
    print("Type 'quit' to exit\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        print("\n" + "─" * 60)

        # ── COMPARISON: Normal RAG vs HyDE ──
        print("\n[NORMAL RAG] Retrieving using raw query directly...")
        normal_results = normal_retrieve(question, chunks, chunk_embeddings, embedder)
        print("[NORMAL RAG] Top chunks found:")
        for i, (chunk, score) in enumerate(normal_results, 1):
            print(f"  [{i}] score={score:.3f} → {chunk[:100]}...")

        # ── HyDE Retrieval ──
        hyde_results, _ = hyde_retrieve(question, chunks, chunk_embeddings, embedder)
        print("[HyDE] Top chunks found:")
        for i, (chunk, score) in enumerate(hyde_results, 1):
            print(f"  [{i}] score={score:.3f} → {chunk[:100]}...")

        # ── Final Answer using HyDE retrieved chunks ──
        print("\n[GENERATING FINAL LEGAL ANSWER using HyDE context]")
        answer = generate_final_answer(question, hyde_results)
        print(f"\n⚖️  Legal Assistant: {answer}\n")
        print("─" * 60)


if __name__ == "__main__":
    main()