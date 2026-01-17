import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import ollama

VECTOR_DIR = Path("data/vectors")
CHUNKS_DIR = Path("data/chunks")
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 4


def load_vector_store():
    index = faiss.read_index(str(VECTOR_DIR / "faiss.index"))
    with open(VECTOR_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def load_chunks():
    chunks = {}
    for file in CHUNKS_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            key = (data["metadata"]["source"], data["metadata"]["chunk_id"])
            chunks[key] = data["text"]
    return chunks


def retrieve_context(query: str):
    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode([query])

    index, metadata = load_vector_store()
    chunks = load_chunks()

    distances, indices = index.search(np.array(query_vector), TOP_K)

    retrieved_texts = []
    for idx in indices[0]:
        meta = metadata[idx]
        key = (meta["source"], meta["chunk_id"])
        retrieved_texts.append(chunks[key])

    return "\n\n".join(retrieved_texts)


def generate_answer(query: str):
    context = retrieve_context(query)

    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        answer = generate_answer(q)
        print("\nAnswer:\n", answer)
