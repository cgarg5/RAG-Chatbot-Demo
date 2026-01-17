from pathlib import Path
from typing import List, Dict

CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150     # characters

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def load_text_files() -> Dict[str, str]:
    """Load all processed text files."""
    texts = {}
    for txt_file in PROCESSED_DIR.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            texts[txt_file.stem] = f.read()
    return texts


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def create_chunks():
    texts = load_text_files()

    for doc_name, text in texts.items():
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "source": doc_name,
                    "chunk_id": i
                }
            }

            output_file = CHUNKS_DIR / f"{doc_name}_chunk_{i}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                import json
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        print(f"Created {len(chunks)} chunks for {doc_name}")


if __name__ == "__main__":
    create_chunks()

    import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIR = Path("data/vectors")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def load_chunks():
    """Load chunk JSON files."""
    chunks = []
    for chunk_file in CHUNKS_DIR.glob("*.json"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks.append(data)
    return chunks


def embed_and_store():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Save index
    faiss.write_index(index, str(VECTOR_DIR / "faiss.index"))

    # Save metadata separately
    with open(VECTOR_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump([chunk["metadata"] for chunk in chunks], f, indent=2)

    print("Vector store created and saved.")


if __name__ == "__main__":
    embed_and_store()

