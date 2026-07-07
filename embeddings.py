import sqlite3
import json
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import ollama_address, QDRANT_URL
from ollama import Client

DB_PATH = "chunks.db"
EMBED_MODEL = "bge-m3"
COLLECTION_NAME = "ticket_chunks"
BATCH_SIZE = 16


def get_unembedded_chunks(db_path: str, limit: int = 10_000) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM chunks WHERE embedded = 0 LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = Client(ollama_address)
    response = client.embed(
        model=EMBED_MODEL,
        input=texts,  # list in, list of vectors out — one per input string
        options={"num_ctx": 8192},
    )
    return response.embeddings


def ensure_collection(client: QdrantClient, vector_size: int):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}' (dim={vector_size})")


def mark_chunks_embedded(db_path: str, chunk_ids: list[int]):
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "UPDATE chunks SET embedded = 1 WHERE chunk_id = ?",
        [(cid,) for cid in chunk_ids],
    )
    conn.commit()
    conn.close()


def build_payload(chunk: dict) -> dict:
    """Flatten chunk metadata into a Qdrant payload for filtering + generation."""
    metadata = json.loads(chunk["metadata"] or "{}")
    return {
        "ticket_id": chunk["ticket_id"],
        "chunk_type": chunk["chunk_type"],
        "problem_text": chunk["text"],
        "linked_resolution": chunk["linked_resolution"],
        "product_title": metadata.get("product_title"),
        "product_sku": metadata.get("product_sku"),
        "status": metadata.get("status"),
        "order_status": metadata.get("order_status"),
        "ai_classification": metadata.get("ai_classification"),
        "tags_ids": metadata.get("tags_ids"),
        "uri": metadata.get("uri"),
    }


def main():
    client = QdrantClient(url=QDRANT_URL)
    chunks = get_unembedded_chunks(DB_PATH)
    print(f"Found {len(chunks)} chunks to embed...")

    if not chunks:
        return

    collection_ready = False

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        try:
            vectors = embed_texts(texts)
        except requests.RequestException as e:
            print(f"  Embedding failed for batch starting at {i}: {e}")
            continue

        if not collection_ready:
            ensure_collection(client, vector_size=len(vectors[0]))
            collection_ready = True

        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=vector,
                payload=build_payload(chunk),
            )
            for chunk, vector in zip(batch, vectors)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        mark_chunks_embedded(DB_PATH, [c["chunk_id"] for c in batch])
        print(f"  Embedded + upserted {i + len(batch)}/{len(chunks)}")

    print("Done.")


if __name__ == "__main__":
    main()