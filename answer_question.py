"""
Retrieval + generation: answer a customer question using precedent tickets.

Flow:
  1. Embed the incoming question with the same model/config used at index time.
  2. Search Qdrant for the most similar problem_text chunks (optionally filtered
     by product_sku / status / etc).
  3. Pull linked_resolution off the top matches.
  4. Feed the question + retrieved resolutions to an Ollama chat model, asking
     it to answer using only that context.
"""

from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

OLLAMA_ADDRESS = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ticket_chunks"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "llama3.1:8b"   # swap for whatever chat model you're serving
TOP_K = 5

ollama_client = Client(OLLAMA_ADDRESS)
qdrant_client = QdrantClient(url=QDRANT_URL)


def embed_query(text: str) -> list[float]:
    response = ollama_client.embed(
        model=EMBED_MODEL,
        input=[text],
        options={"num_ctx": 8192},
        truncate=True,
    )
    return response.embeddings[0]


def search_similar_tickets(
    question: str,
    top_k: int = TOP_K,
    product_sku: str | None = None,
    status: str | None = None,
) -> list[dict]:
    vector = embed_query(question)

    must_conditions = []
    if product_sku:
        must_conditions.append(
            FieldCondition(key="product_sku", match=MatchValue(value=product_sku))
        )
    if status:
        must_conditions.append(
            FieldCondition(key="status", match=MatchValue(value=status))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return [
        {
            "score": point.score,
            "ticket_id": point.payload.get("ticket_id"),
            "problem_text": point.payload.get("problem_text"),
            "linked_resolution": point.payload.get("linked_resolution"),
            "product_title": point.payload.get("product_title"),
        }
        for point in results.points
    ]


def build_prompt(question: str, matches: list[dict]) -> str:
    context_blocks = []
    for i, m in enumerate(matches, 1):
        context_blocks.append(
            f"Precedent {i} (similar issue):\n"
            f"Customer problem: {m['problem_text']}\n"
            f"Resolution: {m['linked_resolution']}\n"
        )
    context = "\n".join(context_blocks)

    return f"""You are a customer support assistant. Answer the customer's question
using ONLY the precedent tickets below. If the precedents don't cover the
question, say you don't have enough information rather than guessing.
Reply in the same language as Customer question.

{context}

Customer question: {question}

Answer:"""


def answer_question(
    question: str,
    product_sku: str | None = None,
    status: str | None = None,
) -> dict:
    matches = search_similar_tickets(question, product_sku=product_sku, status=status)

    if not matches:
        return {
            "answer": "I don't have enough precedent tickets to answer this confidently.",
            "sources": [],
        }

    prompt = build_prompt(question, matches)
    response = ollama_client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response["message"]["content"],
        "sources": [
            {"ticket_id": m["ticket_id"], "score": m["score"]} for m in matches
        ],
    }


if __name__ == "__main__":
    result = answer_question("My computer arrived damaged what do I do?")
    print("Answer:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  ticket {s['ticket_id']} (score={s['score']:.3f})")
