from dotenv import load_dotenv
import os
from prompts import prompt

from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

OLLAMA_ADDRESS = "http://ollama-container:11434"
QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "ticket_chunks"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "llama3.1:8b"
TOP_K = 5
MAX_VERBATIM_TURNS = 8  # keep the last N turns in full; summarize anything older

ollama_client = Client(OLLAMA_ADDRESS)
qdrant_client = QdrantClient(url=QDRANT_URL)

SYSTEM_INSTRUCTIONS = """You are a customer support assistant helping an agent respond to a customer.
Your task is to create a response to the last message by the customer.

[CRITICAL INSTRUCTION]
Regardless of the language used by the customer or the retrieved tickets, you MUST write your entire final response in French. This is a strict rule.

[GUIDELINES]
1. Use the retrieved precedent tickets as reference for likely resolutions.
2. Prioritize what the customer has actually said in this conversation.
3. If the precedents don't cover the current situation, explicitly state so in French rather than guessing.

[OUTPUT FORMAT]
Response (in French):"""


class Conversation:
    def __init__(self, product_sku: str | None = None, status: str | None = None):
        self.turns: list[dict] = []       # [{"role": "user"|"assistant", "content": ...}]
        self.summary: str | None = None   # rolling summary of turns older than the window
        self.product_sku = product_sku
        self.status = status

    def add_customer_message(self, text: str):
        self.turns.append({"role": "user", "content": text})

    def add_agent_message(self, text: str):
        self.turns.append({"role": "assistant", "content": text})

    def _embed(self, text: str) -> list[float]:
        response = ollama_client.embed(
            model=EMBED_MODEL, input=[text], options={"num_ctx": 8192}, truncate=True
        )
        return response.embeddings[0]

    def _retrieve(self, query_text: str) -> list[dict]:
        vector = self._embed(query_text)
        must = []
        if self.product_sku:
            must.append(FieldCondition(key="product_sku", match=MatchValue(value=self.product_sku)))
        if self.status:
            must.append(FieldCondition(key="status", match=MatchValue(value=self.status)))
        query_filter = Filter(must=must) if must else None

        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=query_filter,
            limit=TOP_K,
        )
        return [
            {
                "ticket_id": p.payload.get("ticket_id"),
                "problem_text": p.payload.get("problem_text"),
                "linked_resolution": p.payload.get("linked_resolution"),
            }
            for p in results.points
        ]

    def _maybe_compress_history(self):
        """Once the verbatim window is exceeded, summarize the oldest turns into self.summary."""
        if len(self.turns) <= MAX_VERBATIM_TURNS:
            return

        overflow = self.turns[: len(self.turns) - MAX_VERBATIM_TURNS]
        self.turns = self.turns[len(self.turns) - MAX_VERBATIM_TURNS :]

        overflow_text = "\n".join(f"{t['role']}: {t['content']}" for t in overflow)
        prior_summary = f"Earlier summary: {self.summary}\n\n" if self.summary else ""

        prompt = f"""{prior_summary}Summarize the key facts from this part of a support
conversation in 3-4 sentences — what the customer's issue is and what's been
tried so far. Be factual, no speculation.

{overflow_text}

Summary:"""

        response = ollama_client.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
        self.summary = response["message"]["content"].strip()

    def get_agent_reply(self, customer_message: str) -> dict:
        self.add_customer_message(customer_message)
        self._maybe_compress_history()

        # Retrieve using the latest customer message; for short/ambiguous messages,
        # combining with the last couple of turns tends to retrieve better.
        recent_context = " ".join(
            t["content"] for t in self.turns[-3:] if t["role"] == "user"
        )
        matches = self._retrieve(recent_context)

        context_blocks = [
            f"Precedent {i+1}: Customer problem: {m['problem_text']}\nResolution: {m['linked_resolution']}"
            for i, m in enumerate(matches)
        ]
        retrieved_context = "\n\n".join(context_blocks) if context_blocks else "No close precedents found."

        summary_block = f"Summary of earlier conversation: {self.summary}\n\n" if self.summary else ""

        system_content = f"""{SYSTEM_INSTRUCTIONS}

        {summary_block}Relevant precedent tickets:
        {retrieved_context}"""

        messages = [
            {"role": "system", "content": system_content},
            *self.turns,
        ]

        response = ollama_client.chat(model=CHAT_MODEL, messages=messages)
        reply = response["message"]["content"]
        self.add_agent_message(reply)

        return {
            "reply": reply,
            "sources": [m["ticket_id"] for m in matches],
        }

def gemini_call(last_message, history, reprompt_instructions=None):
    convo = Conversation()
    for msg in history:
        if msg.role == "Customer":
            convo.add_customer_message(msg.text)
        else:
            convo.add_agent_message(msg.text)
    
    result = convo.get_agent_reply(last_message)

    return result

if __name__ == "__main__":
    convo = Conversation()

    result = convo.get_agent_reply("My PC arrived with a broken fan.")
    print("Agent:", result["reply"])
    print("Sources:", result["sources"])

    result = convo.get_agent_reply("It's the case fan, not the CPU cooler.")
    print("\nAgent:", result["reply"])
    print("Sources:", result["sources"])
