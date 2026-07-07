import sqlite3
import json
from main import TicketDatabase


def init_chunks_table(db_path: str):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER NOT NULL,
                chunk_type TEXT,          -- currently always 'problem_resolution'
                text TEXT,                -- problem text: what gets embedded
                linked_resolution TEXT,   -- resolution text: what gets generated from
                metadata TEXT,            -- JSON blob: product, status, ai_classification, etc.
                embedded INTEGER DEFAULT 0,
                FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
            )
        """)


def build_chunk(ticket_with_messages: dict) -> dict | None:
    ticket = ticket_with_messages["ticket"]
    messages = ticket_with_messages["messages"]

    customer_msgs = [m for m in messages if m["sender_role"] == "Customer"]
    agent_msgs = [m for m in messages if m["sender_role"] == "Agent"]

    if not customer_msgs or not agent_msgs:
        return None

    problem_text = "\n".join(m["body"] for m in customer_msgs if m["body"])

    solution_msgs = [m for m in agent_msgs if m["is_solution"]]
    if solution_msgs:
        resolution_text = "\n".join(m["body"] for m in solution_msgs if m["body"])
    else:
        # Fallback: last agent reply in the thread, by created_at order
        # (messages are already ordered by created_at from get_ticket_with_messages)
        resolution_text = agent_msgs[-1]["body"]

    if not problem_text.strip() or not resolution_text.strip():
        return None

    metadata = {
        "product_title": ticket.get("product_title"),
        "product_sku": ticket.get("product_sku"),
        "status": ticket.get("status"),
        "order_status": ticket.get("order_status"),
        "ai_classification": ticket.get("ai_classification"),
        "tags_ids": json.loads(ticket.get("tags_ids") or "[]"),
        "uri": ticket.get("uri"),
        "language": ticket.get("language"),
    }

    return {
        "ticket_id": ticket["ticket_id"],
        "chunk_type": "problem_resolution",
        "text": problem_text,
        "linked_resolution": resolution_text,
        "metadata": json.dumps(metadata),
    }


def insert_chunk(db_path: str, chunk: dict):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO chunks (ticket_id, chunk_type, text, linked_resolution, metadata)
            VALUES (:ticket_id, :chunk_type, :text, :linked_resolution, :metadata)
        """, chunk)


def main():
    db = TicketDatabase("tickets.db")
    init_chunks_table("chunks.db")

    print(db.db_path)
    import os
    print(os.path.abspath(db.db_path))
    ticket_id = 496665198  # or whatever you confirmed has messages
    print(repr(ticket_id), type(ticket_id))
    result = db.get_ticket_with_messages(ticket_id)
    print(result["ticket"]["ticket_id"] if result else "None")
    print(len(result["messages"]) if result else "N/A")

    skipped = 0
    chunked = 0

    tickets = db.get_unembedded_tickets(limit=10000)
    print(f"Found {len(tickets)} tickets to chunk...")

    for ticket in tickets:
        ticket_id = ticket["ticket_id"]
        full = db.get_ticket_with_messages(ticket_id)
        chunk = build_chunk(full)
        if chunk is None:
            print(f"  Skipping ticket {ticket_id}: no usable customer/agent messages")
            skipped += 1
            continue

        insert_chunk("chunks.db", chunk)
        db.mark_embedded(ticket_id)
        chunked += 1

    print(f"Done. Chunked {chunked} tickets, skipped {skipped}.")


if __name__ == "__main__":
    main()