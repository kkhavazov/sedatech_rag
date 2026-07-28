import sqlite3
import json
from datetime import datetime
from typing import Optional
import json

class TicketDatabase:
    def __init__(self, db_path: str = "tickets.db"):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id INTEGER PRIMARY KEY,
                    subject TEXT,
                    status TEXT,
                    type TEXT,
                    language TEXT,
                    created_at TEXT,
                    last_updated_at TEXT,
                    ai_classification TEXT,
                    tags_ids TEXT,          -- JSON list, e.g. "[1056358, 1056360]"
                    replies INTEGER,
                    uri TEXT,
                    contact_id INTEGER,
                    external_order_id TEXT,
                    order_status TEXT,
                    product_title TEXT,
                    product_sku TEXT,
                    embedded INTEGER DEFAULT 0   -- flag: has this ticket been chunked/embedded yet
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY,
                    ticket_id INTEGER NOT NULL,
                    sender_role TEXT,        -- 'customer' or 'agent'
                    body TEXT,
                    created_at TEXT,
                    is_solution INTEGER DEFAULT 0,
                    language TEXT,
                    FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_ticket_id ON messages (ticket_id)"
            )

    def insert_ticket(self, raw_ticket: dict):
        """Extract and store only the fields needed for RAG from a raw ticket payload."""
        sales_order = raw_ticket.get("sales_order") or {}
        order_items = sales_order.get("order_items") or []
        first_item = order_items[0] if order_items else {}
        product = first_item.get("product") or {}

        row = {
            "ticket_id": raw_ticket["id"],
            "subject": raw_ticket.get("subject"),
            "status": raw_ticket.get("status"),
            "type": raw_ticket.get("type"),
            "language": raw_ticket.get("language"),
            "created_at": raw_ticket.get("created_at"),
            "last_updated_at": raw_ticket.get("last_message_created_at"),
            "ai_classification": raw_ticket.get("ai_classification"),
            "tags_ids": json.dumps(raw_ticket.get("tags_ids") or []),
            "replies": raw_ticket.get("replies"),
            "uri": raw_ticket.get("uri"),
            "contact_id": raw_ticket.get("contact_id"),
            "external_order_id": raw_ticket.get("external_order_id"),
            "order_status": sales_order.get("status"),
            "product_title": product.get("title"),
            "product_sku": product.get("sku"),
        }

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO tickets (
                    ticket_id, subject, status, type, language, created_at, last_updated_at,
                    ai_classification, tags_ids, replies, uri, contact_id,
                    external_order_id, order_status, product_title, product_sku
                ) VALUES (
                    :ticket_id, :subject, :status, :type, :language, :created_at, :last_updated_at,
                    :ai_classification, :tags_ids, :replies, :uri, :contact_id,
                    :external_order_id, :order_status, :product_title, :product_sku
                )
                ON CONFLICT(ticket_id) DO UPDATE SET
                    status=excluded.status,
                    last_updated_at=excluded.last_updated_at,
                    ai_classification=excluded.ai_classification,
                    replies=excluded.replies,
                    order_status=excluded.order_status
            """, row)

    def insert_message(
        self,
        ticket_id: int,
        message_id: int,
        sender_role: str,
        body: str,
        created_at: Optional[str] = None,
        is_solution: bool = False,
        language: Optional[str] = None
    ):
        """Store a single message body once fetched from the messages endpoint."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO messages (message_id, ticket_id, sender_role, body, created_at, is_solution, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    body=excluded.body,
                    is_solution=excluded.is_solution,
                    language=excluded.language
            """, (message_id, ticket_id, sender_role, body, created_at, int(is_solution), language))

    def get_ticket_with_messages(self, ticket_id: int) -> Optional[dict]:
        with self._connect() as conn:
            ticket = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if not ticket:
                return None
            messages = conn.execute(
                "SELECT * FROM messages WHERE ticket_id = ? ORDER BY created_at",
                (ticket_id,)
            ).fetchall()
            return {
                "ticket": dict(ticket),
                "messages": [dict(m) for m in messages],
            }

    def get_unembedded_tickets(self, limit: int = 100) -> list[dict]:
        """Tickets not yet chunked/embedded — used to drive the ingestion pipeline."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE embedded = 0 LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_embedded(self, ticket_id: int):
        with self._connect() as conn:
            conn.execute(
                "UPDATE tickets SET embedded = 1 WHERE ticket_id = ?", (ticket_id,)
            )


if __name__ == "__main__":
    db = TicketDatabase("tickets.db")
    message_id = 1000
    tickets = json.load(open("data/tickets.json"))
    for ticket in tickets:
        db.insert_ticket(ticket)
        print(f"ticket inserted: {ticket["id"]}")
        for message in ticket.get("messages", []):
            if message["is_incoming"] == "1":
                role = "Customer"
            elif message["is_incoming"] == "0":
                role = "Agent"
            else:
                continue
            db.insert_message(
                ticket_id=ticket["id"],
                message_id=message_id,
                sender_role=role,
                body=message["message_body"],
                created_at=message.get("created_at"),
                is_solution=message.get("is_solution", False),
                language=message.get("language")
            )
            message_id += 1

