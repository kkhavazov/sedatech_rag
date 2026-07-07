from main import TicketDatabase
from config import number_of_tickets
import os
import requests
from dotenv import load_dotenv

load_dotenv()
EDESK_KEY = os.getenv("EDESK_KEY")

db = TicketDatabase("tickets.db")

number_of_pages = (number_of_tickets + 19) // 20

for page in range(1, number_of_pages + 1):
    print(f"Fetching page {page} of {number_of_pages}...")
    response = requests.get(
        f"https://api.edesk.com/v1/tickets?filter_status_equals=Closed&page={page}",
        headers={"Authorization": EDESK_KEY}
    )
    tickets = response.json()["data"]
    for ticket in tickets:
        db.insert_ticket(ticket)
        ticket_id = ticket["id"]
        ticket_response = requests.get(
            f"https://api.edesk.com/v1/tickets/{ticket_id}?include=messages",
            headers={"Authorization": EDESK_KEY}
        )
        if ticket_response.status_code != 200:
            print(f"Failed to fetch ticket {ticket_id}. Status code: {ticket_response.status_code}")
            continue
        messages_ids = ticket_response.json()["data"]["messages_ids"]
        for message_id in messages_ids:
            message_response = requests.get(
                f"https://api.edesk.com/v1/messages/{message_id}",
                headers={"Authorization": EDESK_KEY}
            )
            if message_response.status_code != 200:
                print(f"Failed to fetch message {message_id} for ticket {ticket_id}. Status code: {message_response.status_code}")
                continue
            message = message_response.json()["data"]
            if message["direction"] == "Incoming":
                role = "Customer"
            elif message["direction"] == "Outgoing":
                role = "Agent"
            else:
                continue  # Skip messages with unknown direction
            db.insert_message(
                ticket_id=ticket_id,
                message_id=message["id"],
                sender_role=role,
                body=message["body"],
                created_at=message.get("created_at"),
                is_solution=message.get("is_solution", False)
            )