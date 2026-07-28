import requests
from fastapi import FastAPI, HTTPException, Request
import httpx
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
from llm_requests import gemini_call
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
import hashlib

app = FastAPI()


try:
    from dotenv import load_dotenv
    load_dotenv()  
except ImportError:
    pass



EDESK_API_KEY = os.getenv("EDESK_API_KEY")
token = EDESK_API_KEY.strip()
headers = {
        "accept": "application/json",
        "authorization": token
    }
class CleanMessage(BaseModel):
    role: str
    text: str

def transform_raw_ticket_to_llm_format(raw_data):
    clean_history = []

    for message_id in raw_data["messages_ids"]:
        url = f"https://api.edesk.com/v1/messages/{message_id}"
        response = requests.get(url, headers=headers)
        direction = response.json()["data"]["direction"]
        if direction == "Incoming":
            clean_history.append(CleanMessage(
                    role="Customer", 
                    text=response.json()["data"]["body"]
                ))
        elif direction == "Outgoing":
            clean_history.append(CleanMessage(
                    role="Sedatech Support", 
                    text=response.json()["data"]["body"]
                ))
        else:
            continue
    return clean_history

class TicketPostResponseBody(BaseModel):
    text: str
    type: str


@app.get("/tickets/{ticket_id}")
async def get_processed_ticket(ticket_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.edesk.com/v1/tickets/{ticket_id}", headers = headers)
        
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Internal API Errors")

    raw_json = response.json()
    formatted_messages = transform_raw_ticket_to_llm_format(raw_json["data"])
    
    return {"ticket_id": ticket_id, "messages": formatted_messages}

@app.get("/tickets/{ticket_id}/llm_response")
async def get_llm_response(ticket_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.edesk.com/v1/tickets/{ticket_id}", headers = headers)
        
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Internal API Errors")

    raw_json = response.json()
    formatted_messages = transform_raw_ticket_to_llm_format(raw_json["data"])
    print(type(formatted_messages))



    draft_text = gemini_call(last_message=formatted_messages[-1].text, history=formatted_messages[:-1])
    
    return {"ticket_id": ticket_id, "draft_response": draft_text}

@app.post("/tickets/{ticket_id}/response")
async def post_response(ticket_id: str, body: TicketPostResponseBody):
    if body.type not in ["Note", "Message"]:
        raise HTTPException(status_code=400, detail="Invalid response type. Must be 'Note' or 'Public'.")

    url = f"https://api.edesk.com/v1/messages"
    if body.type == "Note":
        payload = {
            "type": "Note",
            "ticket_id": ticket_id,
            "body": body.text
        }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": token
    }
    
    response = httpx.post(url, json=payload, headers=headers)
    return {"message": f"Response for ticket {ticket_id} sent successfully!, Status code: {response.status_code}", "Error": {response.text} if response.status_code != 200 else None}