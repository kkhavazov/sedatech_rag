from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import httpx
import asyncio
from pydantic import BaseModel
import os
from llm_requests import gemini_call, reprompt_call
from database import (
    get_cached_draft,
    get_cached_message_ids,
    get_cached_messages,
    get_ticket_revision,
    initialize_database,
    store_draft,
    store_ticket_messages,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(initialize_database)
    yield


app = FastAPI(lifespan=lifespan)


try:
    from dotenv import load_dotenv
    load_dotenv()  
except ImportError:
    pass



EDESK_API_KEY = os.getenv("EDESK_API_KEY")
if not EDESK_API_KEY:
    raise RuntimeError("EDESK_API_KEY is not configured")
token = EDESK_API_KEY.strip()
headers = {
        "accept": "application/json",
        "authorization": token
    }
class CleanMessage(BaseModel):
    role: str
    text: str

MAX_CONCURRENT_REQUESTS = 5


async def fetch_message(
    client: httpx.AsyncClient,
    message_id: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    url = f"https://api.edesk.com/v1/messages/{message_id}"

    try:
        async with semaphore:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        data = response.json()["data"]

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"eDesk rejected message {message_id}: "
                f"{exc.response.status_code} "
                f"{exc.response.text[:300]}"
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not retrieve eDesk message {message_id}: {exc}",
        ) from exc

    if data["direction"] == "Incoming":
        role = "Customer"
    elif data["direction"] == "Outgoing":
        role = "Sedatech Support"
    else:
        # Cache non-conversation events too, so they are not fetched repeatedly.
        return {"role": "", "text": "", "visible": False}

    return {
        "role": role,
        "text": data.get("body", ""),
        "visible": True,
    }

async def fetch_messages_by_id(
    message_ids: list[str],
    client: httpx.AsyncClient,
) -> dict[str, dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = [
        fetch_message(client, message_id, semaphore)
        for message_id in message_ids
    ]

    results = await asyncio.gather(*tasks)

    return {
        message_id: message
        for message_id, message in zip(message_ids, results)
    }


async def get_remote_ticket(client: httpx.AsyncClient, ticket_id: str) -> dict:
    try:
        response = await client.get(
            f"https://api.edesk.com/v1/tickets/{ticket_id}",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()["data"]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"eDesk ticket request failed: {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach eDesk: {exc}") from exc


async def get_ticket_messages(
    client: httpx.AsyncClient,
    ticket_id: str,
    raw_ticket: dict,
) -> tuple[list[dict], bool, str]:
    message_ids = [str(message_id) for message_id in raw_ticket.get("messages_ids", [])]
    if not message_ids:
        raise HTTPException(status_code=404, detail="Ticket has no messages")

    last_message_id = message_ids[-1]
    cached_revision = await asyncio.to_thread(get_ticket_revision, ticket_id)

    if cached_revision == last_message_id:
        cached = await asyncio.to_thread(get_cached_messages, ticket_id, message_ids)
        if cached is not None:
            return cached, True, last_message_id

    cached_ids = await asyncio.to_thread(get_cached_message_ids, ticket_id, message_ids)
    missing_ids = [message_id for message_id in message_ids if message_id not in cached_ids]
    if missing_ids:
        fetched = await fetch_messages_by_id(missing_ids, client)
        await asyncio.to_thread(store_ticket_messages, ticket_id, message_ids, fetched)

    messages = await asyncio.to_thread(get_cached_messages, ticket_id, message_ids)
    if messages is None:
        raise HTTPException(status_code=502, detail="Could not build complete ticket history")

    # Update the ticket revision even if every individual message was already cached.
    await asyncio.to_thread(store_ticket_messages, ticket_id, message_ids, {})
    return messages, False, last_message_id

class TicketPostResponseBody(BaseModel):
    text: str
    type: str


@app.get("/tickets/{ticket_id}")
async def get_processed_ticket(ticket_id: str):
    timeout = httpx.Timeout(30.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        raw_ticket = await get_remote_ticket(client, ticket_id)
        formatted_messages, cache_hit, last_message_id = await get_ticket_messages(
            client, ticket_id, raw_ticket
        )

    return {
        "ticket_id": ticket_id,
        "messages": formatted_messages,
        "last_message_id": last_message_id,
        "cache_hit": cache_hit,
    }

@app.get("/tickets/{ticket_id}/llm_response")
async def get_llm_response(ticket_id: str):
    timeout = httpx.Timeout(30.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        raw_ticket = await get_remote_ticket(client, ticket_id)
        formatted_messages, messages_cache_hit, last_message_id = await get_ticket_messages(
            client, ticket_id, raw_ticket
        )

    cached_draft = await asyncio.to_thread(
        get_cached_draft, ticket_id, last_message_id
    )
    if cached_draft is not None:
        return {
            "ticket_id": ticket_id,
            "draft_response": cached_draft,
            "last_message_id": last_message_id,
            "cache_hit": True,
        }

    draft_text = await asyncio.to_thread(
        gemini_call,
        formatted_messages[-1]["text"],
        formatted_messages[:-1],
    )
    await asyncio.to_thread(store_draft, ticket_id, last_message_id, draft_text)

    return {
        "ticket_id": ticket_id,
        "draft_response": draft_text,
        "last_message_id": last_message_id,
        "cache_hit": False,
        "messages_cache_hit": messages_cache_hit,
    }


@app.get("/tickets/{ticket_id}/cache-status")
async def get_cache_status(ticket_id: str):
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        raw_ticket = await get_remote_ticket(client, ticket_id)

    message_ids = [str(message_id) for message_id in raw_ticket.get("messages_ids", [])]
    remote_last_message_id = message_ids[-1] if message_ids else None
    cached_last_message_id = await asyncio.to_thread(get_ticket_revision, ticket_id)

    return {
        "ticket_id": ticket_id,
        "is_current": bool(remote_last_message_id)
        and cached_last_message_id == remote_last_message_id,
        "remote_last_message_id": remote_last_message_id,
        "cached_last_message_id": cached_last_message_id,
    }

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


class TicketRepromptResponseBody(BaseModel):
    instructions: str
    last_response: str

@app.post("/tickets/{ticket_id}/reprompt")
async def post_reprompt(ticket_id, body: TicketRepromptResponseBody):
    timeout = httpx.Timeout(30.0, connect=5.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        raw_ticket = await get_remote_ticket(client, ticket_id)
        formatted_messages, messages_cache_hit, last_message_id = await get_ticket_messages(
            client, ticket_id, raw_ticket
        )

    draft_text = await asyncio.to_thread(
        reprompt_call,
        body.instructions,
        body.last_response
    )
    await asyncio.to_thread(store_draft, ticket_id, last_message_id, draft_text)

    return {
        "ticket_id": ticket_id,
        "draft_response": draft_text,
        "last_message_id": last_message_id,
        "cache_hit": False,
        "messages_cache_hit": messages_cache_hit,
    }