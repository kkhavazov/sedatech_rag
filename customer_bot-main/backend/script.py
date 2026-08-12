"""Periodically pre-generate cached drafts for all pending eDesk tickets."""

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv


load_dotenv()

EDESK_API_KEY = os.environ["EDESK_API_KEY"].strip()
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "").strip()
BACKEND_TICKETS_URL = os.getenv(
    "BACKEND_TICKETS_URL", "http://127.0.0.1:8000/tickets"
).rstrip("/")
RUN_INTERVAL_SECONDS = int(os.getenv("CACHE_WARM_INTERVAL_SECONDS", "3600"))
MAX_CONCURRENT_GENERATIONS = int(os.getenv("CACHE_WARM_CONCURRENCY", "1"))

EDESK_OPEN_TICKETS_URL = (
    "https://api.edesk.com/v1/tickets?filter_status_equals=Pending"
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cache-warmer")
logging.getLogger("httpx").setLevel(logging.WARNING)


async def wait_for_backend(client: httpx.AsyncClient) -> None:
    health_url = f"{BACKEND_TICKETS_URL.rsplit('/tickets', 1)[0]}/openapi.json"
    while True:
        try:
            response = await client.get(health_url, timeout=5.0)
            response.raise_for_status()
            return
        except httpx.HTTPError as exc:
            logger.warning(
                "Backend is not ready (%s: %r); retrying in 2 seconds",
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(2)


async def get_open_ticket_ids(client: httpx.AsyncClient) -> list[str]:
    response = await client.get(
        EDESK_OPEN_TICKETS_URL,
        headers={
            "accept": "application/json",
            "authorization": EDESK_API_KEY,
        },
    )
    response.raise_for_status()
    return [str(ticket["id"]) for ticket in response.json().get("data", [])]


async def generate_ticket_draft(
    client: httpx.AsyncClient,
    ticket_id: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    headers = {"X-API-Key": INTERNAL_API_KEY} if INTERNAL_API_KEY else {}

    try:
        async with semaphore:
            response = await client.get(
                f"{BACKEND_TICKETS_URL}/{ticket_id}/llm_response",
                headers=headers,
            )
            response.raise_for_status()

        result = response.json()
        logger.info(
            "Ticket %s ready (cache_hit=%s, last_message_id=%s)",
            ticket_id,
            result.get("cache_hit"),
            result.get("last_message_id"),
        )
        return True
    except (httpx.HTTPError, ValueError) as exc:
        response_text = ""
        if isinstance(exc, httpx.HTTPStatusError):
            response_text = f" response={exc.response.text[:500]}"
        logger.error("Ticket %s failed: %s%s", ticket_id, exc, response_text)
        return False


async def warm_cache_once(client: httpx.AsyncClient) -> None:
    try:
        ticket_ids = await get_open_ticket_ids(client)
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Could not retrieve pending tickets: %s", exc)
        return

    if not ticket_ids:
        logger.info("No pending tickets found")
        return

    logger.info("Warming cache for %d pending tickets", len(ticket_ids))
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    results = await asyncio.gather(
        *(
            generate_ticket_draft(client, ticket_id, semaphore)
            for ticket_id in ticket_ids
        )
    )
    logger.info(
        "Cache run complete: %d succeeded, %d failed",
        sum(results),
        len(results) - sum(results),
    )


async def main() -> None:
    if RUN_INTERVAL_SECONDS <= 0:
        raise ValueError("CACHE_WARM_INTERVAL_SECONDS must be greater than zero")
    if MAX_CONCURRENT_GENERATIONS <= 0:
        raise ValueError("CACHE_WARM_CONCURRENCY must be greater than zero")

    timeout = httpx.Timeout(300.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            await wait_for_backend(client)
            await warm_cache_once(client)
            logger.info("Next cache run in %d seconds", RUN_INTERVAL_SECONDS)
            await asyncio.sleep(RUN_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Cache warmer stopped")
