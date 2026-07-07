from ollama import Client
from typing import List
from config import ollama_address

def create_embedding(client: Client, text: str) -> List[float]:
    """
    Given a text, return its embedding as a list of floats.
    """
    client = Client(host = ollama_address)
    response = client.embeddings(
        model="bge-m3",
        input=text
    )
    return response.data[0].embedding

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Given a list of texts, return a list of embeddings.
    """
    client = Client()
    embeddings = []
    for text in texts:
        embedding = get_embedding(client, text)
        embeddings.append(embedding)
    return embeddings