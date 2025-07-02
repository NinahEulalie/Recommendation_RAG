from openai import OpenAI
from core.settings import get_openai_client

def embed_texts(texts: list[str], model="text-embedding-3-small"):
    client = get_openai_client()
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]