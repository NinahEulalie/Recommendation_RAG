from openai import OpenAI
import numpy as np
from config.config import get_openai_client

# Embedding du texte
def get_embedding(texts: list[str], model="text-embedding-3-small")-> np.ndarray:
    client = get_openai_client()
    response = client.embeddings.create(input=texts, model=model)
    return [record.embedding for record in response.data]  # retourne liste de listes de float