import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Embedding du texte
def get_embedding(text: str, model="text-embedding-3-small") -> np.ndarray:
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

# Recherche contextuelle (Retriever)
def retrieve_similar_texts(query_embed: np.ndarray, documents: list[str], top_k=3) -> list[str]:
    doc_embeddings = [get_embedding(doc) for doc in documents]
    similarities = cosine_similarity([query_embed], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# Génération reéponse (Generator)
def generate_response(context: str, prompt: str, model="gpt-4o-mini") -> str:
    full_prompt = f"Contexte:\n{context}\n\nQuestion:\n{prompt}"
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content