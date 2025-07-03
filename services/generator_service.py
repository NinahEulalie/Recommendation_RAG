from config.config import get_openai_client

# Génération reéponse (Generator)
def generate_response(context: str, question: str, model="gpt-4o-mini"):
    client = get_openai_client()
    full_prompt = f"Contexte:\n{context}\n\nQuestion:\n{question}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content