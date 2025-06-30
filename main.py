from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils_rag import get_embedding, retrieve_similar_texts, generate_response
import openai
import numpy as np

app = FastAPI()

class EmbeddingCVandPosteInput(BaseModel):
    offre: str
    cvs: list[str]

class RetrieverInput(BaseModel):
    offre_embedding: list[float]
    cvs: list[str]
    cvs_embedding: list[list[float]]
    top_k: int = 5

class GeneratorInput(BaseModel):
    context: str
    question: str

class AnalyseInput(BaseModel):
    offre: str
    cvs: list[str]
    question: str = "Parmis ces CVs, quels sont les meilleurs profils ?"
    top_k: int = 5

@app.post("/embedding")
async def embed_cvs_and_offre(input_data: EmbeddingCVandPosteInput):
    # Fusion offre et CVs pour un appel batch
    texts = [input_data.offre] + input_data.cvs
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeddings = [e.embedding for e in response.data]
    
    # Séparation embedding offre et CVs
    offre_embedding = embeddings[0]
    cvs_embedding = embeddings[1:]
    
    return {
        "offre_embedding": offre_embedding,
        "cvs_embedding": cvs_embedding
    }


@app.post("/retriever")
async def retrieve_similar(input_data: RetrieverInput):
    # Conversion en matrices numpy
    offre_vector = np.array(input_data.offre_embedding).reshape(1, -1)
    cvs_vector = np.array(input_data.cvs_embedding)
    
    # Similarités cosinus
    similarities = cosine_similarity(offre_vector, cvs_vector)[0]
    top_indices = np.argsort(similarities)[-input_data.top_k:][::-1]

    # Retourner les meilleurs CVs avec leur score
    top_cvs = [
        {
            "cv": input_data.cvs[i], 
            "score": float(similarities[i])
        } 
        for i in top_indices
    ]
    return {
        "top_cvs": top_cvs
    }


@app.post("/generator")
async def generate_answer(input_data: GeneratorInput):
    result = generate_response(input_data.context, input_data.question)
    return {
        "generated_response": result
    }


@app.post("/analyse")
async def analyse_cv(input_data: AnalyseInput):
    texts = [input_data.offre] + input_data.cvs
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeddings = [e.embedding for e in response.data]
    
    offre_embedding = embeddings[0]
    cvs_embedding = embeddings[1:]

    similarities = cosine_similarity([offre_embedding], cvs_embedding)[0]
    top_indices = np.argsort(similarities)[-input_data.top_k:][::-1]

    top_cvs = [input_data.cvs[i] for i in top_indices]
    context = "\n\n".join(top_cvs)

    generated = generate_response(context, input_data.question)
    return {
        "top_cvs": 
        [
            {   
                "cv": input_data.cvs[i], 
                "score": float(similarities[i])
            }
            for i in top_indices
        ],
        "generated_summary": generated
    }