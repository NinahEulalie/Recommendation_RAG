from fastapi import APIRouter
from models.schema import *
from services.embedding_service import get_embedding
from services.retriever_service import retrieve_similar_texts
from services.generator_service import generate_response
import numpy as np

router = APIRouter()

@router.post("/embedding")
def embed_cvs_and_offre(input_data: EmbeddingCVandPosteInput):
    # Fusion offre et CVs pour un appel batch
    texts = [input_data.offre] + input_data.cvs
    embeddings = get_embedding(texts)  # embeddings: np.ndarray of shape (N, D)
    
    return {
        "offre_embedding": embeddings[0],       # list[float]
        "cvs_embedding": embeddings[1:]         # list[list[float]]
    }


@router.post("/retriever")
def retrieve_similar_route(input_data: RetrieverInput):
    top_cvs = retrieve_similar_texts(
        np.array(input_data.offre_embedding),
        input_data.cvs_embedding,
        input_data.cvs,
        input_data.top_k
    )
    return {"top_cvs": top_cvs}


@router.post("/generator")
def generate_answer(input_data: GeneratorInput):
    result = generate_response(input_data.context, input_data.question)
    return {
        "generated_response": result
    }

@router.post("/analyse")
def analyse(input_data: AnalyseInput):
    texts = [input_data.offre] + input_data.cvs
    embeddings = get_embedding(texts)
    offre_emb = embeddings[0]
    cvs_emb = embeddings[1:]
    top_cvs = retrieve_similar_texts(offre_emb, cvs_emb, input_data.cvs, input_data.top_k)
    context = "\n\n".join(cv["cv"] for cv in top_cvs)
    summary = generate_response(context, input_data.question)
    return {
        "top_cvs": top_cvs,
        "generated_summary": summary
    }
