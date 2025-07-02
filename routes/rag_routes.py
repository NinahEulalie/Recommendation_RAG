from fastapi import APIRouter
from models.schema import *
from services.embedding_service import embed_texts
from services.retriever_service import retrieve_similar
from services.generator_service import generate_response

router = APIRouter()

@router.post("/embedding")
def embed_batch(data: EmbeddingCVandPosteInput):
    texts = [data.offre] + data.cvs
    embeddings = embed_texts(texts)
    return {
        "offre_embedding": embeddings[0],
        "cvs_embedding": embeddings[1:]
    }

@router.post("/retriever")
def retrieve(data: RetrieverInput):
    return {
        "top_cvs": retrieve_similar(data.offre_embedding, data.cvs_embedding, data.cvs, data.top_k)
    }

@router.post("/generator")
def generator(data: GeneratorInput):
    result = generate_response(data.context, data.question)
    return {"generated_response": result}

@router.post("/analyse")
def analyse(data: AnalyseInput):
    texts = [data.offre] + data.cvs
    embeddings = embed_texts(texts)
    offre_emb = embeddings[0]
    cvs_emb = embeddings[1:]
    top_cvs = retrieve_similar(offre_emb, cvs_emb, data.cvs, data.top_k)
    context = "\n\n".join(cv["cv"] for cv in top_cvs)
    summary = generate_response(context, data.question)
    return {
        "top_cvs": top_cvs,
        "generated_summary": summary
    }
